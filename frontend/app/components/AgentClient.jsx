"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"

function normalizeBaseUrl(raw) {
  const s = String(raw || "").trim()
  if (!s) return ""
  if (s.startsWith("http://") || s.startsWith("https://")) return s.replace(/\/+$/, "")
  const host = s.replace(/^\/\//, "").split("/")[0] || ""
  const isLocal =
    host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0") || host.endsWith(".local")
  const proto = isLocal ? "http://" : "https://"
  return `${proto}${s.replace(/^\/\//, "")}`.replace(/\/+$/, "")
}

function isImageFileName(name) {
  const n = String(name || "").toLowerCase()
  return n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".webp") || n.endsWith(".gif") || n.endsWith(".bmp") || n.endsWith(".tiff")
}

export default function AgentClient() {
  const [lang, setLang] = useState("en")
  const t = (zh, en) => (lang === "zh" ? zh : en)

  const [apiBase, setApiBase] = useState(process.env.NEXT_PUBLIC_API_BASE_URL || "")
  const [apiStatus, setApiStatus] = useState("")
  const [prompt, setPrompt] = useState("")
  const [file, setFile] = useState(null)
  const [freq, setFreq] = useState("D")
  const [testSize, setTestSize] = useState("0.2")
  const [randomSeed, setRandomSeed] = useState("42")
  const [periods, setPeriods] = useState("10")
  const [timeCol, setTimeCol] = useState("")
  const [demandCol, setDemandCol] = useState("")
  const [includeAdvanced, setIncludeAdvanced] = useState(false)
  const [datasetName, setDatasetName] = useState("")
  const [datasetTheme, setDatasetTheme] = useState("")
  const [dataDescription, setDataDescription] = useState("")
  const [useLlm, setUseLlm] = useState(true)
  const [llmProvider, setLlmProvider] = useState("deepseek")
  const [llmBaseUrl, setLlmBaseUrl] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [llmApiKey, setLlmApiKey] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState(null)
  const [preview, setPreview] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState("")
  const [ocrRefineWithLlm, setOcrRefineWithLlm] = useState(true)

  useEffect(() => {
    if (typeof window !== "undefined") {
      setLlmProvider(sessionStorage.getItem("forecastpro:llmProvider") || "deepseek")
      setLlmBaseUrl(sessionStorage.getItem("forecastpro:llmBaseUrl") || "")
      setLlmModel(sessionStorage.getItem("forecastpro:llmModel") || "")
      setLlmApiKey(sessionStorage.getItem("forecastpro:llmApiKey") || "")
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function ping(base) {
      try {
        const res = await fetch(`/api/health?api_base=${encodeURIComponent(base)}`, { method: "GET" })
        return res.ok
      } catch {
        return false
      }
    }

    async function autoDetect() {
      const normalized = normalizeBaseUrl(apiBase)
      if (!normalized) {
        const candidates = ["http://localhost:8013", "http://localhost:8012", "http://localhost:8011", "http://localhost:8000"]
        for (const c of candidates) {
          if (await ping(c)) {
            if (!cancelled) {
              setApiBase(c)
              setApiStatus("Connected")
            }
            return
          }
        }
        if (!cancelled) setApiStatus("Disconnected")
        return
      }

      const ok = await ping(normalized)
      if (!cancelled) setApiStatus(ok ? "Connected" : "Disconnected")
    }

    autoDetect()
    return () => {
      cancelled = true
    }
  }, [apiBase])

  const summaryBlocks = useMemo(() => {
    if (!result) return []
    if (result?.mode === "timeseries_csv" || result?.mode === "timeseries_ocr") {
      return [
        { k: t("模式", "Mode"), v: result?.mode === "timeseries_ocr" ? t("截图 OCR 时序预测", "Screenshot OCR Time-Series Forecast") : t("CSV/Excel 时序预测", "CSV/Excel Time-Series Forecast") },
        { k: t("识别的时间列", "Detected Time Col"), v: result?.detected?.time_col || "—" },
        { k: t("识别的目标列", "Detected Target Col"), v: result?.detected?.demand_col || "—" },
        { k: t("模型", "Model"), v: result?.forecast_results?.model || "—" },
      ]
    }
    return [
      { k: t("模式", "Mode"), v: t("多模态理解与建议", "Multi-modal Understanding & Suggestions") },
      { k: t("输入类型", "Input Type"), v: result?.input_type || "—" },
    ]
  }, [result, t])

  async function parseOrThrow(res) {
    if (!res.ok) {
      const t = await res.text()
      try {
        const j = JSON.parse(t)
        const d = j?.detail
        if (d?.error) throw new Error(d?.message ? `${d.error}: ${d.message}` : d.error)
        if (j?.error) throw new Error(j?.message ? `${j.error}: ${j.message}` : j.error)
      } catch {}
      throw new Error(t || `Request Failed: ${res.status}`)
    }
    return await res.json()
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError("")
    setResult(null)
    if (!file && !String(prompt || "").trim()) {
      setError(t("请输入一些文本，或上传 CSV/Excel/截图。", "Please enter some text, or upload a CSV/Excel/screenshot."))
      return
    }
    setLoading(true)
    try {
      const f = new FormData()
      if (file) f.append("file", file)
      if (String(prompt || "").trim()) f.append("prompt", String(prompt).trim())
      f.append("freq", freq)
      f.append("test_size", String(Number(testSize) || 0.2))
      f.append("random_seed", String(Number(randomSeed) || 42))
      f.append("periods", String(Number(periods) || 4))
      if (includeAdvanced) f.append("include_advanced", "true")
      if (String(timeCol || "").trim()) f.append("time_col", String(timeCol).trim())
      if (String(demandCol || "").trim()) f.append("demand_col", String(demandCol).trim())
      if (String(datasetName || "").trim()) f.append("dataset_name", String(datasetName).trim())
      if (String(datasetTheme || "").trim()) f.append("dataset_theme", String(datasetTheme).trim())
      if (String(dataDescription || "").trim()) f.append("data_description", String(dataDescription).trim())
      f.append("language", "en")
      f.append("use_llm", useLlm ? "true" : "false")
      if (useLlm) {
        if (llmApiKey) f.append("llm_api_key", llmApiKey);
        if (llmProvider === "doubao") {
          f.append("llm_base_url", llmBaseUrl || "https://ark.cn-beijing.volces.com/api/v3");
          if (llmModel) f.append("llm_model", llmModel);
        } else if (llmProvider === "custom") {
          if (llmBaseUrl) f.append("llm_base_url", llmBaseUrl);
          if (llmModel) f.append("llm_model", llmModel);
        }
      }
      const base = normalizeBaseUrl(apiBase)
      if (base) f.append("api_base", base)
      const imageSelected = isImageFileName(file?.name || "")
      const ocrRows = Array.isArray(preview?.rows) ? preview.rows : []
      if (imageSelected && preview?.source === "ocr_image" && ocrRows.length > 0) {
        f.append("ocr_confirmed", "true")
        f.append("ocr_table_json", JSON.stringify(ocrRows))
      }

      const res = await fetch("/api/agent", { method: "POST", body: f })
      const json = await parseOrThrow(res)
      if (json?.mode === "ocr_preview_needed" && json?.preview) {
        setPreview(json.preview)
        const dt = json?.preview?.detected?.time_col
        const dd = json?.preview?.detected?.demand_col
        if (!String(timeCol || "").trim() && dt) setTimeCol(String(dt))
        if (!String(demandCol || "").trim() && dd) setDemandCol(String(dd))
        setError("已提取截Chart格，请确认时间column/Target Column后，再次点击“Run Agent”CompleteReal Forecast。")
      }
      setResult(json)
      try {
        if (json?.mode === "timeseries_csv" || json?.mode === "timeseries_ocr") {
          sessionStorage.setItem("forecastpro:lastResult", JSON.stringify(json))
        }
      } catch {}
    } catch (err) {
      setError(String(err?.message || err || "Request Failed"))
    } finally {
      setLoading(false)
    }
  }

  async function onPreview() {
    setPreview(null)
    setPreviewError("")
    if (!file) {
      setPreviewError("Please select a CSV/Excel or screenshot file first")
      return
    }
    const name = String(file?.name || "").toLowerCase()
    if (!(name.endsWith(".csv") || name.endsWith(".xlsx") || name.endsWith(".xls") || isImageFileName(name))) {
      setPreviewError("Preview only supports CSV/Excel/Images (PNG/JPG/WEBP/GIF/BMP/TIFF)")
      return
    }
    setPreviewLoading(true)
    try {
      const f = new FormData()
      f.append("file", file)
      const base = normalizeBaseUrl(apiBase)
      if (base) f.append("api_base", base)
      if (String(timeCol || "").trim()) f.append("time_col", String(timeCol).trim())
      if (String(demandCol || "").trim()) f.append("demand_col", String(demandCol).trim())
      if (useLlm) {
        f.append("use_llm", "true");
        if (llmApiKey) f.append("llm_api_key", llmApiKey);
        if (llmProvider === "doubao") {
          f.append("llm_base_url", llmBaseUrl || "https://ark.cn-beijing.volces.com/api/v3");
          if (llmModel) f.append("llm_model", llmModel);
        } else if (llmProvider === "custom") {
          if (llmBaseUrl) f.append("llm_base_url", llmBaseUrl);
          if (llmModel) f.append("llm_model", llmModel);
        }
      }
      if (useLlm && ocrRefineWithLlm && isImageFileName(name)) f.append("ocr_refine_with_llm", "true")
      const res = await fetch("/api/preview", { method: "POST", body: f })
      const json = await parseOrThrow(res)
      setPreview(json)
      const dt = json?.detected?.time_col
      const dd = json?.detected?.demand_col
      if (!String(timeCol || "").trim() && dt) setTimeCol(String(dt))
      if (!String(demandCol || "").trim() && dd) setDemandCol(String(dd))
    } catch (err) {
      setPreviewError(String(err?.message || err || t("预览失败", "Preview failed")))
    } finally {
      setPreviewLoading(false)
    }
  }

  return (
    <div className="panel">
      <div className="card">
        <h2>{t("通用 Forecast Agent", "Universal Forecast Agent")}</h2>
        <div className="muted" style={{ marginBottom: 10 }}>
          {t(
            "支持文本、CSV/Excel、截图输入。CSV/Excel 会直接生成 Forecast 与 Report；截图支持 OCR 抽表后做真实拟合预测。",
            "Supports text, CSV/Excel, and screenshot inputs. CSV/Excel generates Forecast and Report directly; screenshots use OCR table extraction for fitted forecasting."
          )}
        </div>
        <form onSubmit={onSubmit}>
          <div className="formGrid">
            <div className="field">
              <label>{t("界面语言", "UI Language")}</label>
              <select value={lang} onChange={(e) => setLang(e.target.value)}>
                <option value="en">English</option>
                <option value="zh">中文</option>
              </select>
            </div>
            <div className="field">
              <label>{t("API 地址", "API Base URL")}</label>
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://localhost:8013" />
              <div className="muted" style={{ marginTop: 6 }}>
                {apiStatus
                  ? `${t("后端", "Backend")}: ${
                      apiStatus === "Connected" ? t("已连接", "Connected") : apiStatus === "Disconnected" ? t("未连接", "Disconnected") : apiStatus
                    }`
                  : ""}
                {apiStatus === "Disconnected"
                  ? t("(请在项目根目录运行: python3 -m uvicorn backend.main:app --port 8000)", "(Please run in root dir: python3 -m uvicorn backend.main:app --port 8000)")
                  : ""}
              </div>
            </div>
            <div className="field">
              <label>{t("上传文件 (选填)", "Upload File (Optional)")}</label>
              <input type="file" accept=".csv,.xlsx,.xls,.png,.jpg,.jpeg,.webp,.gif" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file ? <div className="muted">{file.name}</div> : null}
            </div>
            <div className="field">
              <label>{t("数据预览 (CSV/Excel/截图OCR)", "Data Preview (CSV/Excel/Screenshot OCR)")}</label>
              <button type="button" onClick={onPreview} disabled={!file || previewLoading}>
                {previewLoading ? t("预览中...", "Previewing...") : t("预览前 30 行", "Preview Top 30 Rows")}
              </button>
              {previewError ? <div style={{ color: "#D84B4B" }}>{previewError}</div> : null}
            </div>
            <div className="field">
              <label>{t("频率", "Frequency")}</label>
              <select value={freq} onChange={(e) => setFreq(e.target.value)}>
                <option value="D">{t("日 (D)", "Daily (D)")}</option>
                <option value="W">{t("周 (W)", "Weekly (W)")}</option>
                <option value="M">{t("月 (M)", "Monthly (M)")}</option>
                <option value="Q">{t("季 (Q)", "Quarterly (Q)")}</option>
                <option value="Y">{t("年 (Y)", "Yearly (Y)")}</option>
              </select>
            </div>
            <div className="field">
              <label>{t("测试集比例", "Test Set Ratio")}</label>
              <input value={testSize} onChange={(e) => setTestSize(e.target.value)} placeholder="0.2" />
            </div>
            <div className="field">
              <label>{t("随机种子", "Random Seed")}</label>
              <input value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} placeholder="42" />
            </div>
            <div className="field">
              <label>{t("预测期数", "Forecast Periods")}</label>
              <input value={periods} onChange={(e) => setPeriods(e.target.value)} placeholder="4" />
            </div>
            <div className="field">
              <label>{t("时间列名 (选填)", "Time Column (Optional)")}</label>
              <input value={timeCol} onChange={(e) => setTimeCol(e.target.value)} placeholder="e.g.：date / Date" />
            </div>
            <div className="field">
              <label>{t("目标变量列名 (选填)", "Target Column (Optional)")}</label>
              <input value={demandCol} onChange={(e) => setDemandCol(e.target.value)} placeholder="e.g. demand" />
            </div>
            <div className="field">
              <label>{t("高级模型 (选填)", "Advanced Models (Optional)")}</label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                <input type="checkbox" checked={includeAdvanced} onChange={(e) => setIncludeAdvanced(e.target.checked)} />
                <span>{t("启用高级模型", "Enable Advanced Models")}</span>
              </label>
            </div>
            <div className="field">
              <label>{t("数据集名称 (选填)", "Dataset Name (Optional)")}</label>
              <input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="e.g. 2026Q2_Sales_East" />
            </div>
            <div className="field">
              <label>{t("业务场景/主题 (选填)", "Theme/Scenario (Optional)")}</label>
              <input value={datasetTheme} onChange={(e) => setDatasetTheme(e.target.value)} placeholder="e.g. retail_inventory" />
            </div>
            <div className="field">
              <label>{t("LLM 开关", "LLM Switch")}</label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(e) => {
                    setUseLlm(e.target.checked);
                    if (!e.target.checked) {
                      setLlmApiKey("");
                      sessionStorage.removeItem("forecastpro:llmApiKey");
                    }
                  }}
                />
                <span>{t("启用 LLM 报告分析", "Enable LLM Analysis")}</span>
              </label>
            </div>
            {useLlm && (
              <div className="field">
                <label>DeepSeek API Key ({t("必填", "Required")})</label>
                <input
                  type="password"
                  placeholder="sk-..."
                  onChange={(e) => {
                    const val = e.target.value;
                    setLlmApiKey(val);
                    sessionStorage.setItem("forecastpro:deepseekKey", val);
                  }}
                  defaultValue={typeof window !== "undefined" ? sessionStorage.getItem("forecastpro:deepseekKey") || "" : ""}
                />
                <div className="muted" style={{ marginTop: 4 }}>{t("配置后即可生成多模态 LLM 管理报告", "Configuring enables LLM Report Analysis")}</div>
              </div>
            )}
            <div className="field">
              <label>OCR Refine (Screenshot)</label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={ocrRefineWithLlm}
                  onChange={(e) => setOcrRefineWithLlm(e.target.checked)}
                  disabled={!useLlm}
                />
                <span>{t("使用 AI 修复 OCR", "Refine OCR with AI")}</span>
              </label>
            </div>
            <div className="field" style={{ gridColumn: "span 12" }}>
              <label>{t("业务描述/提示词 (选填)", "Business Description/Prompt (Optional)")}</label>
              <textarea
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={t("e.g.：这是某零食每日销量和售价数据，请帮我预测未来两周销量并给库存建议；或描述截图中的图表并给出预测方案。", "e.g.: Daily sales data for snacks, predict next 2 weeks and give inventory tips.")}
              />
            </div>
            <div className="field" style={{ gridColumn: "span 12" }}>
              <label>{t("补充说明 (选填)", "Additional Notes (Optional)")}</label>
              <textarea
                rows={2}
                value={dataDescription}
                onChange={(e) => setDataDescription(e.target.value)}
                placeholder={t("补充业务目标、约束条件或交付要求。", "Supplementary business goals, constraints, delivery requirements.")}
              />
            </div>
            {preview?.rows?.length && preview?.columns?.length ? (
              <div className="field" style={{ gridColumn: "span 12" }}>
                <label>{t("预览结果 (前 30 行)", "Preview Results (Top 30 Rows)")}</label>
                {preview?.source === "ocr_image" && preview?.llm?.enabled ? (
                  preview?.llm?.error ? (
                    <div style={{ color: "#D84B4B", marginTop: 6 }}>
                      {String(preview?.llm?.error_code) === "insufficient_balance"
                        ? t("AI 修复 OCR 失败：LLM 余额不足（已回退为原始 OCR 结果）。", "AI OCR refine failed: LLM insufficient balance (fell back to raw OCR).")
                        : `${t("AI 修复 OCR 失败（已回退为原始 OCR 结果）：", "AI OCR refine failed (fell back to raw OCR): ")}${String(preview.llm.error).slice(0, 160)}`}
                    </div>
                  ) : (
                    <div className="muted" style={{ marginTop: 6 }}>
                      {t("已使用 AI 修复 OCR。", "OCR refined with AI. ")}
                      {Array.isArray(preview?.llm?.warnings) && preview.llm.warnings.length ? `${t("提示：", "Tip: ")}${String(preview.llm.warnings[0]).slice(0, 120)}` : ""}
                    </div>
                  )
                ) : null}
                <div className="muted" style={{ marginTop: 6 }}>
                  {t("检测到列：时间", "Detected: Time")} {preview?.detected?.time_col ? String(preview.detected.time_col) : "—"} · {t("目标", "Target")}{" "}
                  {preview?.detected?.demand_col ? String(preview.detected.demand_col) : "—"}
                </div>
                <div style={{ marginTop: 10, overflow: "auto", maxHeight: 360, border: "1px solid rgba(16,50,66,0.14)", borderRadius: 12, background: "rgba(255,255,255,0.65)" }}>
                  <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                    <thead>
                      <tr>
                        {preview.columns.map((c) => (
                          <th
                            key={String(c)}
                            style={{
                              textAlign: "left",
                              padding: "8px 10px",
                              position: "sticky",
                              top: 0,
                              background: "rgba(248,250,252,0.96)",
                              borderBottom: "1px solid rgba(16,50,66,0.12)",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {String(c)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.rows.map((r, i) => (
                        <tr key={i}>
                          {preview.columns.map((c) => (
                            <td
                              key={`${i}-${String(c)}`}
                              style={{
                                padding: "7px 10px",
                                borderBottom: "1px solid rgba(16,50,66,0.06)",
                                whiteSpace: "nowrap",
                              }}
                            >
                              {r?.[c] === null || r?.[c] === undefined ? "" : String(r[c])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="submit" disabled={loading}>
              {loading ? t("分析中...", "Analyzing...") : t("运行 Agent", "Run Agent")}
            </button>
          </div>
          {error ? <div style={{ color: "#D84B4B", marginTop: 10 }}>{error}</div> : null}
        </form>
      </div>

      {result ? (
        <div className="card" style={{ marginTop: 14 }}>
          <h2>{t("分析结果", "Results")}</h2>
          <div className="row" style={{ marginBottom: 8 }}>
            {summaryBlocks.map((x) => (
              <div key={x.k} className="kpi" style={{ minWidth: 220 }}>
                <strong>{x.k}</strong>
                <span className="mono">{String(x.v || "—")}</span>
              </div>
            ))}
          </div>

          {result?.mode === "timeseries_csv" || result?.mode === "timeseries_ocr" ? (
            <>
              {result?.mode === "timeseries_ocr" && result?.ocr_meta ? (
                <div className="muted" style={{ marginBottom: 8 }}>
                  {t("OCR 质量", "OCR Quality")}: {String(result?.ocr_meta?.table_quality || t("未知", "Unknown"))} · {t("平均置信度", "Mean Confidence")}: 
                  {result?.ocr_meta?.ocr_mean_confidence === null || result?.ocr_meta?.ocr_mean_confidence === undefined
                    ? "—"
                    : String(Number(result.ocr_meta.ocr_mean_confidence).toFixed(1))}
                </div>
              ) : null}
              {result?.llm?.enabled && result?.llm?.error ? (
                <div style={{ color: "#D84B4B", marginBottom: 10 }}>
                  {String(result?.llm?.error_code) === "insufficient_balance"
                    ? t("LLM 余额不足（已自动回退到传统模型预测）。", "LLM insufficient balance (fallback to traditional forecast).")
                    : t(`LLM 调用失败（已自动回退到传统模型预测）：${String(result.llm.error).slice(0, 200)}`, `LLM call failed (fallback to traditional forecast): ${String(result.llm.error).slice(0, 200)}`)}
                </div>
              ) : result?.llm?.enabled && !result?.llm?.error ? (
                <div style={{ color: "#3B82F6", marginBottom: 10, padding: 12, background: "rgba(59,130,246,0.05)", borderRadius: 6 }}>
                  {t("LLM 分析调用成功。", "LLM Analysis successful.")}
                </div>
              ) : null}

              <div className="row" style={{ marginBottom: 10 }}>
                <Link className="topLink" href="/forecast">
                  {t("打开 Forecast 图表", "Open Forecast Chart")}
                </Link>
                <Link className="topLink" href="/report">
                  {t("打开 Report 报告", "Open Report")}
                </Link>
              </div>

              <div className="muted">{t("未来预测（前 12 期）", "Future Forecast (Top 12)")}</div>
              <div style={{ overflowX: "auto", marginTop: 8 }}>
                <table className="reportTable">
                  <thead>
                    <tr>
                      <th>{t("日期", "Date")}</th>
                      <th>{t("预测值", "Forecast")}</th>
                      <th>{t("95%下限", "95% Lower")}</th>
                      <th>{t("95%上限", "95% Upper")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result?.forecast_results?.dates || []).slice(0, 12).map((d, i) => (
                      <tr key={`${d}_${i}`}>
                        <td>{String(d).slice(0, 10)}</td>
                        <td>{String(result?.forecast_results?.forecast?.[i] ?? "—")}</td>
                        <td>{String(result?.forecast_results?.lower_bound?.[i] ?? "—")}</td>
                        <td>{String(result?.forecast_results?.upper_bound?.[i] ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {Array.isArray(result?.evaluation_results) && result.evaluation_results.length ? (
                <>
                  <div style={{ height: 12 }} />
                  <div className="muted">{t("模型对比（前 8 名）", "Model Comparison (Top 8)")}</div>
                  <div style={{ overflowX: "auto", marginTop: 8 }}>
                    <table className="reportTable">
                      <thead>
                        <tr>
                          {["model", "MAE", "RMSE", "MAPE", "overfitting_risk"].map((h) => (
                            <th key={h}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.evaluation_results.slice(0, 8).map((r, i) => (
                          <tr key={i}>
                            <td className="mono">{String(r?.model ?? "")}</td>
                            <td>{String(r?.MAE ?? "")}</td>
                            <td>{String(r?.RMSE ?? "")}</td>
                            <td>{String(r?.MAPE ?? "")}</td>
                            <td>{String(r?.overfitting_risk ?? "")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : null}
            </>
          ) : (
            <div style={{ marginTop: 8 }}>
              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0 }}>
                {JSON.stringify(result?.llm_result || {}, null, 2)}
              </pre>
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}


