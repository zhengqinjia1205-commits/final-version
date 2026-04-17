"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"

function fmtDate(v) {
  if (!v) return "—"
  const s = String(v)
  return s.length >= 10 ? s.slice(0, 10) : s
}

function fmtNum(v, digits = 3) {
  if (v === null || v === undefined) return "—"
  const n = Number(v)
  if (!Number.isFinite(n)) return "—"
  return n.toFixed(digits)
}

function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function buildTimelineIndex(dates) {
  const list = (dates || [])
    .map((d) => {
      const t = new Date(d).getTime()
      return Number.isFinite(t) ? t : null
    })
    .filter((t) => t !== null)
  const uniq = Array.from(new Set(list)).sort((a, b) => a - b)
  const map = new Map()
  uniq.forEach((t, i) => map.set(t, i))
  return { uniq, map }
}

function computeDomain(seriesList) {
  const vals = []
  for (const s of seriesList) {
    for (const v of s || []) {
      const n = Number(v)
      if (Number.isFinite(n)) vals.push(n)
    }
  }
  if (!vals.length) return { min: 0, max: 1 }
  let min = Math.min(...vals)
  let max = Math.max(...vals)
  if (min === max) {
    min -= 1
    max += 1
  }
  return { min, max }
}

function xForIndex(idx, width, pad, totalPoints) {
  const innerW = width - pad * 2
  const denom = Math.max(totalPoints - 1, 1)
  return pad + (innerW * idx) / denom
}

function yForValue(v, domain, height, pad) {
  const innerH = height - pad * 2
  const min = domain?.min ?? 0
  const max = domain?.max ?? 1
  const span = max - min || 1
  return pad + innerH - (innerH * (Number(v) - min)) / span
}

function buildPolyline(values, indices, domain, width, height, pad, totalPoints) {
  if (!values || !indices || values.length < 2) return ""
  return values
    .map((v, i) => {
      const idx = indices[i]
      const x = xForIndex(idx, width, pad, totalPoints)
      const y = yForValue(v, domain, height, pad)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}

function buildBandPath(upper, lower, indices, domain, width, height, pad, totalPoints) {
  if (!upper || !lower || !indices || upper.length < 2 || lower.length < 2) return ""
  const top = upper.map((v, i) => {
    const idx = indices[i]
    const x = xForIndex(idx, width, pad, totalPoints)
    const y = yForValue(v, domain, height, pad)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const bot = lower
    .slice()
    .reverse()
    .map((v, j) => {
      const i = lower.length - 1 - j
      const idx = indices[i]
      const x = xForIndex(idx, width, pad, totalPoints)
      const y = yForValue(v, domain, height, pad)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
  return `M ${top[0]} L ${top.slice(1).join(" L ")} L ${bot.join(" L ")} Z`
}

function ReportChart({ chartData }) {
  const width = 980
  const height = 320
  const pad = 22

  const { timeline, hist, fit, comb, splitIdx, domain } = useMemo(() => {
    const h = chartData?.history || {}
    const c = chartData?.combined_forecast || {}
    const hd = h?.dates || []
    const ha = h?.actual || []
    const hf = h?.fitted_best || []

    const cd = c?.dates || []
    const cf = c?.forecast || []
    const clo = c?.lower_bound || []
    const cup = c?.upper_bound || []

    const split = chartData?.split_date || null

    const { uniq, map } = buildTimelineIndex([...(hd || []), ...(cd || [])])
    const toIdx = (d) => {
      const t = new Date(d).getTime()
      return map.has(t) ? map.get(t) : null
    }

    const histIdx = []
    const histVals = []
    const fitIdx = []
    const fitVals = []
    for (let i = 0; i < Math.min(hd.length, ha.length); i++) {
      const idx = toIdx(hd[i])
      const v = Number(ha[i])
      if (idx === null || !Number.isFinite(v)) continue
      histIdx.push(idx)
      histVals.push(v)
      const fv = Number(hf?.[i])
      if (Number.isFinite(fv)) {
        fitIdx.push(idx)
        fitVals.push(fv)
      }
    }

    const combIdx = []
    const combVals = []
    const upperVals = []
    const lowerVals = []
    for (let i = 0; i < Math.min(cd.length, cf.length); i++) {
      const idx = toIdx(cd[i])
      const v = Number(cf[i])
      if (idx === null || !Number.isFinite(v)) continue
      combIdx.push(idx)
      combVals.push(v)
      const uv = Number(cup?.[i])
      const lv = Number(clo?.[i])
      upperVals.push(Number.isFinite(uv) ? uv : v)
      lowerVals.push(Number.isFinite(lv) ? lv : v)
    }

    const sIdx = split ? toIdx(split) : null
    const d = computeDomain([histVals, fitVals, combVals, upperVals, lowerVals])
    return {
      timeline: uniq,
      hist: { idx: histIdx, vals: histVals },
      fit: { idx: fitIdx, vals: fitVals },
      comb: { idx: combIdx, vals: combVals, upper: upperVals, lower: lowerVals },
      splitIdx: sIdx,
      domain: d,
    }
  }, [chartData])

  if (!timeline?.length) return null

  const totalPoints = timeline.length
  const histLine = buildPolyline(hist.vals, hist.idx, domain, width, height, pad, totalPoints)
  const fitLine = buildPolyline(fit.vals, fit.idx, domain, width, height, pad, totalPoints)
  const combLine = buildPolyline(comb.vals, comb.idx, domain, width, height, pad, totalPoints)
  const band = buildBandPath(comb.upper, comb.lower, comb.idx, domain, width, height, pad, totalPoints)

  const splitX = splitIdx !== null && splitIdx !== undefined ? xForIndex(splitIdx, width, pad, totalPoints) : null

  return (
    <div className="svgwrap" style={{ background: "rgba(255,255,255,0.72)" }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
        {band ? <path d={band} fill="rgba(120,120,120,0.25)" stroke="none" /> : null}
        {histLine ? <polyline points={histLine} fill="none" stroke="#0B1F3B" strokeWidth="2.6" /> : null}
        {fitLine ? <polyline points={fitLine} fill="none" stroke="#FF8C00" strokeWidth="2.2" strokeDasharray="6 6" /> : null}
        {combLine ? <polyline points={combLine} fill="none" stroke="#2E8B57" strokeWidth="3" /> : null}
        {splitX !== null ? <line x1={splitX} x2={splitX} y1={pad} y2={height - pad} stroke="rgba(0,0,0,0.35)" strokeWidth="2" strokeDasharray="6 6" /> : null}
      </svg>
      <div className="row" style={{ padding: 10, justifyContent: "space-between" }}>
        <div className="muted">Historical Demand (Dark Blue) · Fitted Values (Orange Dashed) · Forecast Values (Green) · 95% Interval (Gray)</div>
        <div className="muted">
          Y: {fmtNum(domain.min, 2)} ~ {fmtNum(domain.max, 2)}
        </div>
      </div>
    </div>
  )
}

function pickMetricKey(rows) {
  const r0 = rows?.[0] || {}
  if (Object.prototype.hasOwnProperty.call(r0, "MAPE")) return "MAPE"
  if (Object.prototype.hasOwnProperty.call(r0, "RMSE")) return "RMSE"
  if (Object.prototype.hasOwnProperty.call(r0, "MAE")) return "MAE"
  return null
}

function toFiniteNumber(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function sortEvalRows(rows) {
  const list = Array.isArray(rows) ? rows.slice() : []
  const key = pickMetricKey(list)
  if (!key) return { key: null, rows: list }
  list.sort((a, b) => {
    const av = toFiniteNumber(a?.[key])
    const bv = toFiniteNumber(b?.[key])
    if (av === null && bv === null) return 0
    if (av === null) return 1
    if (bv === null) return -1
    return av - bv
  })
  return { key, rows: list }
}

function PageFrame({ pageNo, eyebrow, title, right, footerLeft, footerRight, children }) {
  return (
    <section className="reportPage">
      <div className="reportPageInner">
        <div className="pageHeader">
          <div>
            <div className="eyebrow">{eyebrow}</div>
            <h3 className="pageTitle">{title}</h3>
          </div>
          <div className="pageHeaderRight">{right}</div>
        </div>
        <div>{children}</div>
        <div className="pageFooter">
          <div>{footerLeft}</div>
          <div>{footerRight || `Page ${pageNo} of 10`}</div>
        </div>
      </div>
    </section>
  )
}

export default function ReportPage() {
  const router = useRouter()
  const [data, setData] = useState(null)
  const [lang, setLang] = useState("en")
  const t = (zh, en) => (lang === "zh" ? zh : en)

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("forecastpro:lastResult")
      if (raw) setData(JSON.parse(raw))
    } catch {}
  }, [])

  const report = data?.report || null
  const evalRowsRaw = report?.model_comparison || data?.evaluation_results || []
  const { key: metricKey, rows: evalRows } = useMemo(() => sortEvalRows(evalRowsRaw), [evalRowsRaw])

  useEffect(() => {
    if (!report) return
    const title = report?.cover?.title || report?.dataset_name ? (lang === "zh" ? `需求预测报告 — ${String(report.dataset_name)}` : `Demand Forecast Report — ${String(report.dataset_name)}`) : (lang === "zh" ? "需求预测报告" : "Demand Forecast Report")
    try {
      document.title = title
    } catch {}
  }, [report, lang])

  const headerRight = (
    <>
      <button
        type="button"
        onClick={() => {
          try {
            window.print()
          } catch {}
        }}
        disabled={!report}
      >
        {t("下载 PDF", "Download PDF")}
      </button>
      <button
        type="button"
        onClick={() => {
          if (!report) return
          downloadText("forecast_report.json", JSON.stringify(report, null, 2), "application/json;charset=utf-8")
        }}
        disabled={!report}
      >
        {t("下载 JSON", "Download JSON")}
      </button>
      <button type="button" onClick={() => router.push("/forecast")} disabled={!data}>
        {t("返回预测", "Return to Forecast")}
      </button>
    </>
  )

  if (!data || !report) {
    return (
      <>
        <BgCanvas />
        <div className="mask" />
        <div className="shell">
          <AppShell active="report" title="Report" subtitle={t("结构化管理层预测报告（可打印为 PDF）。", "Structured Management Forecast Report (Printable PDF).")} headerRight={headerRight}>
            <div className="panel">
              <div className="card">
                <h2>{t("尚未产生报告", "No Report Generated Yet")}</h2>
                <div className="muted">{t("请先在 Agent 页面生成预测，然后打开 Report 页面。", "Please generate a forecast on the Agent page first, then open the Report page.")}</div>
              </div>
            </div>
          </AppShell>
        </div>
      </>
    )
  }

  const cover = report.cover || {}
  const tr = cover.time_range || {}
  const datasetLabel = cover.dataset || report.dataset_name || t("数据集", "Dataset")
  const reportTitle = cover.title || t(`需求预测报告 — ${datasetLabel}`, `Demand Forecast Report — ${datasetLabel}`)
  const reportDate = String(report.report_date || "—")
  const footerLeft = `${datasetLabel} · ${reportDate}`
  const pages = Array.isArray(report?.pages) ? report.pages : []

  function renderTable(rows, highlightModel) {
    if (!Array.isArray(rows) || rows.length === 0) return <div className="muted">—</div>
    const keys = Object.keys(rows[0] || {})
    return (
      <div style={{ overflow: "auto", border: "1px solid rgba(12,15,20,0.10)", borderRadius: 16, background: "rgba(255,255,255,0.72)" }}>
        <table className="magTable">
          <thead>
            <tr>
              {keys.map((k) => (
                <th key={`th_${k}`}>{k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const m = r?.["模型"] ?? r?.model
              const isRec = highlightModel && m && String(m) === String(highlightModel)
              return (
                <tr key={`tr_${i}`} className={isRec ? "recommended" : ""}>
                  {keys.map((k) => (
                    <td key={`td_${i}_${k}`}>{r?.[k] === null || r?.[k] === undefined ? "" : String(r[k])}</td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }

  function renderBlock(block, idx, highlightModel) {
    const type = block?.type
    if (type === "h1") return <h1 key={`b_${idx}`} className="magTitle">{String(block?.text || "")}</h1>
    if (type === "p") return <p key={`b_${idx}`} className="magSub" style={{ marginTop: 8 }}>{String(block?.text || "")}</p>
    if (type === "callout") {
      return (
        <div key={`b_${idx}`} className="magCallout" style={{ marginTop: 12 }}>
          <strong>{String(block?.title || "")}</strong>
          <div style={{ lineHeight: 1.55 }}>{String(block?.text || "")}</div>
        </div>
      )
    }
    if (type === "bullets") {
      const items = Array.isArray(block?.items) ? block.items : []
      return (
        <ul key={`b_${idx}`} className="list" style={{ marginTop: 10 }}>
          {items.length ? items.map((it, i) => <li key={`li_${idx}_${i}`}>{String(it)}</li>) : <li>—</li>}
        </ul>
      )
    }
    if (type === "table") {
      return (
        <div key={`b_${idx}`} style={{ marginTop: 10 }}>
          {block?.title ? <div className="magPill" style={{ marginBottom: 10 }}>{String(block.title)}</div> : null}
          {renderTable(block?.rows, highlightModel || block?.highlight_model)}
        </div>
      )
    }
    if (type === "chart") {
      return (
        <div key={`b_${idx}`} style={{ marginTop: 10 }}>
          {block?.title ? <div className="magPill" style={{ marginBottom: 10 }}>{String(block.title)}</div> : null}
          <ReportChart chartData={report?.chart_data} />
        </div>
      )
    }
    if (type === "recommendations") {
      const items = Array.isArray(block?.items) ? block.items : []
      return (
        <div key={`b_${idx}`} className="magGrid" style={{ marginTop: 10 }}>
          {items.map((r, i) => (
            <div key={`rec_${idx}_${i}`} className="magCard">
              <h4>{lang === "zh" ? `建议 ${i + 1}` : `Recommendation ${i + 1}`}</h4>
              <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.01em" }}>{String(r?.title || "—")}</div>
              <div style={{ height: 10 }} />
              <ul className="list">
                <li>{lang === "zh" ? "发现" : "Finding"}：{String(r?.finding || "—")}</li>
                <li>{lang === "zh" ? "行动" : "Action"}：{String(r?.action || "—")}</li>
                <li>{lang === "zh" ? "理由" : "Reason"}：{String(r?.reason || "—")}</li>
              </ul>
            </div>
          ))}
        </div>
      )
    }
    if (type === "bar_chart") {
      const items = Array.isArray(block?.chart_data) ? block.chart_data : []
      if (!items.length) return null
      const maxVal = Math.max(...items.map(it => Number(it.value) || 0)) || 1
      return (
        <div key={`b_${idx}`} style={{ marginTop: 10, padding: 16, background: "rgba(255,255,255,0.6)", borderRadius: 8, border: "1px solid rgba(16,50,66,0.1)" }}>
          {block?.title ? <div className="magPill" style={{ marginBottom: 12 }}>{String(block.title)}</div> : null}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {items.map((it, i) => {
              const val = Number(it.value) || 0
              const pct = Math.min(100, Math.max(0, (val / maxVal) * 100))
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", fontSize: 13 }}>
                  <div style={{ width: 100, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={String(it.label)}>{String(it.label)}</div>
                  <div style={{ flex: 1, height: 16, background: "rgba(16,50,66,0.05)", borderRadius: 4, overflow: "hidden", margin: "0 10px" }}>
                    <div style={{ width: `${pct}%`, height: "100%", background: "rgba(16,50,66,0.8)", borderRadius: 4 }} />
                  </div>
                  <div style={{ width: 60, textAlign: "right", fontFamily: "monospace" }}>{val.toFixed(2)}</div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }
    return null
  }

  if (pages.length >= 10) {
    const highlightModel = report?.best_model?.name || report?.best_model?.model || ""
    return (
      <>
        <BgCanvas />
        <div className="mask" />
        <div className="shell">
          <AppShell active="report" title="Report" subtitle={t("综合管理层预测报告（可打印为 PDF）。", "Comprehensive Management Forecast Report (Printable PDF).")} headerRight={headerRight}>
            <div className="panel">
              <div className="reportDoc">
                <div className="reportBook reportBookMag">
                  {pages.map((p) => (
                    <PageFrame
                      key={`p_${p?.page_no}`}
                      pageNo={Number(p?.page_no) || 0}
                      eyebrow="MGT550 · AI‑Powered Forecasting"
                      title={String(p?.title || "")}
                      right={t(`数据集：${datasetLabel}`, `Dataset: ${datasetLabel}`)}
                      footerLeft={footerLeft}
                    >
                      {(p?.blocks || []).map((b, i) => renderBlock(b, i, highlightModel))}
                    </PageFrame>
                  ))}
                </div>
              </div>
            </div>
          </AppShell>
        </div>
      </>
    )
  }

  const execSum = report.executive_summary || {}
  const overview = report.data_overview || {}
  const best = report.best_model || {}
  const flags = report.uncertainty_flags || []
  const chartTable = report.chart_table || []
  const modelNotes = report.model_comparison_notes || {}
  const actionable = report.actionable_recommendations || []
  const interp = report.forecast_interpretation || {}
  const bestModelName = best?.name ? String(best.name) : ""
  const splitDate = report?.chart_data?.split_date || report?.chart_data?.splitDate || report?.split_date || null

  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell
          active="report"
          title="Report"
          subtitle={t("结构化管理层预测报告（可打印为 PDF）。", "Structured Management Forecast Report (Printable PDF).")}
          headerRight={headerRight}
        >
          <div className="panel">
            <div className="reportDoc">
              <div className="reportBook reportBookMag">
                <section className="reportPage">
                  <div className="reportPageInner">
                    <div className="pageHeader">
                      <div>
                        <div className="eyebrow">Management Forecast Report</div>
                        <h3 className="pageTitle">{t("封面", "Cover Page")}</h3>
                      </div>
                      <div className="pageHeaderRight">{cover.generated_by || t("由人工智能预测系统生成", "Generated by AI Forecasting System")}</div>
                    </div>
                    <h1 className="magTitle">{reportTitle}</h1>
                    <p className="magSub">
                      {t("数据集", "Dataset")}：{datasetLabel} · {t("时间段", "Time Period")}：{fmtDate(tr.start)} {t("至", "to")} {fmtDate(tr.end)} · {t("频率", "Frequency")}：{tr.frequency || report?.data_summary?.frequency || "—"}
                    </p>
                    <div className="magBadges">
                      <span className="magBadge">
                        <span className="magBadgeDot" />
                        {t("编制日期", "Report Date")}：{reportDate}
                      </span>
                      {best?.name ? (
                        <span className="magBadge">
                          <span className="magBadgeDot" />
                          {t("推荐模型", "Recommended Model")}：{String(best.name)}
                        </span>
                      ) : null}
                      {cover?.theme ? (
                        <span className="magBadge">
                          <span className="magBadgeDot" />
                          {t("主题", "Theme")}：{String(cover.theme)}
                        </span>
                      ) : null}
                    </div>
                    {cover?.description ? (
                      <div className="magCallout" style={{ marginTop: 14 }}>
                        <strong>{t("数据主题描述", "Data Theme Description")}</strong>
                        <div className="muted">{String(cover.description)}</div>
                      </div>
                    ) : null}
                    <div className="pageFooter">
                      <div>{footerLeft}</div>
                      <div>{t("第 1 / 10 页", "Page 1 of 10")}</div>
                    </div>
                  </div>
                </section>

                <PageFrame
                  pageNo={2}
                  eyebrow="Executive Summary"
                  title={t("执行摘要（给管理层的 30 秒版本）", "Executive Summary (30-second version for management)")}
                  right={t(`训练/测试分割：${splitDate ? fmtDate(splitDate) : "—"}`, `Train/Test Split: ${splitDate ? fmtDate(splitDate) : "—"}`)}
                  footerLeft={footerLeft}
                >
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("本次分析", "This Analysis")}</h4>
                      <div style={{ fontSize: 14, lineHeight: 1.5 }}>{execSum.analyzed_data || "—"}</div>
                      <div style={{ height: 10 }} />
                      <div className="magPill">
                        {t("胜出模型", "Winning Model")}：{best?.name || "—"}（{execSum?.winning_model?.reason || best?.selection_reason || "—"}）
                      </div>
                    </div>
                    <div className="magCard">
                      <h4>{t("最重要的两条建议", "Top Two Recommendations")}</h4>
                      <ul className="list">
                        {(execSum.top_business_recommendations?.length
                          ? execSum.top_business_recommendations
                          : actionable.map((r) => r.action)
                        )
                          .slice(0, 2)
                          .map((t, i) => (
                            <li key={`toprec_${i}`}>{t}</li>
                          ))}
                      </ul>
                    </div>
                  </div>
                  <div style={{ height: 12 }} />
                  <div className="magKpis">
                    <div className="magKpi">
                      <div className="k">{t("未来均值", "Future Mean")}</div>
                      <div className="v">{fmtNum(interp.avg_forecast, 3)}</div>
                      <div className="s">{t("用于预算/产能基线", "Used for budget/capacity baseline")}</div>
                    </div>
                    <div className="magKpi">
                      <div className="k">{t("峰值 (点预测)", "Peak (Point Forecast)")}</div>
                      <div className="v">{fmtNum(interp?.peak?.forecast, 3)}</div>
                      <div className="s">{fmtDate(interp?.peak?.date)}</div>
                    </div>
                    <div className="magKpi">
                      <div className="k">{t("低谷 (点预测)", "Trough (Point Forecast)")}</div>
                      <div className="v">{fmtNum(interp?.trough?.forecast, 3)}</div>
                      <div className="s">{fmtDate(interp?.trough?.date)}</div>
                    </div>
                    <div className="magKpi">
                      <div className="k">{t("不确定性区间宽度", "Uncertainty Width")}</div>
                      <div className="v">{fmtNum(interp.avg_interval_width, 3)}</div>
                      <div className="s">{t("区间越宽越需要保守", "Wider interval means more conservative")}</div>
                    </div>
                  </div>
                </PageFrame>

                <PageFrame
                  pageNo={3}
                  eyebrow="Forecast"
                  title={t("预测图表 (历史·拟合·未来预测·95% 区间)", "Forecast Chart (History·Fitted·Future Forecast·95% Interval)")}
                  right={t(`标题：需求预测 — ${best?.name || "—"} — ${datasetLabel}`, `Title: Demand Forecast — ${best?.name || "—"} — ${datasetLabel}`)}
                  footerLeft={footerLeft}
                >
                  <ReportChart chartData={report.chart_data} />
                  <div style={{ height: 10 }} />
                  <div className="magCallout">
                    <strong>{t("一句话解读", "One-Sentence Interpretation")}</strong>
                    <div style={{ lineHeight: 1.55 }}>
                      {t(`未来平均预测值约 ${fmtNum(interp.avg_forecast, 3)}；相对最近一期变化约 ${fmtNum(interp.avg_vs_last_delta, 3)} (${fmtNum(interp.avg_vs_last_delta_pct, 2)}%)。`, `Future average forecast is approx ${fmtNum(interp.avg_forecast, 3)}; change vs last period is approx ${fmtNum(interp.avg_vs_last_delta, 3)} (${fmtNum(interp.avg_vs_last_delta_pct, 2)}%). `)}
                      {t(`峰值在 ${fmtDate(interp?.peak?.date)} (上限 ${fmtNum(interp?.peak?.upper_95, 3)})，低谷在 ${fmtDate(interp?.trough?.date)} (下限 ${fmtNum(interp?.trough?.lower_95, 3)})。`, `Peak at ${fmtDate(interp?.peak?.date)} (Upper ${fmtNum(interp?.peak?.upper_95, 3)}), Trough at ${fmtDate(interp?.trough?.date)} (Lower ${fmtNum(interp?.trough?.lower_95, 3)}).`)}
                    </div>
                  </div>
                </PageFrame>

                <PageFrame
                  pageNo={4}
                  eyebrow="Forecast Table"
                  title={t("预测表格（用于核对与执行）", "Forecast Table (For verification and execution)")}
                  right={t("明细可下载 JSON", "Full details available in JSON")}
                  footerLeft={footerLeft}
                >
                  {chartTable?.length ? (
                    <div style={{ overflow: "auto", border: "1px solid rgba(12,15,20,0.10)", borderRadius: 16, background: "rgba(255,255,255,0.72)" }}>
                      <table className="magTable">
                        <thead>
                          <tr>
                            <th>{t("时期", "Period")}</th>
                            <th>{t("历史数据", "Historical Data")}</th>
                            <th>{t("拟合值", "Fitted Value")}</th>
                            <th>{t("预测值", "Forecast Value")}</th>
                            <th>{t("95% 下限", "95% Lower")}</th>
                            <th>{t("95% 上限", "95% Upper")}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {chartTable.slice(0, 36).map((r, i) => (
                            <tr key={`ct_${i}`}>
                              <td>{fmtDate(r.period)}</td>
                              <td>{fmtNum(r.history_actual, 3)}</td>
                              <td>{fmtNum(r.fitted, 3)}</td>
                              <td>{fmtNum(r.forecast, 3)}</td>
                              <td>{fmtNum(r.lower_95, 3)}</td>
                              <td>{fmtNum(r.upper_95, 3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="muted">{t("（无可用图表数据）", "(No chart table data available)")}</div>
                  )}
                  <div className="muted" style={{ marginTop: 10 }}>
                    {t("打印版默认展示前 36 行；完整明细请点击右上角“下载 JSON”。", "Print version shows top 36 rows by default; for full details, click 'Download JSON' in top right.")}
                  </div>
                </PageFrame>

                <PageFrame
                  pageNo={5}
                  eyebrow="Data"
                  title={t("数据概览 (质量与模式)", "Data Overview (Quality and Pattern)")}
                  right={t(`目标列：${report?.detected?.demand_col || data?.detected?.demand_col || "—"}`, `Target Column: ${report?.detected?.demand_col || data?.detected?.demand_col || "—"}`)}
                  footerLeft={footerLeft}
                >
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("数据概况", "Data Summary")}</h4>
                      <ul className="list">
                        <li>{t("观测值", "Observations")}：{overview?.summary?.observations ?? report?.data_summary?.total_observations ?? "—"}</li>
                        <li>{t("时间段", "Time Period")}：{fmtDate(overview?.summary?.start)} {t("至", "to")} {fmtDate(overview?.summary?.end)}</li>
                        <li>{t("频率", "Frequency")}：{overview?.summary?.frequency ?? "—"}</li>
                        <li>{t("缺失值 (目标列)", "Missing Values (Target Column)")}：{overview?.summary?.missing?.missing_values_target ?? "—"}</li>
                      </ul>
                      <div style={{ height: 10 }} />
                      <div className="magPill">{t("当数据过短 (<24期) 时，季节性模型可能不可靠。", "When data is too short (<24), seasonal models may be unreliable.")}</div>
                    </div>
                    <div className="magCard">
                      <h4>{t("关键统计", "Key Statistics")}</h4>
                      <div className="magKpis" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
                        <div className="magKpi">
                          <div className="k">{t("均值", "Mean")}</div>
                          <div className="v">{fmtNum(overview?.key_statistics?.mean, 3)}</div>
                        </div>
                        <div className="magKpi">
                          <div className="k">{t("标准差", "Std Dev")}</div>
                          <div className="v">{fmtNum(overview?.key_statistics?.std, 3)}</div>
                        </div>
                        <div className="magKpi">
                          <div className="k">{t("最小值", "Min")}</div>
                          <div className="v">{fmtNum(overview?.key_statistics?.min, 3)}</div>
                        </div>
                        <div className="magKpi">
                          <div className="k">{t("最大值", "Max")}</div>
                          <div className="v">{fmtNum(overview?.key_statistics?.max, 3)}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </PageFrame>

                <PageFrame
                  pageNo={6}
                  eyebrow="Model Comparison"
                  title={t("模型对比表（按误差排序，最佳模型高亮）", "Model Comparison Table (Sorted by Error, Best Highlighted)")}
                  right={metricKey ? t(`排序指标：${metricKey} (越小越好)`, `Sorting Metric: ${metricKey} (smaller is better)`) : t("排序指标：—", "Sorting Metric: —")}
                  footerLeft={footerLeft}
                >
                  <div style={{ overflow: "auto", border: "1px solid rgba(12,15,20,0.10)", borderRadius: 16, background: "rgba(255,255,255,0.72)" }}>
                    <table className="magTable">
                      <thead>
                        <tr>
                          <th>#</th>
                          {Object.keys(evalRows?.[0] || {}).map((k) => (
                            <th key={`h_${k}`}>{k}</th>
                          ))}
                          <th>{t("标记", "Tag")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(evalRows || []).map((row, i) => {
                          const modelName = String(row?.model ?? "")
                          const isRec = bestModelName && modelName && modelName === bestModelName
                          return (
                            <tr key={`er_${i}`} className={isRec ? "recommended" : ""}>
                              <td>{String(i + 1)}</td>
                              {Object.keys(evalRows?.[0] || {}).map((k) => (
                                <td key={`c_${i}_${k}`}>{row?.[k] === null || row?.[k] === undefined ? "" : String(row[k])}</td>
                              ))}
                              <td>{isRec ? t("推荐", "Recommended") : ""}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ height: 10 }} />
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("未胜出模型原因摘要", "Reasons for Lower-ranked Models (Summary)")}</h4>
                      <ul className="list">
                        {Object.keys(modelNotes || {}).length
                          ? Object.entries(modelNotes)
                              .slice(0, 8)
                              .map(([k, v]) => (
                                <li key={`mn_${k}`}>
                                  {k}：{v}
                                </li>
                              ))
                          : [<li key="mn_none">—</li>]}
                      </ul>
                    </div>
                    <div className="magCard">
                      <h4>{t("数据泄露与过拟合控制", "Data Leakage & Overfitting Checks")}</h4>
                      <ul className="list">
                        <li>{t(`所有模型使用一致的时间序列切分（分割日期：${splitDate ? fmtDate(splitDate) : "—"}）。`, `Consistent train/test split for all models (Split Date: ${splitDate ? fmtDate(splitDate) : "—"}).`)}</li>
                        <li>{t("测试集数据严格不参与训练调参，仅用于保留集误差评估与图表展示。", "Test set observations are not used for training/tuning; only for evaluation and display.")}</li>
                      </ul>
                    </div>
                  </div>
                </PageFrame>

                <PageFrame pageNo={7} eyebrow="Interpretation" title={t("预测解读（业务语言）", "Forecast Interpretation (Business Language)")} right={t("趋势·峰值·低谷·不确定性", "Trend·Peak·Trough·Uncertainty")} footerLeft={footerLeft}>
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("趋势与节奏", "Trend and Rhythm")}</h4>
                      <ul className="list">
                        <li>{t(`趋势：未来平均预测值约 ${fmtNum(interp.avg_forecast, 3)}。`, `Trend: Future Average Forecast is approx ${fmtNum(interp.avg_forecast, 3)}.`)}</li>
                        <li>
                          {t(`相对最近一期变化约 ${fmtNum(interp.avg_vs_last_delta, 3)} (${fmtNum(interp.avg_vs_last_delta_pct, 2)}%)。`, `Change vs last period approx ${fmtNum(interp.avg_vs_last_delta, 3)} (${fmtNum(interp.avg_vs_last_delta_pct, 2)}%).`)}
                        </li>
                        <li>{t(`不确定性：平均预测区间宽度约 ${fmtNum(interp.avg_interval_width, 3)}。`, `Uncertainty: Avg interval width is approx ${fmtNum(interp.avg_interval_width, 3)}.`)}</li>
                      </ul>
                    </div>
                    <div className="magCard">
                      <h4>{t("峰值与低谷（用于划定边界）", "Peaks and Troughs (For Planning Boundaries)")}</h4>
                      <ul className="list">
                        <li>
                          {t(`峰值：${fmtDate(interp?.peak?.date)}，点预测 ${fmtNum(interp?.peak?.forecast, 3)}，95% 上限 ${fmtNum(interp?.peak?.upper_95, 3)}。`, `Peak: ${fmtDate(interp?.peak?.date)}, Point Forecast ${fmtNum(interp?.peak?.forecast, 3)}, 95% Upper ${fmtNum(interp?.peak?.upper_95, 3)}.`)}
                        </li>
                        <li>
                          {t(`低谷：${fmtDate(interp?.trough?.date)}，点预测 ${fmtNum(interp?.trough?.forecast, 3)}，95% 下限 ${fmtNum(interp?.trough?.lower_95, 3)}。`, `Trough: ${fmtDate(interp?.trough?.date)}, Point Forecast ${fmtNum(interp?.trough?.forecast, 3)}, 95% Lower ${fmtNum(interp?.trough?.lower_95, 3)}.`)}
                        </li>
                      </ul>
                      <div style={{ height: 10 }} />
                      <div className="magPill">{t("保守规划参考上限；资源压降参考下限", "Conservative planning refers to upper limit; resource reduction refers to lower limit")}</div>
                    </div>
                  </div>
                </PageFrame>

                <PageFrame pageNo={8} eyebrow="Actions" title={t("两条可执行建议 (量化)", "Two Actionable Suggestions (Quantified)")} right={t("直接源自预测", "Directly derived from Forecast")} footerLeft={footerLeft}>
                  <div className="magGrid">
                    {(actionable.length ? actionable : [{ title: "—", finding: "—", action: "—", reason: "—" }, { title: "—", finding: "—", action: "—", reason: "—" }])
                      .slice(0, 2)
                      .map((r, i) => (
                        <div className="magCard" key={`ar_${i}`}>
                          <h4>{t(`建议 ${i + 1}`, `Recommendation ${i + 1}`)}</h4>
                          <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.01em" }}>{r.title || "—"}</div>
                          <div style={{ height: 10 }} />
                          <ul className="list">
                            <li>{t("发现", "Finding")}：{r.finding || "—"}</li>
                            <li>{t("行动", "Action")}：{r.action || "—"}</li>
                            <li>{t("理由", "Reason")}：{r.reason || "—"}</li>
                          </ul>
                        </div>
                      ))}
                  </div>
                </PageFrame>

                <PageFrame pageNo={9} eyebrow="Uncertainty & Risk" title={t("不确定性与风险提示（面向管理层）", "Uncertainty & Risk Warnings (For Management)")} right={t("95%区间 ≠ 确定", "95% Interval ≠ Certainty")} footerLeft={footerLeft}>
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("如何解读 95% 预测区间", "How to Read 95% Forecast Interval")}</h4>
                      <ul className="list">
                        <li>{t("点预测是“最可能”的单点；95% 区间给出更现实的波动范围。", "Point Forecast is the 'most likely' single point; 95% interval gives a more realistic fluctuation range.")}</li>
                        <li>{t("资源/库存上限规划参考上限；保守成本压降参考下限。", "Prioritize upper bound for resource/inventory ceiling planning; prioritize lower bound for conservative cost cutting.")}</li>
                        <li>{t("区间越宽，不确定性越高，策略需越保守。", "Wider interval indicates higher uncertainty, requiring a more conservative strategy.")}</li>
                      </ul>
                    </div>
                    <div className="magCard">
                      <h4>{t("风险标记", "Risk Flags")}</h4>
                      <ul className="list">{flags.length ? flags.map((t, i) => <li key={`f_${i}`}>{t}</li>) : <li>{t("无", "None")}</li>}</ul>
                      <div style={{ height: 10 }} />
                      <div className="magCallout">
                        <strong>{t("提醒", "Reminder")}</strong>
                        <div style={{ lineHeight: 1.55 }}>
                          {t("若 MAPE > 30%，请将预测结果视为“高不确定性”，建议增加安全边际或缩短滚动预测周期。", "If MAPE > 30%, treat forecast results as 'high uncertainty', recommend increasing safety margin or shortening rolling forecast period.")}
                        </div>
                      </div>
                    </div>
                  </div>
                </PageFrame>

                <PageFrame pageNo={10} eyebrow="Appendix" title={t("技术附录（用于核对与合规）", "Technical Appendix (For verification and compliance)")} right={t("指标公式与分割信息", "Metric Formulas & Split Info")} footerLeft={footerLeft}>
                  <div className="magGrid">
                    <div className="magCard">
                      <h4>{t("训练/测试划分", "Train/Test Split")}</h4>
                      <ul className="list">
                        <li>{t(`分割日期：${splitDate ? fmtDate(splitDate) : "—"}`, `Split Date: ${splitDate ? fmtDate(splitDate) : "—"}`)}</li>
                        <li>{t("测试集数据不参与训练或调参。", "Test set observations are not used for training or tuning.")}</li>
                      </ul>
                      <div style={{ height: 10 }} />
                      <div className="magPill">{t("若数据过短或严重缺失，建议补充数据后重新评估。", "If data is too short or severely missing, suggest supplementing data and reviewing.")}</div>
                    </div>
                    <div className="magCard">
                      <h4>{t("指标公式", "Metric Formulas")}</h4>
                      <ul className="list">
                        <li>MAE = (1/n) × Σ|{t("实际值", "Actual")} - {t("预测值", "Forecast")}|</li>
                        <li>RMSE = √[(1/n) × Σ({t("实际值", "Actual")} - {t("预测值", "Forecast")})²]</li>
                        <li>MAPE = (1/n) × Σ|({t("实际值", "Actual")} - {t("预测值", "Forecast")}) / {t("实际值", "Actual")}| × 100</li>
                      </ul>
                      <div style={{ height: 10 }} />
                      <div className="muted">{t("导出 PDF：点击右上角“下载 PDF”，在浏览器打印对话框中选择“另存为 PDF”。", "Export PDF: Click 'Download PDF' in top right, choose 'Save as PDF' in browser print dialog.")}</div>
                    </div>
                  </div>
                </PageFrame>
              </div>
            </div>
          </div>
        </AppShell>
      </div>
    </>
  )
}
