"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"

function toNumber(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function normalizeBaseUrl(raw) {
  const s = String(raw || "").trim()
  if (!s) return ""
  if (s.startsWith("http://") || s.startsWith("https://")) return s.replace(/\/+$/, "")
  const host = s.replace(/^\/\//, "").split("/")[0] || ""
  const isLocal =
    host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0") || host.endsWith(".local")
  const proto = isLocal ? "http://" : "https://"
  const withProto = `${proto}${s.replace(/^\/\//, "")}`
  return withProto.replace(/\/+$/, "")
}

function clamp(n, min, max) {
  return Math.min(Math.max(n, min), max)
}

function formatNumber(v) {
  if (v === null || v === undefined) return "—"
  const n = Number(v)
  if (!Number.isFinite(n)) return "—"
  if (Math.abs(n) >= 1000) return n.toFixed(0)
  if (Math.abs(n) >= 100) return n.toFixed(1)
  return n.toFixed(2)
}

function formatDate(d) {
  if (!d) return "—"
  const s = String(d)
  return s.length >= 10 ? s.slice(0, 10) : s
}

function buildPolylineWithIndex(values, indices, domain, width, height, pad, totalPoints) {
  if (!values || values.length < 2) return ""
  const min = domain?.min ?? Math.min(...values)
  const max = domain?.max ?? Math.max(...values)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const denom = Math.max(totalPoints - 1, 1)
  return values
    .map((v, i) => {
      const idx = indices[i]
      const x = pad + (innerW * idx) / denom
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}

function buildAreaWithIndex(upper, lower, indices, domain, width, height, pad, totalPoints) {
  if (!upper || !lower || upper.length < 2 || lower.length < 2) return ""
  const min = domain?.min ?? Math.min(...lower)
  const max = domain?.max ?? Math.max(...upper)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const denom = Math.max(totalPoints - 1, 1)

  const top = upper.map((v, i) => {
    const idx = indices[i]
    const x = pad + (innerW * idx) / denom
    const y = pad + innerH - (innerH * (v - min)) / span
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  const bot = lower
    .slice()
    .reverse()
    .map((v, j) => {
      const i = lower.length - 1 - j
      const idx = indices[i]
      const x = pad + (innerW * idx) / denom
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

  return [...top, ...bot].join(" ")
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

export default function ForecastClient({ mode }) {
  const [lang, setLang] = useState("en")
  const t = (zh, en) => (lang === "zh" ? zh : en)

  const router = useRouter()
  const [file, setFile] = useState(null)
  const [freq, setFreq] = useState("D")
  const [testSize, setTestSize] = useState("0.2")
  const [randomSeed, setRandomSeed] = useState("42")
  const [periods, setPeriods] = useState("4")
  const [apiBase, setApiBase] = useState(process.env.NEXT_PUBLIC_API_BASE_URL || "")
  const [timeCol, setTimeCol] = useState("")
  const [demandCol, setDemandCol] = useState("")
  const [includeAdvanced, setIncludeAdvanced] = useState(false)
  const [datasetName, setDatasetName] = useState("")
  const [datasetTheme, setDatasetTheme] = useState("")

  const [llmProvider, setLlmProvider] = useState("deepseek")
  const [llmBaseUrl, setLlmBaseUrl] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [llmApiKey, setLlmApiKey] = useState("")
  const [dataDescription, setDataDescription] = useState("")

  const [methodEts, setMethodEts] = useState(true)
  const [methodNaive, setMethodNaive] = useState(true)
  const [methodSNaive, setMethodSNaive] = useState(true)
  const [methodMA, setMethodMA] = useState(true)
  const [methodArima, setMethodArima] = useState(true)
  const [methodRF, setMethodRF] = useState(false)
  const [methodXGB, setMethodXGB] = useState(false)
  const [methodLR, setMethodLR] = useState(false)
  const [methodRidge, setMethodRidge] = useState(false)
  const [methodLasso, setMethodLasso] = useState(false)

  const [apiStatus, setApiStatus] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [data, setData] = useState(null)
  const [preview, setPreview] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState("")
  const [, setHover] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [hoverXY, setHoverXY] = useState(null)
  const [activeMethod, setActiveMethod] = useState(null)
  const chartWrapRef = useRef(null)
  const [hoverIndexBest, setHoverIndexBest] = useState(null)
  const [hoverXYBest, setHoverXYBest] = useState(null)
  const chartWrapBestRef = useRef(null)

  useEffect(() => {
    try {
      const raw = localStorage.getItem("forecastpro:lastParams")
      if (raw) {
        const p = JSON.parse(raw)
        if (p?.freq) setFreq(String(p.freq))
        if (p?.testSize) setTestSize(String(p.testSize))
        if (p?.randomSeed) setRandomSeed(String(p.randomSeed))
        if (p?.periods) setPeriods(String(p.periods))
        if (p?.apiBase) setApiBase(String(p.apiBase))
        if (p?.timeCol) setTimeCol(String(p.timeCol))
        if (p?.demandCol) setDemandCol(String(p.demandCol))
        if (p?.includeAdvanced !== undefined) setIncludeAdvanced(Boolean(p.includeAdvanced))
        if (p?.methodLR !== undefined) setMethodLR(Boolean(p.methodLR))
        if (p?.methodRidge !== undefined) setMethodRidge(Boolean(p.methodRidge))
        if (p?.methodLasso !== undefined) setMethodLasso(Boolean(p.methodLasso))
        if (p?.methodRF !== undefined) setMethodRF(Boolean(p.methodRF))
        if (p?.methodXGB !== undefined) setMethodXGB(Boolean(p.methodXGB))
        if (p?.datasetName !== undefined) setDatasetName(String(p.datasetName))
        if (p?.datasetTheme !== undefined) setDatasetTheme(String(p.datasetTheme))
        if (p?.dataDescription !== undefined) setDataDescription(String(p.dataDescription))
        if (p?.lang !== undefined) setLang(String(p.lang))
      }
    } catch {}
  }, [])

  useEffect(() => {
    const anyAdv = Boolean(methodLR || methodRidge || methodLasso || methodRF || methodXGB)
    if (anyAdv && !includeAdvanced) setIncludeAdvanced(true)
  }, [methodLR, methodRidge, methodLasso, methodRF, methodXGB, includeAdvanced])

  useEffect(() => {
    if (includeAdvanced) return
    if (methodLR) setMethodLR(false)
    if (methodRidge) setMethodRidge(false)
    if (methodLasso) setMethodLasso(false)
    if (methodRF) setMethodRF(false)
    if (methodXGB) setMethodXGB(false)
  }, [includeAdvanced, methodLR, methodRidge, methodLasso, methodRF, methodXGB])

  useEffect(() => {
    if (mode !== "forecast") return
    try {
      const raw = sessionStorage.getItem("forecastpro:lastResult")
      if (raw) setData(JSON.parse(raw))
    } catch {}
  }, [mode])

  useEffect(() => {
    if (mode !== "upload" && mode !== "preview") return
    setPreview(null)
    setPreviewError("")
    setPreviewLoading(false)
  }, [file, mode])

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
    const isLocalHost =
      typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")

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
        if (!cancelled) setApiStatus("Disconnected")
        return
      }
      if (normalized !== apiBase && !cancelled) setApiBase(normalized)

      const candidates = [normalized].concat(isLocalHost ? ["http://localhost:8011", "http://localhost:8001", "http://localhost:8000"] : [])
      const seen = new Set()
      for (const c of candidates) {
        const key = normalizeBaseUrl(c)
        if (seen.has(key)) continue
        seen.add(key)
        if (await ping(key)) {
          if (!cancelled) {
            setApiBase(key)
            setApiStatus("Connected")
          }
          return
        }
      }
      if (!cancelled) setApiStatus("Disconnected")
    }

    autoDetect()
    return () => {
      cancelled = true
    }
  }, [apiBase])

  function setHoverFromEvent(e, s) {
    const el = chartWrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    setHover({
      label: s?.label || s?.method || "",
      color: s?.color || "#2A8CA8",
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })
  }

  function onChartMove(e) {
    if (!chart?.domain || !chart?.totalPoints) return
    const el = chartWrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const innerW = chart.width - chart.pad * 2
    const denom = Math.max(chart.totalPoints - 1, 1)
    const raw = (x - chart.pad) / innerW
    const idx = Math.round(raw * denom)
    const clamped = clamp(idx, 0, chart.totalPoints - 1)
    setHoverIndex(clamped)
    setHoverXY({ x, y })
  }

  function onChartMoveBest(e) {
    if (!chartBest?.domain || !chartBest?.totalPoints) return
    const el = chartWrapBestRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const innerW = chartBest.width - chartBest.pad * 2
    const denom = Math.max(chartBest.totalPoints - 1, 1)
    const raw = (x - chartBest.pad) / innerW
    const idx = Math.round(raw * denom)
    const clamped = clamp(idx, 0, chartBest.totalPoints - 1)
    setHoverIndexBest(clamped)
    setHoverXYBest({ x, y })
  }

  const chart = useMemo(() => {
    const width = 760
    const height = 260
    const pad = 46
    const actual = data?.history?.actual || []
    const fittedBest = data?.history?.fitted_best || []
    const fittedByMethod = data?.history?.fitted_by_method || {}
    const testPredByMethod = data?.history?.test_pred_by_method || {}
    const trainEndDate = data?.history?.train_end_date || null
    const selected = new Set(
      []
        .concat(methodEts ? ["ets"] : [])
        .concat(methodNaive ? ["naive"] : [])
        .concat(methodSNaive ? ["seasonal_naive"] : [])
        .concat(methodMA ? ["moving_average"] : [])
        .concat(methodArima ? ["arima"] : [])
        .concat(includeAdvanced && methodRF ? ["random_forest"] : [])
        .concat(includeAdvanced && methodXGB ? ["xgboost"] : [])
        .concat(includeAdvanced && methodLR ? ["linear_regression"] : [])
        .concat(includeAdvanced && methodRidge ? ["ridge_regression"] : [])
        .concat(includeAdvanced && methodLasso ? ["lasso_regression"] : [])
    )
    const methods = Array.from(selected).filter((m) => selected.has(m) && (fittedByMethod?.[m] || testPredByMethod?.[m]))
    const cfg = {
      ets: { label: "ETS", color: "#3BA0D3", dash: "10 6" },
      naive: { label: "Naïve", color: "#6C7A89", dash: "2 10" },
      seasonal_naive: { label: "Seasonal Naïve", color: "#1FA971", dash: "6 6" },
      moving_average: { label: "Moving Average", color: "#D9A441", dash: "12 6" },
      arima: { label: "ARIMA/SARIMA", color: "#1A4B5F", dash: "8 4 2 4" },
      random_forest: { label: "Random Forest", color: "#0B1F2A", dash: "10 4" },
      xgboost: { label: "XGBoost", color: "#216A83", dash: "2 2" },
      linear_regression: { label: "OLS", color: "#33A9C2", dash: "12 4 2 4" },
      ridge_regression: { label: "Ridge", color: "#5FC3D3", dash: "6 2" },
      lasso_regression: { label: "Lasso", color: "#8AD9E2", dash: "3 6" },
    }

    if (!actual.length && methods.length === 0) {
      return {
        width,
        height,
        pad,
        totalPoints: 0,
        actualLen: 0,
        domain: null,
        actualPoints: "",
        fittedPoints: "",
        splitIndex: null,
        series: [],
      }
    }

    const allForDomain = []
      .concat(actual || [])
      .concat((fittedBest || []).filter((v) => v !== null && v !== undefined))
      .concat(
        ...methods.map((m) => (Array.isArray(fittedByMethod?.[m]) ? fittedByMethod[m].filter((v) => v !== null && v !== undefined) : []))
      )
      .concat(
        ...methods.map((m) => (Array.isArray(testPredByMethod?.[m]) ? testPredByMethod[m].filter((v) => v !== null && v !== undefined) : []))
      )
      .filter((v) => Number.isFinite(Number(v)))
      .map((v) => Number(v))

    const min = Math.min(...allForDomain)
    const max = Math.max(...allForDomain)
    const domain = { min, max }

    const actualIndices = actual.map((_, i) => i)
    const totalPoints = actual.length

    const actualPoints = actual.length >= 2 ? buildPolylineWithIndex(actual, actualIndices, domain, width, height, pad, totalPoints) : ""

    const fittedPairs = []
    for (let i = 0; i < Math.min(actual.length, fittedBest.length); i++) {
      const v = fittedBest[i]
      if (v === null || v === undefined) continue
      if (!Number.isFinite(Number(v))) continue
      fittedPairs.push([i, Number(v)])
    }
    const fittedPoints =
      fittedPairs.length >= 2
        ? buildPolylineWithIndex(
            fittedPairs.map((x) => x[1]),
            fittedPairs.map((x) => x[0]),
            domain,
            width,
            height,
            pad,
            totalPoints
          )
        : ""

    const series = []
    for (const m of methods) {
      const color = cfg[m]?.color || "#2A8CA8"
      const fittedM = Array.isArray(fittedByMethod?.[m]) ? fittedByMethod[m] : null
      const fittedPairsM = []
      if (fittedM) {
        for (let i = 0; i < Math.min(actual.length, fittedM.length); i++) {
          const v = fittedM[i]
          if (v === null || v === undefined) continue
          if (!Number.isFinite(Number(v))) continue
          fittedPairsM.push([i, Number(v)])
        }
      }
      const fittedPointsM =
        fittedPairsM.length >= 2
          ? buildPolylineWithIndex(
              fittedPairsM.map((x) => x[1]),
              fittedPairsM.map((x) => x[0]),
              domain,
              width,
              height,
              pad,
              totalPoints
            )
          : ""

      const predM = Array.isArray(testPredByMethod?.[m]) ? testPredByMethod[m] : null
      const predPairsM = []
      if (predM) {
        for (let i = 0; i < Math.min(actual.length, predM.length); i++) {
          const v = predM[i]
          if (v === null || v === undefined) continue
          if (!Number.isFinite(Number(v))) continue
          predPairsM.push([i, Number(v)])
        }
      }
      const testPointsM =
        predPairsM.length >= 2
          ? buildPolylineWithIndex(
              predPairsM.map((x) => x[1]),
              predPairsM.map((x) => x[0]),
              domain,
              width,
              height,
              pad,
              totalPoints
            )
          : ""

      series.push({
        method: m,
        color,
        dash: cfg[m]?.dash || "8 6",
        label: cfg[m]?.label || m,
        fittedPoints: fittedPointsM,
        testPoints: testPointsM,
      })
    }

    let splitIndex = null
    try {
      if (trainEndDate && Array.isArray(data?.history?.dates) && data.history.dates.length) {
        const tEnd = new Date(trainEndDate)
        for (let i = 0; i < data.history.dates.length; i++) {
          const dt = new Date(data.history.dates[i])
          if (!Number.isNaN(dt.getTime()) && !Number.isNaN(tEnd.getTime()) && dt.getTime() <= tEnd.getTime()) splitIndex = i
        }
      }
    } catch {}

    return { width, height, pad, totalPoints, actualLen: actual.length, domain, actualPoints, fittedPoints, splitIndex, series }
  }, [
    data,
    methodEts,
    methodNaive,
    methodSNaive,
    methodMA,
    methodArima,
    methodRF,
    methodXGB,
    methodLR,
    methodRidge,
    methodLasso,
    includeAdvanced,
  ])

  const chartBest = useMemo(() => {
    const width = 760
    const height = 260
    const pad = 46
    const actual = data?.history?.actual || []
    const fittedBest = data?.history?.fitted_best || []
    const testPredBest = data?.history?.test_pred_best || []
    const fc = data?.forecast_results?.forecast || []
    const up = data?.forecast_results?.upper_bound || []
    const lo = data?.forecast_results?.lower_bound || []
    const trainEndDate = data?.history?.train_end_date || null

    if (!actual.length && !fc.length) {
      return {
        width,
        height,
        pad,
        totalPoints: 0,
        actualLen: 0,
        domain: null,
        actualPoints: "",
        fittedPoints: "",
        forecastPoints: "",
        areaPoints: "",
        splitIndex: null,
      }
    }

    const actualLen = actual.length
    const totalPoints = actualLen + Math.max(0, fc.length)
    const actualIndices = actual.map((_, i) => i)
    const actualPoints = actualLen >= 2 ? buildPolylineWithIndex(actual, actualIndices, null, width, height, pad, totalPoints) : ""

    const fittedPairs = []
    for (let i = 0; i < Math.min(actualLen, fittedBest.length); i++) {
      const v = fittedBest[i]
      if (v === null || v === undefined) continue
      if (!Number.isFinite(Number(v))) continue
      fittedPairs.push([i, Number(v)])
    }
    const fittedPoints =
      fittedPairs.length >= 2
        ? buildPolylineWithIndex(
            fittedPairs.map((x) => x[1]),
            fittedPairs.map((x) => x[0]),
            null,
            width,
            height,
            pad,
            totalPoints
          )
        : ""

    const lastActual = actualLen ? actual[actualLen - 1] : null
    const fcWithAnchor = lastActual !== null ? [lastActual, ...fc] : fc
    const idx = fcWithAnchor.map((_, i) => Math.max(actualLen - 1, 0) + i)

    const domainVals = []
      .concat(actual || [])
      .concat(fittedPairs.map((x) => x[1]))
      .concat((testPredBest || []).filter((v) => v !== null && v !== undefined))
      .concat(fc || [])
      .concat(up || [])
      .concat(lo || [])
      .filter((v) => Number.isFinite(Number(v)))
      .map((v) => Number(v))

    const domain = domainVals.length ? { min: Math.min(...domainVals), max: Math.max(...domainVals) } : null

    const forecastPoints =
      domain && fcWithAnchor.length >= 2 ? buildPolylineWithIndex(fcWithAnchor.map((v) => Number(v)), idx, domain, width, height, pad, totalPoints) : ""

    const upA = lastActual !== null ? [lastActual, ...up] : up
    const loA = lastActual !== null ? [lastActual, ...lo] : lo
    const areaPoints =
      domain && upA.length >= 2 && loA.length >= 2 ? buildAreaWithIndex(upA.map((v) => Number(v)), loA.map((v) => Number(v)), idx, domain, width, height, pad, totalPoints) : ""

    const actualPoints2 = domain && actualLen >= 2 ? buildPolylineWithIndex(actual.map((v) => Number(v)), actualIndices, domain, width, height, pad, totalPoints) : ""

    const predPairs = []
    for (let i = 0; i < Math.min(actualLen, testPredBest.length); i++) {
      const v = testPredBest[i]
      if (v === null || v === undefined) continue
      if (!Number.isFinite(Number(v))) continue
      predPairs.push([i, Number(v)])
    }
    const testPoints =
      predPairs.length >= 2
        ? buildPolylineWithIndex(
            predPairs.map((x) => x[1]),
            predPairs.map((x) => x[0]),
            domain,
            width,
            height,
            pad,
            totalPoints
          )
        : ""

    let splitIndex = null
    try {
      if (trainEndDate && Array.isArray(data?.history?.dates) && data.history.dates.length) {
        const tEnd = new Date(trainEndDate)
        for (let i = 0; i < data.history.dates.length; i++) {
          const dt = new Date(data.history.dates[i])
          if (!Number.isNaN(dt.getTime()) && !Number.isNaN(tEnd.getTime()) && dt.getTime() <= tEnd.getTime()) splitIndex = i
        }
      }
    } catch {}

    return {
      width,
      height,
      pad,
      totalPoints,
      actualLen,
      domain,
      actualPoints: actualPoints2,
      fittedPoints,
      testPoints,
      forecastPoints,
      areaPoints,
      splitIndex,
    }
  }, [data])

  const requestedMethods = useMemo(() => {
    const methods = []
    if (methodEts) methods.push("ets")
    if (methodNaive) methods.push("naive")
    if (methodSNaive) methods.push("seasonal_naive")
    if (methodMA) methods.push("moving_average")
    if (methodArima) methods.push("arima")
    if (includeAdvanced && methodRF) methods.push("random_forest")
    if (includeAdvanced && methodXGB) methods.push("xgboost")
    if (includeAdvanced && methodLR) methods.push("linear_regression")
    if (includeAdvanced && methodRidge) methods.push("ridge_regression")
    if (includeAdvanced && methodLasso) methods.push("lasso_regression")
    return methods
  }, [methodEts, methodNaive, methodSNaive, methodMA, methodArima, methodRF, methodXGB, methodLR, methodRidge, methodLasso, includeAdvanced])

  const futureErrors = data?.future_errors || {}
  const forecast = data?.forecast_results
  const history = data?.history
  const demandLabel = data?.detected?.demand_col || demandCol || "y"
  const evalRows = data?.evaluation_results || []
  const methodToModel = data?.method_to_model || {}
  const bestName = data?.report?.best_model?.name || null

  const metricRows = useMemo(() => {
    const out = []
    const required = [
      { method: "moving_average", label: "MA" },
      { method: "ets", label: "ETS" },
      { method: "arima", label: "ARIMA" },
    ]
    const byModel = new Map((evalRows || []).map((r) => [String(r?.model), r]))
    const seen = new Set()

    for (const r of required) {
      const modelKey = r.method === "ets" ? String(methodToModel?.ets || "ets") : r.method
      const row = byModel.get(modelKey)
      if (!row) continue
      out.push({ key: `${r.method}:${modelKey}`, label: r.label, model: modelKey, row })
      seen.add(String(modelKey))
    }

    if (bestName && !seen.has(String(bestName))) {
      const row = byModel.get(String(bestName))
      if (row) out.push({ key: `best:${bestName}`, label: "Best", model: String(bestName), row })
    }

    return out
  }, [evalRows, methodToModel, bestName])

  const hoverInfo = useMemo(() => {
    if (!data || hoverIndex === null || !chart?.domain || !chart?.totalPoints) return null
    const historyDates = data?.history?.dates || []
    const historyVals = data?.history?.actual || []
    const fittedByMethod = data?.history?.fitted_by_method || {}
    const testPredByMethod = data?.history?.test_pred_by_method || {}

    const actualLen = chart.actualLen || 0
    const date = historyDates?.[hoverIndex]

    const values = []
    if (hoverIndex < actualLen && historyVals[hoverIndex] !== undefined) {
      values.push({ key: "actual", label: "Historical Actual", color: "#15222A", value: historyVals[hoverIndex] })
    }

    for (const s of chart.series || []) {
      const predArr = testPredByMethod?.[s.method]
      const vpred = Array.isArray(predArr) ? predArr[hoverIndex] : undefined
      if (vpred !== undefined && vpred !== null && Number.isFinite(Number(vpred))) {
        values.push({ key: s.method, label: `${s.label || s.method} Test Set Forecast`, color: s.color, value: vpred })
        continue
      }
      const fitArr = fittedByMethod?.[s.method]
      const vfit = Array.isArray(fitArr) ? fitArr[hoverIndex] : undefined
      if (vfit !== undefined && vfit !== null && Number.isFinite(Number(vfit))) {
        values.push({ key: `fit_${s.method}`, label: `${s.label || s.method} Fitted`, color: s.color, value: vfit })
      }
    }

    const ordered = []
    if (activeMethod) {
      const found = values.find((v) => v.key === activeMethod) || values.find((v) => v.key === `fit_${activeMethod}`)
      if (found) ordered.push(found)
    }
    for (const v of values) {
      if (ordered.find((x) => x.key === v.key)) continue
      ordered.push(v)
    }

    return {
      date: formatDate(date),
      values: ordered,
      xSvg: xForIndex(hoverIndex, chart.width, chart.pad, chart.totalPoints),
    }
  }, [data, hoverIndex, chart, activeMethod])

  const hoverInfoBest = useMemo(() => {
    if (!data || hoverIndexBest === null || !chartBest?.domain || !chartBest?.totalPoints) return null
    const historyDates = data?.history?.dates || []
    const historyVals = data?.history?.actual || []
    const fittedBest = data?.history?.fitted_best || []
    const testPredBest = data?.history?.test_pred_best || []
    const forecast = data?.forecast_results || {}
    const forecastDates = forecast?.dates || []
    const fc = forecast?.forecast || []

    const actualLen = chartBest.actualLen || 0
    const anchorIdx = Math.max(actualLen - 1, 0)
    const step = hoverIndexBest - anchorIdx

    let date = null
    if (hoverIndexBest < actualLen) date = historyDates[hoverIndexBest]
    else if (step > 0) date = forecastDates[step - 1]
    else date = historyDates[anchorIdx]

    const values = []
    if (hoverIndexBest < actualLen && historyVals[hoverIndexBest] !== undefined) {
      values.push({ key: "actual", label: "Historical Actual", color: "#15222A", value: historyVals[hoverIndexBest] })
      const tp = testPredBest?.[hoverIndexBest]
      if (tp !== undefined && tp !== null && Number.isFinite(Number(tp))) {
        values.push({ key: "best_test", label: "Best Model Test Set Forecast", color: "#6C43FF", value: tp })
      }
      const fit = fittedBest?.[hoverIndexBest]
      if (fit !== undefined && fit !== null && Number.isFinite(Number(fit))) {
        values.push({ key: "fit_best", label: "Best Model Fitted", color: "rgba(136,152,170,0.95)", value: fit })
      }
    } else if (step > 0) {
      const v = fc?.[step - 1]
      if (v !== undefined) values.push({ key: "best_fc", label: "Best Model Forecast", color: "#6C43FF", value: v })
    }

    return {
      date: formatDate(date),
      values,
      xSvg: xForIndex(hoverIndexBest, chartBest.width, chartBest.pad, chartBest.totalPoints),
    }
  }, [data, hoverIndexBest, chartBest])

  const parseOrThrow = async (res) => {
    if (!res.ok) {
      const txt = await res.text()
      let errMsg = ""
      try {
        const obj = JSON.parse(txt)
        const detail = obj?.detail
        if (detail?.error) {
          errMsg = detail?.message ? `${detail.error}: ${detail.message}` : detail.error
        } else if (obj?.error) {
          errMsg = obj.error
        }
      } catch {}
      throw new Error(errMsg || txt || `Request Failed: ${res.status}`)
    }
    return await res.json()
  }

  async function onPreview() {
    setPreview(null)
    setPreviewError("")
    if (!file) {
      setPreviewError("Please select a CSV or Excel file first")
      return
    }

    const apiBaseNormalized = normalizeBaseUrl(apiBase)
    const f = new FormData()
    f.append("file", file)
    if (apiBaseNormalized) f.append("api_base", apiBaseNormalized)
    if (String(timeCol || "").trim()) f.append("time_col", String(timeCol).trim())
    if (String(demandCol || "").trim()) f.append("demand_col", String(demandCol).trim())
    if (String(datasetName || "").trim()) f.append("dataset_name", String(datasetName).trim())
    if (String(datasetTheme || "").trim()) f.append("dataset_theme", String(datasetTheme).trim())
    if (String(dataDescription || "").trim()) f.append("data_description", String(dataDescription).trim())
    f.append("language", "en")

    setPreviewLoading(true)
    try {
      const res = await fetch(`/api/preview`, { method: "POST", body: f })
      const json = await parseOrThrow(res)
      setPreview(json)
      const dt = json?.detected?.time_col
      const dd = json?.detected?.demand_col
      if (!String(timeCol || "").trim() && dt) setTimeCol(String(dt))
      if (!String(demandCol || "").trim() && dd) setDemandCol(String(dd))
    } catch (err) {
      const msg = String(err?.message || "")
      const isNetwork = err?.name === "TypeError" || msg.toLowerCase().includes("load failed") || msg.toLowerCase().includes("failed to fetch")
      const proxyUnreachable = msg.includes("unreachable api_base")
      if ((isNetwork || proxyUnreachable) && apiBaseNormalized) {
        try {
          const directF = new FormData()
          directF.append("file", file)
          if (String(timeCol || "").trim()) directF.append("time_col", String(timeCol).trim())
          if (String(demandCol || "").trim()) directF.append("demand_col", String(demandCol).trim())
          if (String(datasetName || "").trim()) directF.append("dataset_name", String(datasetName).trim())
          if (String(datasetTheme || "").trim()) directF.append("dataset_theme", String(datasetTheme).trim())
          if (String(dataDescription || "").trim()) directF.append("data_description", String(dataDescription).trim())
          directF.append("language", "en")
          const direct = await fetch(`${apiBaseNormalized}/api/preview`, { method: "POST", body: directF })
          const json = await parseOrThrow(direct)
          setPreview(json)
          const dt = json?.detected?.time_col
          const dd = json?.detected?.demand_col
          if (!String(timeCol || "").trim() && dt) setTimeCol(String(dt))
          if (!String(demandCol || "").trim() && dd) setDemandCol(String(dd))
          return
        } catch (e2) {
          setPreviewError(String(e2?.message || e2) || "Preview failed (backend unresponsive or file error)")
          return
        } finally {
          setPreviewLoading(false)
        }
      }
      setPreviewError(msg || "Preview failed (backend unresponsive or file error)")
    } finally {
      setPreviewLoading(false)
    }
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError("")
    setData(null)

    if (!file) {
      setError("Please select a CSV or Excel file first")
      return
    }

    const apiBaseNormalized = normalizeBaseUrl(apiBase)
    const buildForm = (includeApiBase) => {
      const f = new FormData()
      f.append("file", file)
      f.append("freq", freq)
      f.append("test_size", String(toNumber(testSize, 0.2)))
      f.append("random_seed", String(toNumber(randomSeed, 42)))
      f.append("periods", String(toNumber(periods, 10)))
      if (includeApiBase) f.append("api_base", apiBaseNormalized)
      f.append("include_advanced", includeAdvanced ? "true" : "false")
      if (llmApiKey) {
        f.append("use_llm", "true");
        f.append("llm_api_key", llmApiKey);
        if (llmProvider === "doubao") {
          f.append("llm_base_url", llmBaseUrl || "https://ark.cn-beijing.volces.com/api/v3");
          if (llmModel) f.append("llm_model", llmModel);
        } else if (llmProvider === "custom") {
          if (llmBaseUrl) f.append("llm_base_url", llmBaseUrl);
          if (llmModel) f.append("llm_model", llmModel);
        }
      }
      if (String(timeCol || "").trim()) f.append("time_col", String(timeCol).trim())
      if (String(demandCol || "").trim()) f.append("demand_col", String(demandCol).trim())
      if (String(datasetName || "").trim()) f.append("dataset_name", String(datasetName).trim())
      if (String(datasetTheme || "").trim()) f.append("dataset_theme", String(datasetTheme).trim())
      if (String(dataDescription || "").trim()) f.append("data_description", String(dataDescription).trim())
      f.append("language", "en")
      const methods = []
      if (methodEts) methods.push("ets")
      if (methodNaive) methods.push("naive")
      if (methodSNaive) methods.push("seasonal_naive")
      if (methodMA) methods.push("moving_average")
      if (methodArima) methods.push("arima")
      if (includeAdvanced && methodRF) methods.push("random_forest")
      if (includeAdvanced && methodXGB) methods.push("xgboost")
      if (includeAdvanced && methodLR) methods.push("linear_regression")
      if (includeAdvanced && methodRidge) methods.push("ridge_regression")
      if (includeAdvanced && methodLasso) methods.push("lasso_regression")
      if (methods.length) f.append("methods", methods.join(","))
      return f
    }

    setLoading(true)
    try {
      const res = await fetch(`/api/forecast`, { method: "POST", body: buildForm(true) })
      const json = await parseOrThrow(res)
      setData(json)
      try {
        sessionStorage.setItem("forecastpro:lastResult", JSON.stringify(json))
      } catch {}
      try {
        localStorage.setItem(
          "forecastpro:lastParams",
          JSON.stringify({
            freq,
            testSize,
            randomSeed,
            periods,
            apiBase: apiBaseNormalized || apiBase,
            timeCol,
            demandCol,
            includeAdvanced,
            methodLR,
            methodRidge,
            methodLasso,
            methodRF,
            methodXGB,
            datasetName,
            datasetTheme,
            dataDescription,
            lang,
          })
        )
      } catch {}
      if (mode === "upload") router.push("/forecast")
    } catch (err) {
      const msg = String(err?.message || "")
      const isNetwork = err?.name === "TypeError" || msg.toLowerCase().includes("load failed") || msg.toLowerCase().includes("failed to fetch")
      const proxyUnreachable = msg.includes("unreachable api_base")
      if ((isNetwork || proxyUnreachable) && apiBaseNormalized && apiBaseNormalized.startsWith("https://")) {
        try {
          const direct = await fetch(`${apiBaseNormalized}/api/forecast`, { method: "POST", body: buildForm(false) })
          const json = await parseOrThrow(direct)
          setData(json)
          try {
            sessionStorage.setItem("forecastpro:lastResult", JSON.stringify(json))
          } catch {}
          try {
            localStorage.setItem(
              "forecastpro:lastParams",
              JSON.stringify({
                freq,
                testSize,
                randomSeed,
                periods,
                apiBase: apiBaseNormalized || apiBase,
                timeCol,
                demandCol,
                includeAdvanced,
                methodLR,
                methodRidge,
                methodLasso,
                methodRF,
                methodXGB,
                datasetName,
                datasetTheme,
                dataDescription,
                lang,
              })
            )
          } catch {}
          if (mode === "upload") router.push("/forecast")
          return
        } catch (e2) {
          const msg2 = e2?.message ? String(e2.message) : ""
          setError(msg2 || msg || "Request Failed（Possibly backend timeout or file too large）")
          return
        } finally {
          setLoading(false)
        }
      }
      setError(msg || "Request Failed（Possibly backend timeout or file too large）")
    } finally {
      setLoading(false)
    }
  }

  if (mode === "upload") {
    return (
      <div className="panel">
        <div className="card">
          <h2>{t("上传与参数设置", "Upload & Parameters")}</h2>
          <form onSubmit={onSubmit}>
            <div className="formGrid">
              <div className="field">
                <label>{t("数据文件 (CSV / Excel)", "Data File (CSV / Excel)")}</label>
                <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                {file ? (
                  <div className="muted" style={{ marginTop: 6 }}>
                    {file.name} · {Math.max(1, Math.round(file.size / 1024))} KB
                  </div>
                ) : null}
              </div>
              <div className="field">
                <label>{t("界面语言", "UI Language")}</label>
                <select value={lang} onChange={(e) => setLang(e.target.value)}>
                  <option value="en">English</option>
                  <option value="zh">中文</option>
                </select>
              </div>
              <div className="field">
                <label>{t("API 地址", "API Base URL")}</label>
                <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://<backend>.up.railway.app" />
              </div>
              <div className="field">
                <label>DeepSeek API Key ({t("选填", "Optional")})</label>
                <input
                  type="password"
                  placeholder="sk-..."
                  onChange={(e) => {
                    const val = e.target.value;
                    sessionStorage.setItem("forecastpro:deepseekKey", val);
                  }}
                  defaultValue={typeof window !== "undefined" ? sessionStorage.getItem("forecastpro:deepseekKey") || "" : ""}
                />
                <div className="muted" style={{ marginTop: 4 }}>{t("配置后即可生成多模态 LLM 管理报告", "Configuring enables LLM Report Analysis")}</div>
              </div>
              <div className="field" style={{ gridColumn: "span 12" }}>
                <label>{t("数据预览 (前 30 行)", "Data Preview (Top 30 rows)")}</label>
                <div className="row" style={{ marginTop: 6, alignItems: "center", justifyContent: "space-between" }}>
                  <div className="row" style={{ alignItems: "center" }}>
                    <button type="button" onClick={onPreview} disabled={!file || previewLoading}>
                      {previewLoading ? t("预览中...", "Previewing...") : t("预览数据", "Preview Data")}
                    </button>
                    <div className="muted" style={{ marginLeft: 10 }}>
                      {normalizeBaseUrl(apiBase) || "—"} {apiStatus ? `· ${apiStatus}` : ""}
                    </div>
                  </div>
                  {preview?.detected ? (
                    <div className="muted">
                      {t("检测到列：时间", "Detected: Time")} {preview?.detected?.time_col ? String(preview.detected.time_col) : "—"} · {t("目标", "Target")}{" "}
                      {preview?.detected?.demand_col ? String(preview.detected.demand_col) : "—"}
                    </div>
                  ) : null}
                </div>

                {previewError ? <div style={{ color: "#D84B4B", marginTop: 8 }}>{previewError}</div> : null}

                {preview?.rows?.length && preview?.columns?.length ? (
                  <div
                    style={{
                      marginTop: 10,
                      overflow: "auto",
                      maxHeight: 280,
                      border: "1px solid rgba(16,50,66,0.14)",
                      borderRadius: 12,
                      background: "rgba(255,255,255,0.65)",
                    }}
                  >
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
                ) : (
                  <div className="muted" style={{ marginTop: 8 }}>
                    {t("选择文件并点击“预览数据”，即可查看前 30 行并自动检测时间/目标列。", "Select file and click 'Preview Data' to view top 30 rows and auto-detect Time/Target columns.")}
                  </div>
                )}
              </div>
              <div className="field" style={{ gridColumn: "span 12" }}>
                <label>{t("数据主题与业务场景（用于报告编写）", "Data Theme & Scenario (used for Report writing)")}</label>
                <div className="row" style={{ marginTop: 6 }}>
                  <div className="field" style={{ minWidth: 220 }}>
                    <label>{t("数据集名称 (选填)", "Dataset Name (Optional)")}</label>
                    <input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="e.g. 2026Q2_Sales_East" />
                  </div>
                  <div className="field" style={{ minWidth: 220 }}>
                    <label>{t("主题/场景", "Theme/Scenario")}</label>
                    <select value={datasetTheme} onChange={(e) => setDatasetTheme(e.target.value)}>
                      <option value="">{t("未选择", "Unselected")}</option>
                      <option value="retail_inventory">{t("零售库存/补货", "Retail Inventory / Replenishment")}</option>
                      <option value="sales_operations">{t("销售运营", "Sales Operations")}</option>
                      <option value="workforce">{t("排班/人力配置", "Workforce Scheduling")}</option>
                      <option value="facility_climate">{t("设施环控/能耗", "Facility Climate Control")}</option>
                      <option value="marketing">{t("营销活动", "Marketing Campaigns")}</option>
                      <option value="other">{t("其他", "Other")}</option>
                    </select>
                  </div>
                </div>
                <div style={{ marginTop: 8 }}>
                  <label>{t("补充说明/分析要求 (选填)", "Data Description (Optional)")}</label>
                  <textarea
                    value={dataDescription}
                    onChange={(e) => setDataDescription(e.target.value)}
                    placeholder={t("e.g.：这是华东区某零食每日销量，目标是降低缺货并控制库存周转率。请提供备货建议。", "e.g.: Daily sales data for snacks in East China. Target is to reduce stockouts and control inventory turnover.")}
                    rows={3}
                    style={{ width: "100%", resize: "vertical" }}
                  />
                </div>
                <div className="muted" style={{ marginTop: 6 }}>
                  {t("这些信息将包含在报告封面与摘要中，帮助 LLM 提供更贴合业务的建议。", "This info will be included in the report cover and executive summary to provide more contextual advice.")}
                </div>
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
                <label>{t("模型范围", "Model Scope")}</label>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <input type="checkbox" checked={includeAdvanced} onChange={(e) => setIncludeAdvanced(e.target.checked)} />
                  <span>{t("启用高级模型", "Enable Advanced Models")}</span>
                </label>
              </div>
              <div className="field" style={{ gridColumn: "span 12" }}>
                <label>{t("可选高级模型（勾选后将参与图表绘制）", "Optional Advanced Models (Will appear in Forecast charts when checked)")}</label>
                <div className="row" style={{ marginTop: 6 }}>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLR} onChange={(e) => setMethodLR(e.target.checked)} disabled={!includeAdvanced} />
                    <span>OLS</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRidge} onChange={(e) => setMethodRidge(e.target.checked)} disabled={!includeAdvanced} />
                    <span>Ridge</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLasso} onChange={(e) => setMethodLasso(e.target.checked)} disabled={!includeAdvanced} />
                    <span>Lasso</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRF} onChange={(e) => setMethodRF(e.target.checked)} disabled={!includeAdvanced} />
                    <span>RF</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodXGB} onChange={(e) => setMethodXGB(e.target.checked)} disabled={!includeAdvanced} />
                    <span>XGB</span>
                  </label>
                </div>
                <div className="muted" style={{ marginTop: 6 }}>
                  {t("提示：在图表页面只能隐藏已生成的曲线。若需查看某高级模型的曲线，请在此处勾选并重新生成。", "Tip: You can only toggle visibility on the Forecast page. To generate curves for an advanced model, check it here and regenerate.")}
                </div>
              </div>
              <div className="field">
                <label>{t("时间列名 (选填)", "Time Column (Optional)")}</label>
                <input
                  value={timeCol}
                  onChange={(e) => setTimeCol(e.target.value)}
                  placeholder="e.g.：Date / date"
                  list={preview?.columns?.length ? "forecastpro-columns" : undefined}
                />
              </div>
              <div className="field">
                <label>{t("目标变量列名 (选填)", "Target Column (Optional)")}</label>
                <input
                  value={demandCol}
                  onChange={(e) => setDemandCol(e.target.value)}
                  placeholder="e.g. demand"
                  list={preview?.columns?.length ? "forecastpro-columns" : undefined}
                />
              </div>
              {preview?.columns?.length ? (
                <datalist id="forecastpro-columns">
                  {preview.columns.map((c) => (
                    <option key={String(c)} value={String(c)} />
                  ))}
                </datalist>
              ) : null}
            </div>

            <div className="row" style={{ marginTop: 12, alignItems: "center" }}>
              <button type="submit" disabled={loading}>
                {loading ? t("运行中...", "Running...") : t("生成预测", "Generate Forecast")}
              </button>
              <div className="muted">
                {normalizeBaseUrl(apiBase) || "—"} {apiStatus ? `· ${apiStatus}` : ""}
              </div>
            </div>

            {error ? <div style={{ color: "#D84B4B", marginTop: 10 }}>{error}</div> : null}
          </form>
        </div>
      </div>
    )
  }

  if (mode === "preview") {
    return (
      <div className="panel">
        <div className="card">
          <h2>{t("数据预览", "Data Preview")}</h2>
          <div className="muted" style={{ marginBottom: 10 }}>
            {t("上传文件后点击“预览数据”，系统将展示前 30 行并自动检测时间列与目标列。", "Click 'Preview Data' after uploading, system will show top 30 rows and auto-detect Time and Target columns.")}
          </div>

          <div className="formGrid">
            <div className="field">
              <label>{t("数据文件 (CSV / Excel)", "Data File (CSV / Excel)")}</label>
              <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file ? (
                <div className="muted" style={{ marginTop: 6 }}>
                  {file.name} · {Math.max(1, Math.round(file.size / 1024))} KB
                </div>
              ) : null}
            </div>

            <div className="field">
              <label>{t("API 地址", "API Base URL")}</label>
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://<backend>.up.railway.app" />
            </div>

            <div className="field" style={{ gridColumn: "span 12" }}>
              <label>{t("数据主题与业务场景（用于报告编写）", "Data Theme & Scenario (used for Report writing)")}</label>
              <div className="row" style={{ marginTop: 6 }}>
                <div className="field" style={{ minWidth: 220 }}>
                  <label>{t("数据集名称 (选填)", "Dataset Name (Optional)")}</label>
                  <input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="e.g. 2026Q2_Sales_East" />
                </div>
                <div className="field" style={{ minWidth: 220 }}>
                  <label>{t("主题/场景", "Theme/Scenario")}</label>
                  <select value={datasetTheme} onChange={(e) => setDatasetTheme(e.target.value)}>
                    <option value="">{t("未选择", "Unselected")}</option>
                    <option value="retail_inventory">{t("零售库存/补货", "Retail Inventory / Replenishment")}</option>
                    <option value="sales_operations">{t("销售运营", "Sales Operations")}</option>
                    <option value="workforce">{t("排班/人力配置", "Workforce Scheduling")}</option>
                    <option value="facility_climate">{t("设施环控/能耗", "Facility Climate Control")}</option>
                    <option value="marketing">{t("营销活动", "Marketing Campaigns")}</option>
                    <option value="other">{t("其他", "Other")}</option>
                  </select>
                </div>
              </div>
              <div style={{ marginTop: 8 }}>
                <label>{t("补充说明/分析要求 (选填)", "Data Description (Optional)")}</label>
                <textarea
                  value={dataDescription}
                  onChange={(e) => setDataDescription(e.target.value)}
                  placeholder={t("e.g.：这是华东区某零食每日销量，目标是降低缺货并控制库存周转率。请提供备货建议。", "e.g.: Daily sales data for snacks in East China. Target is to reduce stockouts and control inventory turnover.")}
                  rows={3}
                  style={{ width: "100%", resize: "vertical" }}
                />
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                {t("这些信息将包含在报告封面与摘要中，帮助 LLM 提供更贴合业务的建议。", "This info will be included in the report cover and executive summary to provide more contextual advice.")}
              </div>
            </div>

            <div className="field">
              <label>{t("时间列名 (选填)", "Time Column (Optional)")}</label>
              <input
                value={timeCol}
                onChange={(e) => setTimeCol(e.target.value)}
                placeholder="e.g.：Date / date"
                list={preview?.columns?.length ? "forecastpro-columns" : undefined}
              />
            </div>
            <div className="field">
              <label>{t("目标变量列名 (选填)", "Target Column (Optional)")}</label>
              <input
                value={demandCol}
                onChange={(e) => setDemandCol(e.target.value)}
                placeholder="e.g. demand"
                list={preview?.columns?.length ? "forecastpro-columns" : undefined}
              />
            </div>
            {preview?.columns?.length ? (
              <datalist id="forecastpro-columns">
                {preview.columns.map((c) => (
                  <option key={String(c)} value={String(c)} />
                ))}
              </datalist>
            ) : null}

            <div className="field" style={{ gridColumn: "span 12" }}>
              <label>{t("预览 (前 30 行)", "Preview (Top 30 Rows)")}</label>
              <div className="row" style={{ marginTop: 6, alignItems: "center", justifyContent: "space-between" }}>
                <div className="row" style={{ alignItems: "center" }}>
                  <button type="button" onClick={onPreview} disabled={!file || previewLoading}>
                    {previewLoading ? t("预览中...", "Previewing...") : t("预览数据", "Preview Data")}
                  </button>
                  <div className="muted" style={{ marginLeft: 10 }}>
                    {normalizeBaseUrl(apiBase) || "—"} {apiStatus ? `· ${apiStatus}` : ""}
                  </div>
                </div>
                {preview?.detected ? (
                  <div className="muted">
                    {t("检测到列：时间", "Detected: Time")} {preview?.detected?.time_col ? String(preview.detected.time_col) : "—"} · {t("目标", "Target")}{" "}
                    {preview?.detected?.demand_col ? String(preview.detected.demand_col) : "—"}
                  </div>
                ) : null}
              </div>

              {previewError ? <div style={{ color: "#D84B4B", marginTop: 8 }}>{previewError}</div> : null}

              {preview?.rows?.length && preview?.columns?.length ? (
                <div
                  style={{
                    marginTop: 10,
                    overflow: "auto",
                    maxHeight: 420,
                    border: "1px solid rgba(16,50,66,0.14)",
                    borderRadius: 12,
                    background: "rgba(255,255,255,0.65)",
                  }}
                >
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
              ) : (
                <div className="muted" style={{ marginTop: 8 }}>
                  {t("选择文件并点击“预览数据”。", "Select file and click 'Preview Data'.")}
                </div>
              )}
            </div>

            <div className="field" style={{ gridColumn: "span 12" }}>
              <div className="row" style={{ marginTop: 6, alignItems: "center", justifyContent: "space-between" }}>
                <button type="button" onClick={() => router.push("/upload")} disabled={!file}>
                  {t("去生成预测", "Go to Generate Forecast")}
                </button>
                <div className="muted">{t("生成预测需要在 Upload 页面设置频率、测试集比例、模型等参数。", "Generate Forecast requires setting frequency, test ratio, and models on the Upload page.")}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="card">
        <h2>{t("未来预测结果", "Future Forecast")}</h2>
        <div className="row" style={{ marginTop: 12, alignItems: "center", justifyContent: "space-between" }}>
          <div className="row">
            <button
              type="button"
              onClick={() => {
                try {
                  const d = sessionStorage.getItem("forecastpro:lastResult")
                  if (d) {
                    downloadText("forecast_result.json", JSON.stringify(JSON.parse(d), null, 2), "application/json;charset=utf-8")
                  }
                } catch {}
              }}
              disabled={!data}
            >
              {t("下载 JSON 结果", "Download JSON")}
            </button>
            <button type="button" onClick={() => router.push("/report")} disabled={!data}>
              {t("打开 Report 报告", "Open Report")}
            </button>
          </div>
        </div>
        <div className="row" style={{ marginBottom: 8, marginTop: 12 }}>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodEts} disabled onChange={(e) => setMethodEts(e.target.checked)} />
            <span>ETS</span>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodArima} disabled onChange={(e) => setMethodArima(e.target.checked)} />
            <span>ARIMA</span>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodNaive} onChange={(e) => setMethodNaive(e.target.checked)} />
            <span>Naïve</span>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodSNaive} onChange={(e) => setMethodSNaive(e.target.checked)} />
            <span>Seasonal Naïve</span>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodMA} disabled onChange={(e) => setMethodMA(e.target.checked)} />
            <span>MA</span>
          </label>
          {includeAdvanced ? (
            <>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={methodLR} onChange={(e) => setMethodLR(e.target.checked)} />
                <span>OLS</span>
              </label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={methodRidge} onChange={(e) => setMethodRidge(e.target.checked)} />
                <span>Ridge</span>
              </label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={methodLasso} onChange={(e) => setMethodLasso(e.target.checked)} />
                <span>Lasso</span>
              </label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={methodRF} onChange={(e) => setMethodRF(e.target.checked)} />
                <span>RF</span>
              </label>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={methodXGB} onChange={(e) => setMethodXGB(e.target.checked)} />
                <span>XGB</span>
              </label>
            </>
          ) : null}
        </div>

        {requestedMethods.length ? (
          <div className="muted" style={{ margin: "6px 0 10px" }}>
            {t("已返回方法：", "Returned Methods: ")}{Object.keys(data?.future_by_method || {}).join(", ") || "—"}
            <div style={{ marginTop: 6, color: "#D84B4B" }}>
              {(() => {
                const fbm = data?.future_by_method || {}
                const missing = requestedMethods.filter((m) => !fbm?.[m])
                if (!missing.length) return null
                return (
                  <>
                    {t("未生成/未返回的方法：", "Un-generated/un-returned methods:")}
                    {missing
                      .map((m) => {
                        const reason = futureErrors?.[m]
                        return `${m}（${reason ? String(reason).slice(0, 80) : t("请回到 Upload 勾选并重新生成", "Please return to Upload, check it, and regenerate")}）`
                      })
                      .join("；") || "—"}
                  </>
                )
              })()}
            </div>
          </div>
        ) : null}

        <div className="muted" style={{ margin: "4px 0 10px" }}>
          {t("Chart 1：回测对比（前 80% Train / 后 20% Test，使用 Test Set Forecast 进行对比，Future period 不参与误差评估）", "Chart 1: Backtest Comparison (First 80% Train / Last 20% Test, Test Set Forecast used for comparison, Future period not used for error eval)")}
        </div>
        <div
          className="svgwrap"
          ref={chartWrapRef}
          style={{ position: "relative" }}
          onMouseMove={onChartMove}
          onMouseLeave={() => {
            setHover(null)
            setHoverIndex(null)
            setHoverXY(null)
            setActiveMethod(null)
          }}
        >
          {hoverInfo?.date && hoverXY ? (
            <div
              style={{
                position: "absolute",
                left: clamp(hoverXY.x + 10, 10, 520),
                top: clamp(hoverXY.y + 10, 10, 190),
                padding: "8px 10px",
                borderRadius: 12,
                background: "rgba(255,255,255,0.92)",
                border: "1px solid rgba(16,50,66,0.14)",
                boxShadow: "0 10px 26px rgba(11,31,42,0.14)",
                backdropFilter: "blur(10px)",
                fontSize: 12,
                pointerEvents: "none",
              }}
            >
              <div className="mono" style={{ marginBottom: 6 }}>
                {hoverInfo.date}
              </div>
              {(hoverInfo.values || []).slice(0, 8).map((v) => (
                <div key={v.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 999, background: v.color }} />
                    <span>{v.label}</span>
                  </span>
                  <span className="mono">{formatNumber(v.value)}</span>
                </div>
              ))}
            </div>
          ) : null}

          <svg viewBox="0 0 760 260" width="100%" height="260" preserveAspectRatio="none">
            {chart.domain ? (
              <>
                {[0, 1, 2, 3].map((i) => {
                  const min = chart.domain.min
                  const max = chart.domain.max
                  const v = min + ((max - min) * i) / 3
                  const y = yForValue(v, chart.domain, chart.height, chart.pad)
                  return (
                    <g key={`yt_${i}`}>
                      <line x1={chart.pad} y1={y} x2={chart.width - chart.pad} y2={y} stroke="rgba(16,50,66,0.08)" />
                      <text x={10} y={y + 4} fontSize="11" fill="rgba(21,34,42,0.65)" className="mono">
                        {formatNumber(v)}
                      </text>
                    </g>
                  )
                })}
                <text x={12} y={18} fontSize="12" fill="rgba(21,34,42,0.72)">
                  {demandLabel}
                </text>
                <text x={chart.width - chart.pad} y={chart.height - 8} fontSize="12" fill="rgba(21,34,42,0.72)" textAnchor="end">
                  {t("时间", "Time")}
                </text>
                <text x={chart.pad} y={chart.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="start" className="mono">
                  {history?.dates?.length ? String(history.dates[0]).slice(0, 10) : "—"}
                </text>
                <text x={chart.width - chart.pad - 44} y={chart.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="end" className="mono">
                  {forecast?.dates?.length ? String(forecast.dates[forecast.dates.length - 1]).slice(0, 10) : "—"}
                </text>
                {chart.splitIndex !== null && chart.splitIndex !== undefined ? (
                  <line
                    x1={xForIndex(chart.splitIndex, chart.width, chart.pad, chart.totalPoints)}
                    y1={chart.pad}
                    x2={xForIndex(chart.splitIndex, chart.width, chart.pad, chart.totalPoints)}
                    y2={chart.height - chart.pad}
                    stroke="rgba(16,50,66,0.10)"
                    strokeDasharray="4 4"
                  />
                ) : null}
              </>
            ) : null}

            {chart.actualPoints ? <polyline points={chart.actualPoints} fill="none" stroke="#15222A" strokeWidth="2.5" opacity="0.90" /> : null}
            {chart.fittedPoints ? (
              <polyline
                points={chart.fittedPoints}
                fill="none"
                stroke="rgba(136,152,170,0.95)"
                strokeWidth="2.2"
                opacity="0.75"
                strokeDasharray="6 6"
              />
            ) : null}

            {(chart.series || []).map((s) =>
              s.fittedPoints || s.testPoints ? (
                <g key={s.method}>
                {s.fittedPoints ? (
                  <polyline
                    points={s.fittedPoints}
                    fill="none"
                    stroke={s.color}
                    strokeWidth="2.2"
                    opacity="0.45"
                    strokeDasharray="3 7"
                  />
                ) : null}
                  {s.testPoints ? (
                    <>
                      <polyline points={s.testPoints} fill="none" stroke={s.color} strokeWidth="3" opacity="0.95" strokeDasharray={s.dash || "8 6"}>
                        <title>{s.label || s.method}</title>
                      </polyline>
                      <polyline
                        points={s.testPoints}
                        fill="none"
                        stroke="transparent"
                        strokeWidth="16"
                        pointerEvents="stroke"
                        onMouseMove={(e) => {
                          setActiveMethod(s.method)
                          setHoverFromEvent(e, s)
                        }}
                        onMouseEnter={(e) => {
                          setActiveMethod(s.method)
                          setHoverFromEvent(e, s)
                        }}
                        onMouseLeave={() => {
                          setActiveMethod(null)
                          setHover(null)
                        }}
                      />
                    </>
                  ) : null}
                </g>
              ) : null
            )}
            {hoverInfo?.xSvg !== undefined && chart.domain ? (
              <line x1={hoverInfo.xSvg} y1={chart.pad} x2={hoverInfo.xSvg} y2={chart.height - chart.pad} stroke="rgba(16,50,66,0.16)" />
            ) : null}
          </svg>
        </div>

        <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
          <div className="muted">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
              <span style={{ display: "inline-block", width: 12, height: 3, background: "#15222A", borderRadius: 999 }} />
              {t("历史实际值", "Historical Actual")}
            </span>
            {chart.fittedPoints ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                <span style={{ display: "inline-block", width: 12, height: 3, background: "rgba(136,152,170,0.95)", borderRadius: 999 }} />
                {t("拟合值（最佳模型）", "Fitted (Best Model)")}
              </span>
            ) : null}
            {(chart.series || []).map((s) => (
              <span key={`lg_${s.method}`} style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                <span style={{ display: "inline-block", width: 12, height: 3, background: s.color, borderRadius: 999 }} />
                {s.label || s.method} {t("（测试集预测）", "(Test Set Forecast)")}
              </span>
            ))}
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={{ display: "inline-block", width: 12, height: 3, background: "rgba(16,50,66,0.10)", borderRadius: 999 }} />
              {t("训练/测试分界", "Train/Test Split")}
            </span>
          </div>
          <div className="muted mono">
            {history?.dates?.length ? String(history.dates[0]).slice(0, 10) : "—"} →{" "}
            {history?.dates?.length ? String(history.dates[history.dates.length - 1]).slice(0, 10) : "—"}
          </div>
        </div>

        {metricRows.length ? (
          <>
            <div style={{ height: 12 }} />
            <div className="muted" style={{ marginBottom: 8 }}>
              {t("模型指标（MA/ETS/ARIMA 为必选项；并提供最佳模型）：样本内与保留集误差", "Metrics (MA / ETS / ARIMA are mandatory; Also providing Best): in-sample and hold-out MAE / RMSE / MAPE")}
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Method", "Model Name", "in-sample MAE", "in-sample RMSE", "in-sample MAPE", "hold-out MAE", "hold-out RMSE", "hold-out MAPE", "Risk"].map((h) => (
                      <th
                        key={h}
                        style={{
                          textAlign: "left",
                          padding: "10px 8px",
                          borderBottom: "1px solid rgba(16,50,66,0.12)",
                          fontSize: 12,
                          color: "rgba(21,34,42,0.72)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metricRows.map((r) => {
                    const row = r.row || {}
                    const isBest = bestName && String(row?.model) === String(bestName)
                    const cell = (v, pct) => {
                      if (v === null || v === undefined || v === "") return "—"
                      const n = Number(v)
                      if (!Number.isFinite(n)) return String(v)
                      return pct ? `${n.toFixed(3)}%` : n.toFixed(3)
                    }
                    return (
                      <tr key={r.key} style={isBest ? { background: "rgba(42,140,168,0.08)" } : undefined}>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)", fontWeight: 700 }}>{r.label}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }} className="mono">
                          {String(r.model)}
                        </td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_MAE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_RMSE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_MAPE, true)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.MAE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.RMSE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.MAPE, true)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{String(row?.overfitting_risk ?? "—")}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : null}

        <div style={{ height: 18 }} />
        <div className="muted" style={{ margin: "4px 0 10px" }}>
          {t("Chart 2：最佳模型（历史 + 拟合值 + 测试集预测 + 未来预测）", "Chart 2: Best Model (History + Fitted + Test Set Forecast + Future Forecast)")}
        </div>
        <div
          className="svgwrap"
          ref={chartWrapBestRef}
          style={{ position: "relative" }}
          onMouseMove={onChartMoveBest}
          onMouseLeave={() => {
            setHoverIndexBest(null)
            setHoverXYBest(null)
          }}
        >
          {hoverInfoBest?.date && hoverXYBest ? (
            <div
              style={{
                position: "absolute",
                left: clamp(hoverXYBest.x + 10, 10, 520),
                top: clamp(hoverXYBest.y + 10, 10, 190),
                padding: "8px 10px",
                borderRadius: 12,
                background: "rgba(255,255,255,0.92)",
                border: "1px solid rgba(16,50,66,0.14)",
                boxShadow: "0 10px 26px rgba(11,31,42,0.14)",
                backdropFilter: "blur(10px)",
                fontSize: 12,
                pointerEvents: "none",
              }}
            >
              <div className="mono" style={{ marginBottom: 6 }}>
                {hoverInfoBest.date}
              </div>
              {(hoverInfoBest.values || []).slice(0, 6).map((v) => (
                <div key={v.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 999, background: v.color }} />
                    <span>{v.label}</span>
                  </span>
                  <span className="mono">{formatNumber(v.value)}</span>
                </div>
              ))}
            </div>
          ) : null}

          <svg viewBox="0 0 760 260" width="100%" height="260" preserveAspectRatio="none">
            {chartBest.domain ? (
              <>
                {[0, 1, 2, 3].map((i) => {
                  const min = chartBest.domain.min
                  const max = chartBest.domain.max
                  const v = min + ((max - min) * i) / 3
                  const y = yForValue(v, chartBest.domain, chartBest.height, chartBest.pad)
                  return (
                    <g key={`ytb_${i}`}>
                      <line x1={chartBest.pad} y1={y} x2={chartBest.width - chartBest.pad} y2={y} stroke="rgba(16,50,66,0.08)" />
                      <text x={10} y={y + 4} fontSize="11" fill="rgba(21,34,42,0.65)" className="mono">
                        {formatNumber(v)}
                      </text>
                    </g>
                  )
                })}
                <text x={12} y={18} fontSize="12" fill="rgba(21,34,42,0.72)">
                  {demandLabel}
                </text>
                <text x={chartBest.width - chartBest.pad} y={chartBest.height - 8} fontSize="12" fill="rgba(21,34,42,0.72)" textAnchor="end">
                  {t("时间", "Time")}
                </text>
                <text x={chartBest.pad} y={chartBest.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="start" className="mono">
                  {history?.dates?.length ? String(history.dates[0]).slice(0, 10) : "—"}
                </text>
                <text x={chartBest.width - chartBest.pad - 44} y={chartBest.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="end" className="mono">
                  {forecast?.dates?.length ? String(forecast.dates[forecast.dates.length - 1]).slice(0, 10) : "—"}
                </text>
                {chartBest.splitIndex !== null && chartBest.splitIndex !== undefined ? (
                  <line
                    x1={xForIndex(chartBest.splitIndex, chartBest.width, chartBest.pad, chartBest.totalPoints)}
                    y1={chartBest.pad}
                    x2={xForIndex(chartBest.splitIndex, chartBest.width, chartBest.pad, chartBest.totalPoints)}
                    y2={chartBest.height - chartBest.pad}
                    stroke="rgba(16,50,66,0.14)"
                    strokeDasharray="4 4"
                  />
                ) : null}
                {chartBest.actualLen > 1 ? (
                  <line
                    x1={xForIndex(chartBest.actualLen - 1, chartBest.width, chartBest.pad, chartBest.totalPoints)}
                    y1={chartBest.pad}
                    x2={xForIndex(chartBest.actualLen - 1, chartBest.width, chartBest.pad, chartBest.totalPoints)}
                    y2={chartBest.height - chartBest.pad}
                    stroke="rgba(108,67,255,0.55)"
                  />
                ) : null}
              </>
            ) : null}

            {chartBest.areaPoints ? <polygon points={chartBest.areaPoints} fill="rgba(108,67,255,0.14)" stroke="none" /> : null}
            {chartBest.actualPoints ? <polyline points={chartBest.actualPoints} fill="none" stroke="#15222A" strokeWidth="2.5" opacity="0.90" /> : null}
            {chartBest.testPoints ? <polyline points={chartBest.testPoints} fill="none" stroke="#6C43FF" strokeWidth="2.8" opacity="0.80" strokeDasharray="4 6" /> : null}
            {chartBest.fittedPoints ? (
              <polyline
                points={chartBest.fittedPoints}
                fill="none"
                stroke="rgba(136,152,170,0.95)"
                strokeWidth="2.2"
                opacity="0.75"
                strokeDasharray="6 6"
              />
            ) : null}
            {chartBest.forecastPoints ? (
              <polyline points={chartBest.forecastPoints} fill="none" stroke="#6C43FF" strokeWidth="3" opacity="0.95" strokeDasharray="10 6" />
            ) : null}
            {hoverInfoBest?.xSvg !== undefined && chartBest.domain ? (
              <line x1={hoverInfoBest.xSvg} y1={chartBest.pad} x2={hoverInfoBest.xSvg} y2={chartBest.height - chartBest.pad} stroke="rgba(16,50,66,0.16)" />
            ) : null}
          </svg>
        </div>

        <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
          <div className="muted">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
              <span style={{ display: "inline-block", width: 12, height: 3, background: "#15222A", borderRadius: 999 }} />
              {t("历史实际值", "Historical Actual")}
            </span>
            {chartBest.testPoints ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                <span style={{ display: "inline-block", width: 12, height: 3, background: "#6C43FF", borderRadius: 999 }} />
                {t("测试集预测（最佳模型）", "Test Set Forecast (Best Model)")}
              </span>
            ) : null}
            {chartBest.fittedPoints ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                <span style={{ display: "inline-block", width: 12, height: 3, background: "rgba(136,152,170,0.95)", borderRadius: 999 }} />
                {t("拟合值（最佳模型）", "Fitted (Best Model)")}
              </span>
            ) : null}
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
              <span style={{ display: "inline-block", width: 12, height: 3, background: "#6C43FF", borderRadius: 999 }} />
              {t("最佳预测", "Best Forecast")}{bestName ? ` (${String(bestName)})` : ""}
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span style={{ display: "inline-block", width: 12, height: 8, background: "rgba(108,67,255,0.14)", borderRadius: 6 }} />
              {t("置信区间", "Confidence Interval")}
            </span>
          </div>
          <div className="muted mono">
            {history?.dates?.length ? String(history.dates[0]).slice(0, 10) : "—"} →{" "}
            {forecast?.dates?.length ? String(forecast.dates[forecast.dates.length - 1]).slice(0, 10) : "—"}
          </div>
        </div>
      </div>
    </div>
  )
}
