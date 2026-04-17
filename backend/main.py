import os
import sys
import tempfile
import json
import base64
import urllib.request
import urllib.error
import re
from typing import Optional, Any, List, Dict
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecastpro import ForecastProAgent


app = FastAPI(title="ForecastPro API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    def _flag(name: str) -> bool:
        return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes"}

    return {
        "ok": True,
        "config": {
            "FORECASTPRO_FAST": _flag("FORECASTPRO_FAST"),
            "FORECASTPRO_SKIP_ARIMA": _flag("FORECASTPRO_SKIP_ARIMA"),
            "FORECASTPRO_ARIMA_MAX_SECONDS": os.getenv("FORECASTPRO_ARIMA_MAX_SECONDS"),
            "FORECASTPRO_ARIMA_MAXITER": os.getenv("FORECASTPRO_ARIMA_MAXITER"),
        },
    }


class DeepSeekChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: float = 0.7
    response_format: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 60
    base_url: Optional[str] = None


def _read_preview(file_bytes: bytes, suffix: str, nrows: int = 10) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(BytesIO(file_bytes), nrows=int(nrows))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(file_bytes), nrows=int(nrows))
    raise ValueError(f"不支持的文件格式: {suffix}")


def _is_image_suffix(suffix: str) -> bool:
    return str(suffix or "").lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}


def _extract_table_from_image(file_bytes: bytes, nrows: int = 30):
    try:
        from PIL import Image
    except Exception as e:
        raise ValueError("缺少 Pillow 依赖，请安装后重试：pip install pillow") from e
    try:
        import pytesseract
    except Exception as e:
        raise ValueError("缺少 pytesseract 依赖，请安装后重试：pip install pytesseract") from e
    try:
        import numpy as np
    except Exception as e:
        raise ValueError("缺少 numpy 依赖，无法执行OCR") from e

    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    arr = np.array(image)

    cv2 = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        proc = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
    else:
        proc = image.convert("L")

    data = pytesseract.image_to_data(
        proc,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )

    n = len(data.get("text", []))
    lines = {}
    conf_vals = []
    for i in range(n):
        text = str(data.get("text", [""])[i] or "").strip()
        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if conf >= 0:
            conf_vals.append(conf)
        if not text or conf < 25:
            continue
        key = (
            int(data.get("block_num", [0])[i] or 0),
            int(data.get("par_num", [0])[i] or 0),
            int(data.get("line_num", [0])[i] or 0),
        )
        lines.setdefault(key, []).append(text)

    line_texts = [" ".join(parts).strip() for _, parts in sorted(lines.items()) if parts]
    if not line_texts:
        raise ValueError("OCR未识别到有效文本，请尝试更清晰截图（提高分辨率/对比度）")

    split_rows = []
    for line in line_texts:
        if not line:
            continue
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if str(c).strip()]
        elif "\t" in line:
            cells = [c.strip() for c in line.split("\t") if str(c).strip()]
        else:
            cells = [c.strip() for c in re.split(r"\s{2,}", line) if str(c).strip()]
        if len(cells) >= 2:
            split_rows.append(cells)

    if not split_rows:
        # 兜底：至少返回一列文本，允许前端人工复制修正
        df = pd.DataFrame({"raw_text": line_texts[: int(max(1, nrows))]})
        meta = {
            "ocr_engine": "pytesseract",
            "ocr_mean_confidence": float(sum(conf_vals) / len(conf_vals)) if conf_vals else None,
            "line_count": int(len(line_texts)),
            "table_quality": "low",
        }
        return df, meta

    max_cols = max(len(r) for r in split_rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in split_rows]

    def _is_header_row(cells):
        if not cells:
            return False
        alpha_like = 0
        for c in cells:
            s = str(c or "").strip()
            if not s:
                continue
            if re.search(r"[A-Za-z\u4e00-\u9fff]", s):
                alpha_like += 1
        return alpha_like >= max(1, len(cells) // 2)

    if len(normalized) >= 2 and _is_header_row(normalized[0]):
        headers = []
        used = set()
        for i, h in enumerate(normalized[0]):
            name = str(h or "").strip() or f"col_{i+1}"
            base = name
            k = 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            headers.append(name)
        data_rows = normalized[1:]
    else:
        headers = [f"col_{i+1}" for i in range(max_cols)]
        data_rows = normalized

    df = pd.DataFrame(data_rows, columns=headers)
    df = df.dropna(how="all")
    for c in df.columns:
        df[c] = df[c].map(lambda x: str(x).strip() if x is not None else "")

    if nrows and int(nrows) > 0:
        df = df.head(int(nrows))

    mean_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else None
    quality = "high" if (mean_conf is not None and mean_conf >= 75) else ("medium" if (mean_conf is not None and mean_conf >= 55) else "low")
    meta = {
        "ocr_engine": "pytesseract",
        "ocr_mean_confidence": mean_conf,
        "line_count": int(len(line_texts)),
        "table_quality": quality,
    }
    return df, meta


def _safe_json_value(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return str(v)
    return v


def _df_to_rows(df: pd.DataFrame, nrows: int = 30):
    rows = []
    raw_rows = df.head(int(max(1, nrows))).to_dict(orient="records")
    for r in raw_rows:
        rows.append({str(k): _safe_json_value(v) for k, v in r.items()})
    return rows


def _llm_refine_ocr_table(
    df: pd.DataFrame,
    user_prompt: str,
    llm_model: Optional[str],
    llm_api_key: Optional[str],
    llm_base_url: Optional[str],
    llm_timeout_seconds: int,
):
    columns = [str(c) for c in df.columns.tolist()]
    rows = _df_to_rows(df, nrows=min(200, int(max(1, len(df)))))
    payload = {
        "task": "refine_ocr_table",
        "user_prompt": str(user_prompt or "").strip(),
        "input_table": {"columns": columns, "rows": rows},
        "rules": [
            "只允许重排/拆分/合并单元格文本，不允许编造任何数字或日期",
            "如果无法确定某个单元格内容，输出 null 或空字符串",
            "尽量识别时间列与目标列（数值列），并给出列名建议",
            "保持行的顺序，去掉明显的页眉/页脚/水印噪声行（如无法判断则保留）",
            "输出严格 JSON",
        ],
        "output_schema": {
            "refined": {"columns": ["string"], "rows": [{"any": "any"}]},
            "detected": {"time_col": "string|null", "demand_col": "string|null"},
            "warnings": ["string"],
        },
    }
    obj = _deepseek_chat_json(
        messages=[
            {"role": "system", "content": "你是 OCR 表格清洗助手。你只做结构化清洗，不得编造数据。输出严格JSON。"},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        timeout_seconds=int(llm_timeout_seconds),
    )
    refined = obj.get("refined") if isinstance(obj, dict) else None
    if not isinstance(refined, dict):
        raise ValueError("LLM未返回 refined 表格")
    refined_rows = refined.get("rows")
    if not isinstance(refined_rows, list) or not refined_rows:
        raise ValueError("LLM refined.rows 为空")
    refined_cols = refined.get("columns")
    rdf = pd.DataFrame(refined_rows)
    if isinstance(refined_cols, list) and refined_cols:
        safe_cols = []
        used = set()
        for i, c in enumerate(refined_cols):
            name = str(c or "").strip() or f"col_{i+1}"
            base = name
            k = 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            safe_cols.append(name)
        try:
            rdf = rdf.reindex(columns=safe_cols)
        except Exception:
            pass
    rdf = rdf.dropna(how="all")
    for c in rdf.columns:
        rdf[c] = rdf[c].map(lambda x: "" if x is None else str(x).strip())
    detected = obj.get("detected") if isinstance(obj, dict) else None
    if not isinstance(detected, dict):
        detected = {"time_col": None, "demand_col": None}
    warnings = obj.get("warnings") if isinstance(obj, dict) else None
    if not isinstance(warnings, list):
        warnings = []
    return rdf, {"detected": detected, "warnings": warnings}


def _to_bool(v, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _deepseek_chat_json(
    messages,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 60,
):
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("未配置 LLM API KEY")
    key = str(key).strip().encode("latin-1", "ignore").decode("latin-1")
    base = str(base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).strip().rstrip("/")
    payload = {
        "model": model or os.getenv("LLM_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise ValueError(f"DeepSeek请求失败: HTTP {getattr(e, 'code', 'ERR')} - {detail}") from e
    except Exception as e:
        raise ValueError(f"DeepSeek请求失败: {e}") from e
    try:
        obj = json.loads(body)
        txt = obj["choices"][0]["message"]["content"]
        return json.loads(str(txt).strip().strip("`"))
    except Exception as e:
        raise ValueError(f"DeepSeek响应解析失败: {e}; body={body[:500]}") from e


def _deepseek_chat_raw(
    messages,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 60,
):
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("未配置 LLM API KEY")
    key = str(key).strip().encode("latin-1", "ignore").decode("latin-1")
    base = str(base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).strip().rstrip("/")
    payload = {
        "model": model or os.getenv("LLM_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": float(temperature),
    }
    if response_format:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise ValueError(f"DeepSeek请求失败: HTTP {getattr(e, 'code', 'ERR')} - {detail}") from e
    except Exception as e:
        raise ValueError(f"DeepSeek请求失败: {e}") from e
    try:
        return json.loads(body)
    except Exception as e:
        raise ValueError(f"DeepSeek响应解析失败: {e}; body={body[:500]}") from e


@app.post("/api/deepseek/chat")
async def deepseek_chat(req: DeepSeekChatRequest):
    """便于在没有前端的情况下，用 curl 直接测试 DeepSeek Chat 接口。"""
    try:
        obj = _deepseek_chat_raw(
            messages=req.messages,
            model=req.model,
            temperature=req.temperature,
            response_format=req.response_format,
            timeout_seconds=req.timeout_seconds,
            base_url=req.base_url,
        )
        content = None
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            content = None
        return {"ok": True, "content": content, "raw": obj}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "deepseek_call_failed", "message": str(e)})

def _detect_columns(df: pd.DataFrame, time_col: Optional[str], demand_col: Optional[str]):
    cols = [str(c) for c in df.columns.tolist()]

    def _valid(col: Optional[str]) -> bool:
        return bool(col) and col in df.columns

    if _valid(time_col) and _valid(demand_col):
        return time_col, demand_col

    time_keywords = [
        "date",
        "time",
        "datetime",
        "timestamp",
        "txnDate",
        "交易日期",
        "日期",
        "时间",
        "月份",
        "月",
        "周",
        "year",
        "年",
    ]
    demand_keywords = [
        "demand",
        "sales",
        "quantity",
        "volume",
        "target",
        "y",
        "revenue",
        "consumption",
        "cost",
        "price",
        "销量",
        "销售",
        "需求",
        "数量",
        "金额",
        "收入",
        "用量",
        "成本",
        "价格",
    ]

    detected_time = time_col if _valid(time_col) else None
    if detected_time is None:
        for c in cols:
            if any(k.lower() in c.lower() for k in time_keywords):
                detected_time = c
                break
    if detected_time is None:
        best = (None, 0.0)
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            ratio = float(s.notna().mean()) if len(s) else 0.0
            if ratio > best[1]:
                best = (c, ratio)
        detected_time = best[0] if best[1] >= 0.6 else None

    detected_demand = demand_col if _valid(demand_col) else None
    if detected_demand is None:
        for c in cols:
            if detected_time is not None and c == detected_time:
                continue
            if any(k.lower() in c.lower() for k in demand_keywords):
                detected_demand = c
                break
    if detected_demand is None:
        best = (None, 0.0, 0.0)
        for c in df.columns:
            if detected_time is not None and c == detected_time:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            ratio = float(s.notna().mean()) if len(s) else 0.0
            std = float(s.dropna().std()) if s.notna().sum() > 1 else 0.0
            score = ratio * (1.0 + min(std, 1e6) / 1000.0)
            if ratio >= 0.6 and score > best[1]:
                best = (c, score, ratio)
        detected_demand = best[0]

    return detected_time, detected_demand


@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...),
    time_col: Optional[str] = Form(None),
    demand_col: Optional[str] = Form(None),
    use_llm: bool = Form(False),
    llm_provider: Optional[str] = Form("deepseek"),
    llm_model: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_timeout_seconds: int = Form(60),
    ocr_refine_with_llm: bool = Form(False),
):
    filename = file.filename or "upload.csv"
    suffix = os.path.splitext(filename)[1].lower()
    content = await file.read()

    ocr_meta = None
    llm_meta = {"enabled": bool(use_llm and ocr_refine_with_llm), "error": None, "error_code": None, "warnings": []}
    try:
        if _is_image_suffix(suffix):
            df_preview, ocr_meta = _extract_table_from_image(content, nrows=30)
            if bool(use_llm and ocr_refine_with_llm):
                try:
                    refined_df, refine_info = _llm_refine_ocr_table(
                        df_preview,
                        user_prompt="",
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        llm_timeout_seconds=int(llm_timeout_seconds),
                    )
                    df_preview = refined_df
                    llm_meta["warnings"] = list(refine_info.get("warnings") or [])
                except Exception as e:
                    llm_meta["error"] = str(e)
                    if "HTTP 402" in llm_meta["error"] or "Insufficient Balance" in llm_meta["error"] or "Payment Required" in llm_meta["error"]:
                        llm_meta["error_code"] = "insufficient_balance"
        else:
            df_preview = _read_preview(content, suffix, nrows=30)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "预览失败（文件格式或内容问题）",
                "message": str(e),
            },
        )

    detected_time_col, detected_demand_col = _detect_columns(df_preview, time_col, demand_col)
    columns = [str(c) for c in df_preview.columns.tolist()]

    def _safe(v):
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, pd.Timestamp):
            try:
                return v.isoformat()
            except Exception:
                return str(v)
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                pass
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="ignore")
            except Exception:
                return str(v)
        return v

    rows = []
    try:
        raw_rows = df_preview.head(30).to_dict(orient="records")
        for r in raw_rows:
            rows.append({str(k): _safe(v) for k, v in r.items()})
    except Exception:
        rows = []

    return {
        "ok": True,
        "filename": filename,
        "source": "ocr_image" if _is_image_suffix(suffix) else "file_table",
        "ocr_meta": ocr_meta,
        "llm": llm_meta,
        "columns": columns,
        "detected": {"time_col": detected_time_col, "demand_col": detected_demand_col},
        "rows": rows,
        "row_count": int(len(rows)),
    }


@app.post("/api/forecast")
async def forecast(
    file: UploadFile = File(...),
    freq: str = Form("D"),
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
    periods: int = Form(10),
    methods: Optional[str] = Form(None),  # 逗号分隔: naive,seasonal_naive,moving_average,ets,arima,(advanced...)
    include_advanced: bool = Form(False),
    time_col: Optional[str] = Form(None),
    demand_col: Optional[str] = Form(None),
    dataset_name: Optional[str] = Form(None),
    dataset_theme: Optional[str] = Form(None),
    data_description: Optional[str] = Form(None),
    language: str = Form("en"),
    use_llm: bool = Form(False),
    llm_for_forecast: bool = Form(True),
    llm_for_report: bool = Form(True),
    llm_provider: Optional[str] = Form("deepseek"),
    llm_model: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_timeout_seconds: int = Form(60),
):
    filename = file.filename or "upload.csv"
    suffix = os.path.splitext(filename)[1].lower()
    content = await file.read()

    df_preview = _read_preview(content, suffix)
    detected_time_col, detected_demand_col = _detect_columns(df_preview, time_col, demand_col)
    if detected_time_col is None or detected_demand_col is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "无法自动识别时间列或目标列",
                "columns": df_preview.columns.tolist(),
                "hint": "请在请求中传入 time_col 和 demand_col（或调整列名包含 date/demand/销量/日期 等关键词）",
            },
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            agent = ForecastProAgent(
                data_path=tmp_path,
                time_col=detected_time_col,
                demand_col=detected_demand_col,
                freq=freq,
                random_seed=random_seed,
                language=str(language).strip(),
            )
            agent.test_size = test_size
            agent.configure_llm(
                enabled=bool(use_llm),
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                timeout_seconds=int(llm_timeout_seconds),
            )

            agent.load_data(tmp_path)
            agent.prepare_data()
            agent.run_baseline_models()
            if include_advanced:
                agent.run_advanced_models()
            agent.evaluate_models()
            # 主报告基于auto
            llm_error = None
            llm_error_code = None
            if bool(use_llm) and bool(llm_for_forecast):
                try:
                    agent.generate_forecast(periods=int(periods), forecast_method="llm")
                except Exception as e:
                    llm_error = str(e)
                    if "HTTP 402" in llm_error or "Insufficient Balance" in llm_error or "Payment Required" in llm_error:
                        llm_error_code = "insufficient_balance"
                    agent.generate_forecast(periods=int(periods), forecast_method="auto")
            else:
                agent.generate_forecast(periods=int(periods), forecast_method="auto")
            main_forecast_results = agent._clean_for_json(agent.forecast_results)
            try:
                if dataset_name and str(dataset_name).strip():
                    agent.dataset_name_override = str(dataset_name).strip()
                if dataset_theme and str(dataset_theme).strip():
                    agent.dataset_theme = str(dataset_theme).strip()
                if data_description and str(data_description).strip():
                    agent.data_description = str(data_description).strip()
            except Exception:
                pass
            report = agent.generate_report(save_to_disk=False, use_llm=bool(use_llm and llm_for_report and not llm_error))
            # 额外方法的未来预测
            future_by_method = getattr(agent, 'future_by_method', {})
            if not future_by_method:
                future_by_method = {}
            else:
                future_by_method = dict(future_by_method)
            future_errors = {}
            method_list = []
            if methods:
                method_list = [m.strip() for m in methods.split(",") if m.strip()]
            else:
                method_list = ["naive", "seasonal_naive", "moving_average", "ets", "arima"]
            for m in method_list:
                if m not in future_by_method:
                    try:
                        agent.generate_forecast(periods=int(periods), forecast_method=m)
                        if agent.forecast_results:
                            future_by_method[m] = agent._clean_for_json(agent.forecast_results)
                    except Exception as e:
                        future_errors[m] = str(e)
            
            report['future_by_method'] = future_by_method
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "预测失败（数据格式或列识别问题）",
                    "message": str(e),
                    "detected": {"time_col": detected_time_col, "demand_col": detected_demand_col},
                    "columns": df_preview.columns.tolist(),
                },
            )

        y = agent.data[agent.demand_col].astype(float) if agent.data is not None else None
        if y is not None and len(y) > 0:
            tail_n = min(len(y), 120)
            y_tail = y.iloc[-tail_n:]
            fitted_best_tail = None
            fitted_by_method_tail = {}
            test_pred_best_tail = None
            test_pred_by_method_tail = {}
            method_to_model = {}
            try:
                best_name = getattr(agent, "best_model", None)
                train_size = int(len(agent.train_data["y"])) if agent.train_data and "y" in agent.train_data else None
                train_end_date = None
                test_start_date = None
                try:
                    if agent.train_data and agent.train_data.get("y") is not None and len(agent.train_data["y"]) > 0:
                        train_end_date = agent.train_data["y"].index[-1]
                    if agent.test_data and agent.test_data.get("y") is not None and len(agent.test_data["y"]) > 0:
                        test_start_date = agent.test_data["y"].index[0]
                except Exception:
                    train_end_date = None
                    test_start_date = None
                fitted = None
                if best_name and isinstance(best_name, str):
                    if (agent.baseline_models or {}).get(best_name) is not None:
                        fitted = (agent.baseline_models or {}).get(best_name, {}).get("fitted")
                    elif (agent.advanced_models or {}).get(best_name) is not None:
                        fitted = (agent.advanced_models or {}).get(best_name, {}).get("fitted")
                if fitted is not None:
                    try:
                        fitted_list = list(fitted)
                    except Exception:
                        fitted_list = []

                    idx = None
                    try:
                        if (agent.baseline_models or {}).get(best_name) is not None:
                            idx = agent.train_data["y"].index
                        elif (agent.advanced_models or {}).get(best_name) is not None and getattr(agent, "train_features", None) is not None:
                            idx = agent.train_features.get("y").index if agent.train_features.get("y") is not None else None
                    except Exception:
                        idx = None

                    fit_map = {}
                    if idx is not None:
                        n = min(len(idx), len(fitted_list))
                        for i in range(n):
                            try:
                                v = float(fitted_list[i])
                                fit_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                            except Exception:
                                fit_map[idx[i]] = None

                    fitted_best_tail = [fit_map.get(d, None) for d in y_tail.index]
            except Exception:
                fitted_best_tail = None

            try:
                best_name = getattr(agent, "best_model", None)
                preds = None
                idx = None
                if best_name and isinstance(best_name, str):
                    if (agent.baseline_models or {}).get(best_name) is not None:
                        preds = (agent.baseline_models or {}).get(best_name, {}).get("predictions")
                        idx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                    elif (agent.advanced_models or {}).get(best_name) is not None:
                        preds = (agent.advanced_models or {}).get(best_name, {}).get("predictions")
                        idx = None
                        try:
                            if getattr(agent, "test_features", None) is not None and agent.test_features.get("y") is not None:
                                idx = agent.test_features.get("y").index
                        except Exception:
                            idx = None
                        if idx is None:
                            idx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None

                pred_map = {}
                if preds is not None and idx is not None:
                    try:
                        preds_list = list(preds)
                    except Exception:
                        preds_list = []
                    n = min(len(idx), len(preds_list))
                    for i in range(n):
                        try:
                            v = float(preds_list[i])
                            pred_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                        except Exception:
                            pred_map[idx[i]] = None
                test_pred_best_tail = [pred_map.get(d, None) for d in y_tail.index]
            except Exception:
                test_pred_best_tail = None

            try:
                eval_df = agent.evaluation_results
                best_ets_variant = None
                if eval_df is not None and len(eval_df) > 0 and "model" in eval_df.columns:
                    ets_candidates = ["ets_simple", "ets_holt", "ets_holt_winters", "seasonal_ets", "ets"]
                    dfv = eval_df[eval_df["model"].isin(ets_candidates)]
                    if len(dfv) > 0 and "RMSE" in dfv.columns:
                        best_ets_variant = str(dfv.sort_values("RMSE").iloc[0]["model"])

                def _model_key_for_method(m: str) -> str:
                    mm = str(m or "").strip().lower()
                    if mm == "ets":
                        return best_ets_variant or "ets"
                    return mm

                def _fitted_for_model(model_key: str):
                    if not model_key:
                        return None
                    if (agent.baseline_models or {}).get(model_key) is not None:
                        return (agent.baseline_models or {}).get(model_key, {}).get("fitted")
                    if (agent.advanced_models or {}).get(model_key) is not None:
                        return (agent.advanced_models or {}).get(model_key, {}).get("fitted")
                    return None

                def _pred_for_model(model_key: str):
                    if not model_key:
                        return None
                    if (agent.baseline_models or {}).get(model_key) is not None:
                        return (agent.baseline_models or {}).get(model_key, {}).get("predictions")
                    if (agent.advanced_models or {}).get(model_key) is not None:
                        return (agent.advanced_models or {}).get(model_key, {}).get("predictions")
                    return None

                for m in (method_list or []):
                    mk = _model_key_for_method(m)
                    method_to_model[str(m)] = mk
                    fitted_m = _fitted_for_model(mk)
                    if fitted_m is None:
                        continue
                    try:
                        fitted_list = list(fitted_m)
                    except Exception:
                        fitted_list = []

                    idx = None
                    try:
                        if (agent.baseline_models or {}).get(mk) is not None:
                            idx = agent.train_data["y"].index
                        elif (agent.advanced_models or {}).get(mk) is not None and getattr(agent, "train_features", None) is not None:
                            idx = agent.train_features.get("y").index if agent.train_features.get("y") is not None else None
                    except Exception:
                        idx = None

                    fit_map = {}
                    if idx is not None:
                        n = min(len(idx), len(fitted_list))
                        for i in range(n):
                            try:
                                v = float(fitted_list[i])
                                fit_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                            except Exception:
                                fit_map[idx[i]] = None

                    fitted_by_method_tail[str(m)] = [fit_map.get(d, None) for d in y_tail.index]

                    pred_m = _pred_for_model(mk)
                    if pred_m is None:
                        continue
                    try:
                        preds_list = list(pred_m)
                    except Exception:
                        preds_list = []

                    pidx = None
                    try:
                        if (agent.baseline_models or {}).get(mk) is not None:
                            pidx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                        elif (agent.advanced_models or {}).get(mk) is not None and getattr(agent, "test_features", None) is not None:
                            pidx = agent.test_features.get("y").index if agent.test_features.get("y") is not None else None
                    except Exception:
                        pidx = None
                    if pidx is None:
                        pidx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None

                    pred_map = {}
                    if pidx is not None:
                        n = min(len(pidx), len(preds_list))
                        for i in range(n):
                            try:
                                v = float(preds_list[i])
                                pred_map[pidx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                            except Exception:
                                pred_map[pidx[i]] = None

                    test_pred_by_method_tail[str(m)] = [pred_map.get(d, None) for d in y_tail.index]
            except Exception:
                fitted_by_method_tail = {}
                method_to_model = {}
            history = {
                "dates": [d.to_pydatetime() for d in y_tail.index],
                "actual": [float(v) for v in y_tail.values],
                "train_size": int(len(agent.train_data["y"])) if agent.train_data and "y" in agent.train_data else None,
                "total_size": int(len(y)),
                "fitted_best": fitted_best_tail,
                "fitted_by_method": fitted_by_method_tail,
                "test_pred_best": test_pred_best_tail,
                "test_pred_by_method": test_pred_by_method_tail,
                "train_end_date": train_end_date.to_pydatetime() if "train_end_date" in locals() and train_end_date is not None else None,
                "test_start_date": test_start_date.to_pydatetime() if "test_start_date" in locals() and test_start_date is not None else None,
            }
        else:
            history = None

        forecast_results = main_forecast_results if "main_forecast_results" in locals() else (agent._clean_for_json(agent.forecast_results) if getattr(agent, "forecast_results", None) is not None else None)
        evaluation_results = agent._clean_for_json(agent.evaluation_results.to_dict("records")) if agent.evaluation_results is not None else None
        history = agent._clean_for_json(history) if history is not None else None

        # 每个方法单独导出到独立文件夹 (Excel为主，附带PNG)
        try:
            agent.export_method_folders(periods=max(4, int(periods)))
            agent.export_combined_excel(periods=max(4, int(periods)))
        except Exception as e:
            print(f"Method export error: {e}")

        return {
            "report": report,
            "evaluation_results": evaluation_results,
            "forecast_results": forecast_results,
            "history": history,
            "method_to_model": agent._clean_for_json(method_to_model) if "method_to_model" in locals() else None,
            "detected": {"time_col": agent.time_col, "demand_col": agent.demand_col},
            "available_methods": sorted(set(
                ["naive", "seasonal_naive", "moving_average", "ets", "arima", "llm"]
                + list((agent.baseline_models or {}).keys())
                + list((agent.advanced_models or {}).keys())
            )),
            "future_by_method": future_by_method,
            "future_errors": agent._clean_for_json(future_errors) if "future_errors" in locals() else None,
            "llm": {
                "enabled": bool(use_llm),
                "provider": llm_provider,
                "model": llm_model,
                "for_forecast": bool(llm_for_forecast),
                "for_report": bool(llm_for_report),
                "error": llm_error,
                "error_code": llm_error_code,
            },
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/agent")
async def agent_predict(
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    freq: str = Form("D"),
    periods: int = Form(10),
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
    time_col: Optional[str] = Form(None),
    demand_col: Optional[str] = Form(None),
    include_advanced: bool = Form(False),
    dataset_name: Optional[str] = Form(None),
    dataset_theme: Optional[str] = Form(None),
    data_description: Optional[str] = Form(None),
    language: str = Form("en"),
    use_llm: bool = Form(True),
    llm_provider: Optional[str] = Form("deepseek"),
    llm_model: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_timeout_seconds: int = Form(60),
    ocr_confirmed: bool = Form(False),
    ocr_table_json: Optional[str] = Form(None),
):
    user_prompt = str(prompt or "").strip()
    file_name = file.filename if file is not None else None
    if file is None and not user_prompt:
        raise HTTPException(status_code=400, detail={"error": "请至少提供文本描述或文件（CSV/Excel/图片）"})

    file_bytes = b""
    suffix = ""
    if file is not None:
        file_bytes = await file.read()
        suffix = os.path.splitext(file_name or "")[1].lower()

    ocr_df = None
    ocr_meta = None
    if ocr_table_json and str(ocr_table_json).strip():
        try:
            parsed = json.loads(str(ocr_table_json))
            if isinstance(parsed, dict):
                rows = parsed.get("rows")
                if isinstance(rows, list):
                    parsed = rows
            if not isinstance(parsed, list):
                raise ValueError("ocr_table_json 必须是数组或包含 rows 数组的对象")
            ocr_df = pd.DataFrame(parsed)
            if ocr_df is None or ocr_df.empty:
                raise ValueError("OCR 表格为空")
            ocr_df = ocr_df.dropna(how="all")
            ocr_meta = {"source": "frontend_confirmed_table"}
        except Exception as e:
            raise HTTPException(status_code=400, detail={"error": "OCR表格解析失败", "message": str(e)})

    # 图片首次上传：先返回OCR表格预览，前端确认后再跑真实预测
    if file is not None and _is_image_suffix(suffix) and ocr_df is None and not _to_bool(ocr_confirmed, False):
        try:
            extracted_df, extracted_meta = _extract_table_from_image(file_bytes, nrows=120)
            llm_meta = {"enabled": bool(_to_bool(use_llm, True)), "error": None, "error_code": None, "warnings": []}
            if _to_bool(use_llm, True):
                try:
                    refined_df, refine_info = _llm_refine_ocr_table(
                        extracted_df,
                        user_prompt=user_prompt,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        llm_timeout_seconds=int(llm_timeout_seconds),
                    )
                    extracted_df = refined_df
                    llm_meta["warnings"] = list(refine_info.get("warnings") or [])
                except Exception as e:
                    llm_meta["error"] = str(e)
                    if "HTTP 402" in llm_meta["error"] or "Insufficient Balance" in llm_meta["error"] or "Payment Required" in llm_meta["error"]:
                        llm_meta["error_code"] = "insufficient_balance"
            detected_time_col, detected_demand_col = _detect_columns(extracted_df, time_col, demand_col)
            return {
                "ok": True,
                "mode": "ocr_preview_needed",
                "input_type": "image",
                "filename": file_name,
                "preview": {
                    "source": "ocr_image",
                    "ocr_meta": extracted_meta,
                    "llm": llm_meta,
                    "columns": [str(c) for c in extracted_df.columns.tolist()],
                    "detected": {"time_col": detected_time_col, "demand_col": detected_demand_col},
                    "rows": _df_to_rows(extracted_df, nrows=120),
                    "row_count": int(len(extracted_df)),
                },
                "message": "已从截图中提取表格，请确认列名后再次运行预测。",
            }
        except Exception:
            # OCR失败时，继续走下方多模态LLM解释分支
            pass

    # 1) 若是CSV/Excel，或用户确认OCR表格，直接做真实预测（可选LLM预测+LLM报告）
    if (file is not None and suffix in {".csv", ".xlsx", ".xls"}) or (ocr_df is not None and _to_bool(ocr_confirmed, False)):
        if ocr_df is not None:
            df_preview = ocr_df.copy()
            csv_text = df_preview.to_csv(index=False)
            file_bytes = csv_text.encode("utf-8")
            suffix = ".csv"
            if file_name:
                stem = os.path.splitext(file_name)[0]
                file_name = f"{stem}_ocr.csv"
            else:
                file_name = "ocr_extracted.csv"
        else:
            df_preview = _read_preview(file_bytes, suffix)
        detected_time_col, detected_demand_col = _detect_columns(df_preview, time_col, demand_col)
        if detected_time_col is None or detected_demand_col is None:
            # 不能识别为标准时序时，退回LLM做数据理解
            preview_rows = df_preview.head(20).to_dict(orient="records")
            llm_obj = _deepseek_chat_json(
                messages=[
                    {"role": "system", "content": "你是数据分析助手。输出严格JSON。"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "identify_forecastable_targets",
                                "prompt": user_prompt,
                                "columns": [str(c) for c in df_preview.columns.tolist()],
                                "preview_rows": preview_rows,
                                "output_schema": {
                                    "summary": "string",
                                    "candidate_targets": [{"name": "string", "reason": "string"}],
                                    "required_time_col_hint": "string",
                                    "clarifying_questions": ["string"],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                timeout_seconds=llm_timeout_seconds,
            )
            return {
                "ok": True,
                "mode": "llm_data_understanding",
                "input_type": "table_nonstandard",
                "filename": file_name,
                "llm_result": llm_obj,
                "ocr_meta": ocr_meta,
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            agent = ForecastProAgent(
                data_path=tmp_path,
                time_col=detected_time_col,
                demand_col=detected_demand_col,
                freq=freq,
                random_seed=int(random_seed),
                language=str(language).strip()
            )
            agent.test_size = float(test_size)
            agent.configure_llm(
                enabled=_to_bool(use_llm, True),
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
                timeout_seconds=int(llm_timeout_seconds),
            )
            if dataset_name and str(dataset_name).strip():
                agent.dataset_name_override = str(dataset_name).strip()
            if dataset_theme and str(dataset_theme).strip():
                agent.dataset_theme = str(dataset_theme).strip()
            if data_description and str(data_description).strip():
                agent.data_description = str(data_description).strip()
            elif user_prompt:
                agent.data_description = user_prompt

            agent.load_data(tmp_path)
            agent.prepare_data()
            agent.run_baseline_models()
            if include_advanced:
                agent.run_advanced_models()
            agent.evaluate_models()
            llm_enabled = _to_bool(use_llm, True)
            llm_error = None
            llm_error_code = None
            if llm_enabled:
                try:
                    agent.generate_forecast(periods=max(4, int(periods)), forecast_method="llm")
                except Exception as e:
                    llm_error = str(e)
                    if "HTTP 402" in llm_error or "Insufficient Balance" in llm_error or "Payment Required" in llm_error:
                        llm_error_code = "insufficient_balance"
                    agent.generate_forecast(periods=max(4, int(periods)), forecast_method="auto")
            else:
                agent.generate_forecast(periods=max(4, int(periods)), forecast_method="auto")

            try:
                report = agent.generate_report(save_to_disk=False, use_llm=bool(llm_enabled and not llm_error))
            except Exception as e:
                report = agent.generate_report(save_to_disk=False, use_llm=False)
                if llm_error:
                    llm_error = f"{llm_error}; report_llm_error={e}"
                else:
                    llm_error = f"report_llm_error={e}"

            y = agent.data[agent.demand_col].astype(float) if agent.data is not None else None
            history = None
            method_to_model = {}
            if y is not None and len(y) > 0:
                tail_n = min(len(y), 120)
                y_tail = y.iloc[-tail_n:]
                fitted_best_tail = None
                fitted_by_method_tail = {}
                test_pred_best_tail = None
                test_pred_by_method_tail = {}

                train_end_date = None
                test_start_date = None
                try:
                    if agent.train_data and agent.train_data.get("y") is not None and len(agent.train_data["y"]) > 0:
                        train_end_date = agent.train_data["y"].index[-1]
                    if agent.test_data and agent.test_data.get("y") is not None and len(agent.test_data["y"]) > 0:
                        test_start_date = agent.test_data["y"].index[0]
                except Exception:
                    train_end_date = None
                    test_start_date = None

                try:
                    best_name = getattr(agent, "best_model", None)
                    fitted = None
                    idx = None
                    if best_name and isinstance(best_name, str):
                        if (agent.baseline_models or {}).get(best_name) is not None:
                            fitted = (agent.baseline_models or {}).get(best_name, {}).get("fitted")
                            idx = agent.train_data["y"].index if agent.train_data and agent.train_data.get("y") is not None else None
                        elif (agent.advanced_models or {}).get(best_name) is not None:
                            fitted = (agent.advanced_models or {}).get(best_name, {}).get("fitted")
                            try:
                                idx = agent.train_features.get("y").index if getattr(agent, "train_features", None) is not None and agent.train_features.get("y") is not None else None
                            except Exception:
                                idx = None
                            if idx is None:
                                idx = agent.train_data["y"].index if agent.train_data and agent.train_data.get("y") is not None else None
                    if fitted is not None and idx is not None:
                        fitted_list = list(fitted)
                        fit_map = {}
                        n = min(len(idx), len(fitted_list))
                        for i in range(n):
                            try:
                                v = float(fitted_list[i])
                                fit_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                            except Exception:
                                fit_map[idx[i]] = None
                        fitted_best_tail = [fit_map.get(d, None) for d in y_tail.index]
                except Exception:
                    fitted_best_tail = None

                try:
                    best_name = getattr(agent, "best_model", None)
                    preds = None
                    idx = None
                    if best_name and isinstance(best_name, str):
                        if (agent.baseline_models or {}).get(best_name) is not None:
                            preds = (agent.baseline_models or {}).get(best_name, {}).get("predictions")
                            idx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                        elif (agent.advanced_models or {}).get(best_name) is not None:
                            preds = (agent.advanced_models or {}).get(best_name, {}).get("predictions")
                            try:
                                idx = agent.test_features.get("y").index if getattr(agent, "test_features", None) is not None and agent.test_features.get("y") is not None else None
                            except Exception:
                                idx = None
                            if idx is None:
                                idx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                    pred_map = {}
                    if preds is not None and idx is not None:
                        preds_list = list(preds)
                        n = min(len(idx), len(preds_list))
                        for i in range(n):
                            try:
                                v = float(preds_list[i])
                                pred_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                            except Exception:
                                pred_map[idx[i]] = None
                    test_pred_best_tail = [pred_map.get(d, None) for d in y_tail.index]
                except Exception:
                    test_pred_best_tail = None

                try:
                    eval_df = agent.evaluation_results
                    best_ets_variant = None
                    if eval_df is not None and len(eval_df) > 0 and "model" in eval_df.columns:
                        ets_candidates = ["ets_simple", "ets_holt", "ets_holt_winters", "seasonal_ets", "ets"]
                        dfv = eval_df[eval_df["model"].isin(ets_candidates)]
                        if len(dfv) > 0 and "RMSE" in dfv.columns:
                            best_ets_variant = str(dfv.sort_values("RMSE").iloc[0]["model"])

                    def _model_key_for_method(m: str) -> str:
                        mm = str(m or "").strip().lower()
                        if mm == "ets":
                            return best_ets_variant or "ets"
                        return mm

                    def _fitted_for_model(model_key: str):
                        if not model_key:
                            return None
                        if (agent.baseline_models or {}).get(model_key) is not None:
                            return (agent.baseline_models or {}).get(model_key, {}).get("fitted")
                        if (agent.advanced_models or {}).get(model_key) is not None:
                            return (agent.advanced_models or {}).get(model_key, {}).get("fitted")
                        return None

                    def _pred_for_model(model_key: str):
                        if not model_key:
                            return None
                        if (agent.baseline_models or {}).get(model_key) is not None:
                            return (agent.baseline_models or {}).get(model_key, {}).get("predictions")
                        if (agent.advanced_models or {}).get(model_key) is not None:
                            return (agent.advanced_models or {}).get(model_key, {}).get("predictions")
                        return None

                    method_list = ["naive", "seasonal_naive", "moving_average", "ets", "arima"]
                    for m in method_list:
                        mk = _model_key_for_method(m)
                        method_to_model[str(m)] = mk

                        fitted_m = _fitted_for_model(mk)
                        if fitted_m is not None:
                            fitted_list = list(fitted_m)
                            idx = None
                            try:
                                if (agent.baseline_models or {}).get(mk) is not None:
                                    idx = agent.train_data["y"].index
                                elif (agent.advanced_models or {}).get(mk) is not None and getattr(agent, "train_features", None) is not None:
                                    idx = agent.train_features.get("y").index if agent.train_features.get("y") is not None else None
                            except Exception:
                                idx = None
                            fit_map = {}
                            if idx is not None:
                                n = min(len(idx), len(fitted_list))
                                for i in range(n):
                                    try:
                                        v = float(fitted_list[i])
                                        fit_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                                    except Exception:
                                        fit_map[idx[i]] = None
                            fitted_by_method_tail[str(m)] = [fit_map.get(d, None) for d in y_tail.index]

                        pred_m = _pred_for_model(mk)
                        if pred_m is not None:
                            preds_list = list(pred_m)
                            pidx = None
                            try:
                                if (agent.baseline_models or {}).get(mk) is not None:
                                    pidx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                                elif (agent.advanced_models or {}).get(mk) is not None and getattr(agent, "test_features", None) is not None:
                                    pidx = agent.test_features.get("y").index if agent.test_features.get("y") is not None else None
                            except Exception:
                                pidx = None
                            if pidx is None:
                                pidx = agent.test_data["y"].index if agent.test_data and agent.test_data.get("y") is not None else None
                            pred_map = {}
                            if pidx is not None:
                                n = min(len(pidx), len(preds_list))
                                for i in range(n):
                                    try:
                                        v = float(preds_list[i])
                                        pred_map[pidx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                                    except Exception:
                                        pred_map[pidx[i]] = None
                            test_pred_by_method_tail[str(m)] = [pred_map.get(d, None) for d in y_tail.index]
                except Exception:
                    fitted_by_method_tail = {}
                    test_pred_by_method_tail = {}
                    method_to_model = {}

                history = {
                    "dates": [d.to_pydatetime() for d in y_tail.index],
                    "actual": [float(v) for v in y_tail.values],
                    "train_size": int(len(agent.train_data["y"])) if agent.train_data and "y" in agent.train_data else None,
                    "total_size": int(len(y)),
                    "fitted_best": fitted_best_tail,
                    "fitted_by_method": fitted_by_method_tail,
                    "test_pred_best": test_pred_best_tail,
                    "test_pred_by_method": test_pred_by_method_tail,
                    "train_end_date": pd.to_datetime(train_end_date).to_pydatetime() if train_end_date is not None else None,
                    "test_start_date": pd.to_datetime(test_start_date).to_pydatetime() if test_start_date is not None else None,
                }

            future_by_method = dict(getattr(agent, "future_by_method", {}) or {})
            future_errors = {}
            for m in ["naive", "seasonal_naive", "moving_average", "ets", "arima"]:
                if m in future_by_method:
                    continue
                try:
                    agent.generate_forecast(periods=max(4, int(periods)), forecast_method=m)
                    if agent.forecast_results:
                        future_by_method[m] = agent._clean_for_json(agent.forecast_results)
                except Exception as e:
                    future_errors[m] = str(e)
            
            try:
                agent.export_method_folders(periods=max(4, int(periods)))
                agent.export_combined_excel(periods=max(4, int(periods)))
            except Exception as e:
                print(f"Method export error in agent route: {e}")

            return {
                "ok": True,
                "mode": "timeseries_ocr" if ocr_df is not None else "timeseries_csv",
                "filename": file_name,
                "source": "ocr_table" if ocr_df is not None else "file_table",
                "ocr_meta": ocr_meta,
                "report": report,
                "evaluation_results": agent._clean_for_json(agent.evaluation_results.to_dict("records")) if agent.evaluation_results is not None else None,
                "forecast_results": agent._clean_for_json(agent.forecast_results),
                "history": agent._clean_for_json(history) if history is not None else None,
                "method_to_model": agent._clean_for_json(method_to_model) if method_to_model else None,
                "detected": {"time_col": agent.time_col, "demand_col": agent.demand_col},
                "available_methods": sorted(
                    set(
                        ["naive", "seasonal_naive", "moving_average", "ets", "arima", "llm"]
                        + list((agent.baseline_models or {}).keys())
                        + list((agent.advanced_models or {}).keys())
                    )
                ),
                "future_by_method": future_by_method,
                "future_errors": agent._clean_for_json(future_errors) if future_errors else None,
                "llm": {
                    "enabled": bool(llm_enabled),
                    "provider": "deepseek",
                    "model": llm_model,
                    "error": llm_error,
                    "error_code": llm_error_code,
                },
            }
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # 2) 文本 / 图片：走多模态LLM做可预测目标识别与预测策略
    input_type = "text"
    user_content = [
        {"type": "text", "text": json.dumps(
            {
                "task": "multimodal_forecast_advisor",
                "user_prompt": user_prompt,
                "freq": freq,
                "periods": int(periods),
                "dataset_theme": dataset_theme,
                "data_description": data_description,
                "output_schema": {
                    "summary": "string",
                    "forecastable_targets": [{"name": "string", "why": "string", "required_columns": ["string"]}],
                    "recommended_target": "string",
                    "forecast_plan": ["string"],
                    "report_outline": ["string"],
                    "clarifying_questions": ["string"],
                },
                "constraints": [
                    "不要编造真实指标数值",
                    "如果信息不足，明确说明并提出澄清问题",
                    "语言使用中文，适合业务用户",
                ],
            },
            ensure_ascii=False,
        )},
    ]

    if file is not None and suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        input_type = "image"
        mime = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".gif":
            mime = "image/gif"
        data_url = f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    llm_obj = _deepseek_chat_json(
        messages=[
            {"role": "system", "content": "你是多模态预测顾问。输出严格JSON。"},
            {"role": "user", "content": user_content},
        ],
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        timeout_seconds=llm_timeout_seconds,
    )

    return {
        "ok": True,
        "mode": "llm_multimodal",
        "input_type": input_type,
        "filename": file_name,
        "llm_result": llm_obj,
    }
