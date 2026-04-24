"""
ForecastPro Agent核心模块
实现AI驱动的需求预测任务，包含数据摄取、基准建模、高级模型探索、评估诊断和管理报告生成。
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import urllib.request
import urllib.error

# 数据科学库
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 时间序列库
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 深度学习库（可选）
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, NHITS, TFT
    HAS_NEURALFORECAST = True
except ImportError:
    HAS_NEURALFORECAST = False

try:
    from darts import TimeSeries
    from darts.models import (
        ExponentialSmoothing as DartsExponentialSmoothing,
        ARIMA as DartsARIMA,
        RandomForest as DartsRandomForest,
        XGBModel,
        RNNModel,
        TFTModel,
        NHiTSModel
    )
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


class ForecastProAgent:
    """
    ForecastPro AI需求预测Agent

    专门负责为企业设计和执行AI驱动的需求预测任务，核心目标是减少因预测不准
    导致的缺货、库存积压或产能闲置，从而提高运营效率。
    """

    def __init__(self,
                 data_path: Optional[str] = None,
                 time_col: str = 'date',
                 demand_col: str = 'demand',
                 freq: str = 'D',
                 random_seed: int = 42,
                 language: str = 'en'):
        """
        初始化ForecastPro Agent

        参数:
            data_path: 数据文件路径 (CSV/Excel)
            time_col: 时间列名称
            demand_col: 需求列名称
            freq: 时间频率 ('D'=日, 'W'=周, 'M'=月, 'Q'=季, 'Y'=年)
            random_seed: 随机种子
        """
        self.data_path = data_path
        self.time_col = time_col
        self.demand_col = demand_col
        self.freq = freq
        self.random_seed = random_seed
        self.language = str(language or "en").strip().lower()
        np.random.seed(random_seed)

        # 数据属性
        self.data = None
        self.X = None
        self.y = None
        self.features = None
        self.covariates = None
        self.train_data = None
        self.test_data = None

        # 模型存储
        self.baseline_models = {}
        self.advanced_models = {}
        self.model_results = {}
        self.best_model = None

        # 报告存储
        self.report = {}

        # 配置
        self.test_size = 0.2  # 20%测试集

        # 变量术语映射 - 根据实际变量名适配报告术语
        self.variable_labels = {
            # 需求相关
            'demand': '需求',
            'quantity': '数量',
            'volume': '成交量',
            'units_sold': '销售数量',
            # 销售/收入相关
            'sales': '销售额',
            'revenue': '收入',
            'sales_revenue': '销售收入',
            'value': '价值',
            'Value': '价值',
            # 消耗相关
            'consumption': '消耗量',
            'Consumption': '消耗量',
            # 成本相关
            'cost': '成本',
            'Cost': '成本',
            # 价格相关
            'price': '价格',
            'Price': '价格',
            # 通用目标变量
            'y': '目标变量',
            'target': '目标变量'
        }

        print("=" * 60)
        print("ForecastPro AI需求预测Agent初始化完成")
        print("=" * 60)

    def configure_llm(
        self,
        enabled: bool = False,
        provider: str = "deepseek",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
    ):
        """配置LLM参数（默认按DeepSeek兼容OpenAI Chat接口）"""
        # Determine defaults based on provider
        prov = str(provider or "deepseek").lower().strip()
        default_model = "deepseek-chat"
        default_base = "https://api.deepseek.com"
        
        if prov == "doubao":
            default_base = "https://ark.cn-beijing.volces.com/api/v3"
            default_model = "" # Usually ep-xxxx
            
        self.llm_config = {
            "enabled": bool(enabled),
            "provider": prov,
            "model": str(model or os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", default_model)).strip(),
            "api_key": api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            "base_url": (base_url or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", default_base)).rstrip("/"),
            "timeout_seconds": int(timeout_seconds or 60),
        }
        return self.llm_config

    def _llm_enabled(self) -> bool:
        cfg = getattr(self, "llm_config", None) or {}
        return bool(cfg.get("enabled"))

    def _extract_json_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return raw
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw

    def _deepseek_chat(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, str]] = None) -> str:
        cfg = getattr(self, "llm_config", None) or {}
        if not bool(cfg.get("enabled")):
            raise ValueError("LLM未启用，请先调用 configure_llm(enabled=True, ...)")
        # Remove strict deepseek provider check to support doubao and custom OpenAI endpoints
        # if str(cfg.get("provider", "")).lower() != "deepseek":
        #     raise ValueError(f"暂不支持的LLM提供商: {cfg.get('provider')}")
        if not cfg.get("api_key"):
            raise ValueError("未配置 LLM API KEY")

        endpoint = f"{cfg.get('base_url')}/chat/completions"
        payload = {
            "model": cfg.get("model", "deepseek-chat"),
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format:
            payload["response_format"] = response_format

        api_key = str(cfg.get('api_key') or "").strip().encode("latin-1", "ignore").decode("latin-1")
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=int(cfg.get("timeout_seconds", 60))) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise ValueError(f"DeepSeek API请求失败: HTTP {getattr(e, 'code', 'ERR')} - {detail}") from e
        except Exception as e:
            raise ValueError(f"DeepSeek API请求失败: {e}") from e

        try:
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"]
            return str(content)
        except Exception as e:
            raise ValueError(f"DeepSeek API响应解析失败: {e}; body={body[:500]}") from e

    def generate_forecast_with_llm(self, periods: int = 4):
        """使用LLM进行生成式未来预测，并返回统一forecast_results结构"""
        if self.data is None:
            raise ValueError("请先加载数据")
        if periods < 1:
            raise ValueError("periods 必须 >= 1")
        if not self._llm_enabled():
            raise ValueError("未启用LLM，请先配置configure_llm")

        y = self.data[self.demand_col].astype(float).dropna()
        if len(y) < 8:
            raise ValueError("LLM预测至少需要8个有效观测值")

        last_date = self.data.index[-1]
        try:
            idx = pd.date_range(start=last_date, periods=periods + 1, freq=self.freq)[1:]
        except Exception:
            idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

        history_tail = []
        for d, v in zip(y.index[-min(120, len(y)):], y.values[-min(120, len(y)):]):
            history_tail.append({"date": pd.to_datetime(d).isoformat(), "value": float(v)})

        ts_hint = getattr(self, "ts_analysis_results", None) or {}
        eval_hint = None
        try:
            eval_hint = self.evaluation_results.to_dict("records") if self.evaluation_results is not None else None
        except Exception:
            eval_hint = None

        system_prompt = (
            "你是资深时间序列预测专家。任务：根据历史序列生成未来预测。"
            "必须输出严格JSON，不要输出任何解释文本。"
        )
        user_payload = {
            "task": "forecast",
            "periods": int(periods),
            "freq": str(self.freq),
            "target": str(self.demand_col),
            "history_tail": history_tail,
            "hints": {
                "ts_analysis": ts_hint,
                "evaluation_results": eval_hint,
            },
            "required_output_schema": {
                "forecast": ["float"] * int(periods),
                "lower_bound": ["float"] * int(periods),
                "upper_bound": ["float"] * int(periods),
                "assumptions": ["string"],
            },
            "constraints": [
                "数组长度必须都等于periods",
                "每个位置必须满足 lower_bound[i] <= forecast[i] <= upper_bound[i]",
                "数值必须是可解析浮点数",
            ],
        }
        user_payload = self._clean_for_json(user_payload)
        content = self._deepseek_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(self._extract_json_text(content))
        fc = [float(x) for x in (parsed.get("forecast") or [])]
        lo = [float(x) for x in (parsed.get("lower_bound") or [])]
        up = [float(x) for x in (parsed.get("upper_bound") or [])]

        if not (len(fc) == len(lo) == len(up) == int(periods)):
            raise ValueError("LLM预测返回长度不匹配periods")
        for i in range(int(periods)):
            a, b, c = lo[i], fc[i], up[i]
            if not (a <= b <= c):
                lo[i], up[i] = min(a, b, c), max(a, b, c)
                fc[i] = min(max(b, lo[i]), up[i])

        self.forecast_results = {
            "dates": [d.to_pydatetime() for d in idx],
            "forecast": fc,
            "lower_bound": lo,
            "upper_bound": up,
            "model": f"llm_{(self.llm_config or {}).get('provider', 'deepseek')}",
            "params": {
                "llm_model": (self.llm_config or {}).get("model"),
                "assumptions": parsed.get("assumptions") or [],
            },
        }
        return self.forecast_results

    def _recommendation_profile(self) -> Dict[str, Any]:
        theme = ""
        desc = ""
        try:
            theme = str(getattr(self, "dataset_theme", "") or "").strip().lower()
        except Exception:
            theme = ""
        try:
            desc = str(getattr(self, "data_description", "") or "").strip().lower()
        except Exception:
            desc = ""

        allow_inventory = False
        if any(k in theme for k in ["inventory", "stock", "retail", "ecommerce", "warehouse", "supply", "replenish", "procurement"]):
            allow_inventory = True
        if any(k in desc for k in ["库存", "补货", "缺货", "在库", "仓库", "订货", "再订购", "安全库存"]):
            allow_inventory = True

        allow_staffing = False
        if any(k in theme for k in ["staff", "staffing", "workforce", "callcenter", "call_center", "labor", "shift"]):
            allow_staffing = True
        if any(k in desc for k in ["人员", "排班", "班次", "工时", "客服", "人力"]):
            allow_staffing = True

        allow_capacity = False
        if any(k in theme for k in ["manufacturing", "production", "capacity", "factory", "plant", "operations"]):
            allow_capacity = True
        if any(k in desc for k in ["产能", "生产", "工厂", "设备", "产线"]):
            allow_capacity = True

        if allow_inventory:
            kind = "inventory"
        elif allow_staffing:
            kind = "staffing"
        elif allow_capacity:
            kind = "capacity"
        else:
            kind = "generic"

        forbidden_terms = []
        if not allow_inventory:
            forbidden_terms += ["库存", "安全库存", "补货", "缺货", "在库", "订货", "再订购", "仓库"]
        if not allow_staffing:
            forbidden_terms += ["排班", "班次", "人力", "人员配置", "加班", "外包"]

        return {
            "kind": kind,
            "theme": theme or None,
            "description": desc or None,
            "allow_inventory": bool(allow_inventory),
            "allow_staffing": bool(allow_staffing),
            "allow_capacity": bool(allow_capacity),
            "forbidden_terms": forbidden_terms,
        }

    def _actionable_recommendations_from_forecast(self, interp: Dict[str, Any]) -> List[Dict[str, str]]:
        is_en = str(getattr(self, "language", "en")).lower().startswith("en")
        def _t(zh_str: str, en_str: str) -> str:
            return en_str if is_en else zh_str
        profile = self._recommendation_profile()
        kind = str(profile.get("kind") or "generic")

        peak = (interp or {}).get("peak") or {}
        trough = (interp or {}).get("trough") or {}
        peak_date = pd.to_datetime(peak.get("date")) if peak.get("date") else None
        trough_date = pd.to_datetime(trough.get("date")) if trough.get("date") else None
        peak_val = peak.get("forecast")
        trough_val = trough.get("forecast")
        peak_up = peak.get("upper_95") if peak.get("upper_95") is not None else peak_val
        trough_lo = trough.get("lower_95") if trough.get("lower_95") is not None else trough_val

        def _f(v):
            try:
                return float(v)
            except Exception:
                return None

        peak_val = _f(peak_val)
        trough_val = _f(trough_val)
        peak_up = _f(peak_up)
        trough_lo = _f(trough_lo)

        if peak_val is None or trough_val is None:
            return []

        label = self.get_variable_label()
        peak_date_s = peak_date.strftime("%Y-%m-%d") if peak_date is not None else "—"
        trough_date_s = trough_date.strftime("%Y-%m-%d") if trough_date is not None else "—"
        buffer_amt = None
        if peak_up is not None and peak_val is not None:
            buffer_amt = max(0.0, float(peak_up) - float(peak_val))

        if kind == "inventory":
            return [
                {
                    "title": _t("库存/补货目标", "Inventory/Replenishment Target"),
                    "finding": _t(
                        f"未来预测峰值出现在 {peak_date_s}，点预测约为 {peak_val:.3f}；95%上限约为 {(peak_up if peak_up is not None else peak_val):.3f}。",
                        f"Forecast peak occurs at {peak_date_s}; point forecast is approx {peak_val:.3f}; 95% upper bound is approx {(peak_up if peak_up is not None else peak_val):.3f}."
                    ),
                    "action": _t(
                        f"以保守策略将该峰值对应周期的备货目标设置在 95%上限水平（约 {(peak_up if peak_up is not None else peak_val):.3f}），并将安全缓冲设置为 上限-点预测（约 {(buffer_amt if buffer_amt is not None else 0.0):.3f}）。",
                        f"For conservative planning, set the inventory target for that peak period to the 95% upper bound (approx {(peak_up if peak_up is not None else peak_val):.3f}) and keep a safety buffer of Upper - Point (approx {(buffer_amt if buffer_amt is not None else 0.0):.3f})."
                    ),
                    "reason": _t("预测区间反映不确定性；用上限做规划可降低缺货风险。", "Prediction intervals reflect uncertainty; planning with the upper bound reduces stockout risk."),
                },
                {
                    "title": _t("采购节奏/再订购时机", "Procurement/Reorder Timing"),
                    "finding": _t(
                        f"未来低谷期预计在 {trough_date_s}，点预测约为 {trough_val:.3f}；95%下限约为 {(trough_lo if trough_lo is not None else trough_val):.3f}。",
                        f"Forecast trough occurs at {trough_date_s}; point forecast is approx {trough_val:.3f}; 95% lower bound is approx {(trough_lo if trough_lo is not None else trough_val):.3f}."
                    ),
                    "action": _t(
                        f"在低谷周期将采购/订货节奏下调到 点预测水平（约 {trough_val:.3f}），并保留最低保障量在 95%下限附近（约 {(trough_lo if trough_lo is not None else trough_val):.3f}）。",
                        f"In the trough period, reduce procurement/reorder volume toward the point forecast (approx {trough_val:.3f}) and keep a minimum protection level near the 95% lower bound (approx {(trough_lo if trough_lo is not None else trough_val):.3f})."
                    ),
                    "reason": _t("低谷期下调可减少积压成本，同时用下限作为底线可应对需求反弹。", "Reducing in trough periods lowers overstock costs while the lower bound provides a safety floor."),
                },
            ]

        if kind == "staffing":
            return [
                {
                    "title": _t("人员配置上限", "Staffing Upper Bound"),
                    "finding": _t(
                        f"未来高峰预计在 {peak_date_s}，点预测约为 {peak_val:.3f}；95%上限约为 {(peak_up if peak_up is not None else peak_val):.3f}。",
                        f"Peak expected at {peak_date_s}; point forecast is approx {peak_val:.3f}; 95% upper bound is approx {(peak_up if peak_up is not None else peak_val):.3f}."
                    ),
                    "action": _t(
                        f"将高峰周期的人员/班次配置上限按 95%上限规划（约 {(peak_up if peak_up is not None else peak_val):.3f}），并准备弹性人力覆盖 上限-点预测（约 {(buffer_amt if buffer_amt is not None else 0.0):.3f}）。",
                        f"Plan staffing/shift capacity using the 95% upper bound (approx {(peak_up if peak_up is not None else peak_val):.3f}) and keep flexible coverage for Upper - Point (approx {(buffer_amt if buffer_amt is not None else 0.0):.3f})."
                    ),
                    "reason": _t("用上限做排班可以降低服务水平下降或积压的风险。", "Planning with the upper bound reduces service-level risk and backlog risk."),
                },
                {
                    "title": _t("低谷期排班收缩", "Scale Down in Trough"),
                    "finding": _t(
                        f"未来低谷预计在 {trough_date_s}，点预测约为 {trough_val:.3f}；95%下限约为 {(trough_lo if trough_lo is not None else trough_val):.3f}。",
                        f"Trough expected at {trough_date_s}; point forecast is approx {trough_val:.3f}; 95% lower bound is approx {(trough_lo if trough_lo is not None else trough_val):.3f}."
                    ),
                    "action": _t(
                        f"在低谷周期将排班/外包使用量收缩到点预测水平（约 {trough_val:.3f}），并保留最低保障在 95%下限附近（约 {(trough_lo if trough_lo is not None else trough_val):.3f}）。",
                        f"In the trough period, reduce staffing/outsourcing toward the point forecast (approx {trough_val:.3f}) and keep a minimum protection level near the 95% lower bound (approx {(trough_lo if trough_lo is not None else trough_val):.3f})."
                    ),
                    "reason": _t("低谷期收缩可控制成本，同时保留底线避免意外波动导致的服务中断。", "Scaling down controls cost while the minimum level protects against unexpected swings."),
                },
            ]

        if kind == "capacity":
            return [
                {
                    "title": _t("产能上限规划", "Capacity Upper Bound Planning"),
                    "finding": _t(
                        f"未来需求峰值预计在 {peak_date_s}，点预测约为 {peak_val:.3f}；95%上限约为 {(peak_up if peak_up is not None else peak_val):.3f}。",
                        f"Peak expected at {peak_date_s}; point forecast is approx {peak_val:.3f}; 95% upper bound is approx {(peak_up if peak_up is not None else peak_val):.3f}."
                    ),
                    "action": _t(
                        f"将该周期的产能/供应上限按 95%上限规划（约 {(peak_up if peak_up is not None else peak_val):.3f}），并预留弹性覆盖 上限-点预测（约 {(buffer_amt if buffer_amt is not None else 0.0):.3f}）。",
                        f"Plan capacity/supply upper bound using the 95% upper bound (approx {(peak_up if peak_up is not None else peak_val):.3f}) and reserve flexibility for Upper - Point (approx {(buffer_amt if buffer_amt is not None else 0.0):.3f})."
                    ),
                    "reason": _t("预测区间反映不确定性；用上限规划可以降低延期交付或产能瓶颈风险。", "Prediction intervals reflect uncertainty; planning with the upper bound reduces delivery delay and bottleneck risk."),
                },
                {
                    "title": _t("低谷期资源压降", "Scale Down Resources in Trough"),
                    "finding": _t(
                        f"未来低谷期预计在 {trough_date_s}，点预测约为 {trough_val:.3f}；95%下限约为 {(trough_lo if trough_lo is not None else trough_val):.3f}。",
                        f"Trough expected at {trough_date_s}; point forecast is approx {trough_val:.3f}; 95% lower bound is approx {(trough_lo if trough_lo is not None else trough_val):.3f}."
                    ),
                    "action": _t(
                        f"在低谷周期将可变资源投入压降到点预测水平（约 {trough_val:.3f}），并设置最低保障在 95%下限附近（约 {(trough_lo if trough_lo is not None else trough_val):.3f}）。",
                        f"In the trough period, reduce variable resource input toward the point forecast (approx {trough_val:.3f}) and set a minimum protection level near the 95% lower bound (approx {(trough_lo if trough_lo is not None else trough_val):.3f})."
                    ),
                    "reason": _t("低谷期压降可降低闲置成本，同时保留底线应对波动。", "Scaling down reduces idle cost while the minimum level protects against volatility."),
                },
            ]

        return [
            {
                "title": _t("资源上限规划", "Upper Bound Planning"),
                "finding": _t(
                    f"未来峰值预计在 {peak_date_s}，点预测约为 {peak_val:.3f}；95%上限约为 {(peak_up if peak_up is not None else peak_val):.3f}。",
                    f"Peak expected at {peak_date_s}; point forecast is approx {peak_val:.3f}; 95% upper bound is approx {(peak_up if peak_up is not None else peak_val):.3f}."
                ),
                "action": _t(
                    f"将高峰周期的资源/预算上限按 95%上限规划（约 {(peak_up if peak_up is not None else peak_val):.3f}），并预留弹性覆盖 上限-点预测（约 {(buffer_amt if buffer_amt is not None else 0.0):.3f}）。",
                    f"Plan resource/budget upper bounds using the 95% upper bound (approx {(peak_up if peak_up is not None else peak_val):.3f}) and reserve flexibility for Upper - Point (approx {(buffer_amt if buffer_amt is not None else 0.0):.3f})."
                ),
                "reason": _t("预测区间反映不确定性；用上限做规划更稳健。", "Prediction intervals reflect uncertainty; planning with the upper bound is more robust."),
            },
            {
                "title": _t("低谷期节奏调整", "Adjust Pace in Trough"),
                "finding": _t(
                    f"未来低谷预计在 {trough_date_s}，点预测约为 {trough_val:.3f}；95%下限约为 {(trough_lo if trough_lo is not None else trough_val):.3f}。",
                    f"Trough expected at {trough_date_s}; point forecast is approx {trough_val:.3f}; 95% lower bound is approx {(trough_lo if trough_lo is not None else trough_val):.3f}."
                ),
                "action": _t(
                    f"在低谷周期将可变投入调整到点预测水平（约 {trough_val:.3f}），并保留最低保障在 95%下限附近（约 {(trough_lo if trough_lo is not None else trough_val):.3f}）。",
                    f"In the trough period, adjust variable input toward the point forecast (approx {trough_val:.3f}) and keep a minimum protection level near the 95% lower bound (approx {(trough_lo if trough_lo is not None else trough_val):.3f})."
                ),
                "reason": _t("低谷期调整可以控制成本，同时避免过度收缩带来的风险。", "Adjusting in trough periods controls cost while avoiding over-contraction risk."),
            },
        ]

    def _llm_generate_report_sections(self, base_report: Dict[str, Any]) -> Dict[str, Any]:
        """基于现有结构化结果，调用LLM生成可读性更高的报告文本与建议。"""
        if not self._llm_enabled():
            return {}
        is_en = str(getattr(self, "language", "en")).lower().startswith("en")
        profile = self._recommendation_profile()
        if is_en:
            payload = {
                "task": "management_report_en",
                "language": "en-US",
                "input_report": base_report,
                "recommendation_profile": profile,
                "required_output_schema": {
                    "executive_summary_text": "string (Plain-language summary for management: include business impact, expected peak demand, low-demand period, trend direction and seasonality)",
                    "section5_plain_text": "string (Detailed business impact and forecast interpretation)",
                    "actionable_recommendations": [
                        {
                            "title": "string",
                            "finding": "string",
                            "action": "string (Quantified, directly derived from the forecast: e.g., inventory targets, staffing levels, reorder timing, procurement scheduling)",
                            "reason": "string",
                        }
                    ],
                    "model_comparison_commentary": {"model_name": "reason_text"},
                },
                "constraints": [
                    "Do NOT fabricate any metrics; only cite numbers from input_report.",
                    "Write for management; avoid jargon; be clear and sufficiently detailed.",
                    "actionable_recommendations must output exactly 2 items and be as quantified as possible.",
                    "executive_summary_text must be structured and include: (1) business impact (2) expected peak demand (3) low-demand period (4) trend direction and seasonality.",
                    "If recommendation_profile.allow_inventory is false, do not mention inventory/replenishment/stockouts/safety stock/reorder/warehouse.",
                    "If recommendation_profile.allow_staffing is false, do not mention staffing/shifts/labor/overtime/outsourcing.",
                    "Recommendations must come from the forecast and data only; avoid assumptions not supported by input_report.",
                ],
            }
        else:
            payload = {
                "task": "management_report_cn",
                "language": "zh-CN",
                "input_report": base_report,
                "recommendation_profile": profile,
                "required_output_schema": {
                    "executive_summary_text": "string (详细的通俗语言摘要：总结预测对业务的核心影响——必须包含预期峰值需求、低需求期、趋势方向及季节性模式)",
                    "section5_plain_text": "string (详细的业务影响与预测解读)",
                    "actionable_recommendations": [
                        {"title": "string", "finding": "string", "action": "string (直接源自预测的量化可执行建议，例如明确的库存目标、人员配置水平、再订购时机或采购计划安排)", "reason": "string"}
                    ],
                    "model_comparison_commentary": {"model_name": "reason_text"},
                },
                "constraints": [
                    "禁止编造指标数值，只能引用input_report中的真实数字",
                    "面向管理层，非专业术语，通俗易懂且字数充实",
                    "actionable_recommendations必须输出2条，直接源自预测的可执行建议（例如库存目标、人员配置水平、再订购时机等），且必须尽量量化",
                    "executive_summary_text 必须结构化包含：1.预测对业务的影响 2.预期峰值需求 3.低需求期 4.趋势方向及季节性模式",
                    "如果recommendation_profile.allow_inventory为false，禁止出现库存/补货/缺货/安全库存/再订购等表述",
                    "如果recommendation_profile.allow_staffing为false，禁止出现排班/班次/人力/人员配置/加班/外包等表述",
                    "建议必须来自数据与预测本身，不要引入表格未体现的业务假设；若行业/场景不明确，输出更中性的运营建议",
                ],
            }
        payload = self._clean_for_json(payload)
        text = self._deepseek_chat(
            messages=[
                {"role": "system", "content": "You are a management report writing assistant. Output strict JSON only." if is_en else "你是企业管理报告写作助手。输出严格JSON。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(self._extract_json_text(text))
        recs = obj.get("actionable_recommendations") or []
        if isinstance(recs, list):
            recs = recs[:2]
        else:
            recs = []
        return {
            "executive_summary_text": obj.get("executive_summary_text"),
            "section5_plain_text": obj.get("section5_plain_text"),
            "actionable_recommendations": recs,
            "model_comparison_commentary": obj.get("model_comparison_commentary") or {},
        }

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Step 1: 数据摄取与画像

        数据读取：接受CSV、Excel等格式的历史数据
        特征识别：自动识别时间索引、需求变量以及协变量
        质量检查：检查并报告缺失值、异常值和不规则时间步长

        返回:
            处理后的DataFrame
        """
        if data_path is None:
            data_path = self.data_path

        if data_path is None:
            raise ValueError("请提供数据文件路径")

        print(f"正在加载数据: {data_path}")

        # 根据文件扩展名选择读取方法
        file_ext = Path(data_path).suffix.lower()

        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        print(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")

        # 数据预处理：删除不必要的列
        # 删除完全为空的列
        df = df.dropna(axis=1, how='all')

        # 删除可能是索引列的未命名列（如'Unnamed: 0'）
        unnamed_cols = [col for col in df.columns if 'unnamed' in str(col).lower()]
        if unnamed_cols:
            print(f"删除未命名列: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)

        # 自动识别时间列
        time_candidates = ['date', 'time', 'timestamp', 'datetime', 'Date', 'Time', 'TxnDate', 'TxnTime',
                          'Date', 'TIME', 'TIMESTAMP', 'Datetime', 'transaction_date', 'TransactionDate']
        if self.time_col not in df.columns:
            for col in time_candidates:
                if col in df.columns:
                    self.time_col = col
                    print(f"自动识别时间列: {self.time_col}")
                    break

        # 自动识别需求列
        demand_candidates = ['demand', 'sales', 'quantity', 'volume', 'y', 'target', 'consumption', 'Consumption', 'value', 'Value',
                            'sales_revenue', 'units_sold', 'revenue', 'Revenue', 'consumption', 'Consumption',
                            'cost', 'Cost', 'price', 'Price']
        if self.demand_col not in df.columns:
            for col in demand_candidates:
                if col in df.columns:
                    self.demand_col = col
                    print(f"自动识别需求列: {self.demand_col}")
                    break

        # 转换为时间序列
        if self.time_col in df.columns:
            # 检查是否有分离的日期和时间列
            if self.time_col == 'TxnDate' and 'TxnTime' in df.columns:
                # 合并日期和时间
                print("检测到分离的日期和时间列，合并为datetime...")
                df['datetime'] = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'])
                df = df.set_index('datetime')
                # 删除原始的日期和时间列，避免它们成为协变量
                if 'TxnDate' in df.columns:
                    df = df.drop(columns=['TxnDate'])
                if 'TxnTime' in df.columns:
                    df = df.drop(columns=['TxnTime'])
                self.time_col = 'datetime'  # 更新时间列名称
            else:
                # 标准时间列处理
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                df = df.set_index(self.time_col)

            # 检查是否有重复的时间索引
            if df.index.duplicated().any():
                print(f"警告: 时间索引中有 {df.index.duplicated().sum()} 个重复值")
                print("正在删除重复的时间索引（保留第一个）...")
                df = df[~df.index.duplicated(keep='first')]

            # 仅当数据是规则时间序列时才使用asfreq
            # 对于高频率或不规则数据，跳过asfreq以避免创建大量NaN
            try:
                if len(df) > 10:  # 只有足够数据时才尝试
                    # 检查时间间隔是否大致规则
                    time_diffs = df.index.to_series().diff().dropna()
                    if time_diffs.nunique() <= 3:  # 时间间隔相对规则
                        df = df.asfreq(self.freq)
                        print(f"应用频率: {self.freq}")
                    else:
                        print(f"警告: 时间序列不规则，跳过asfreq()")
                else:
                    print("数据量较少，跳过asfreq()")
            except Exception as e:
                print(f"asfreq()失败: {e}，跳过频率调整")

        self.preprocessing_summary = {"missing": {}, "outliers": {}}

        try:
            numeric_cols = []
            for c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except Exception:
                    pass
                if pd.api.types.is_numeric_dtype(df[c]):
                    numeric_cols.append(c)

            for c in numeric_cols:
                miss_rate = float(df[c].isna().mean())
                if miss_rate <= 0:
                    continue

                strategy = "locf"
                if miss_rate < 0.05:
                    strategy = "linear_interpolation"
                    try:
                        if isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing:
                            df[c] = df[c].interpolate(method="time")
                        else:
                            df[c] = df[c].interpolate(method="linear")
                    except Exception:
                        df[c] = df[c].interpolate(method="linear")

                if strategy == "locf":
                    df[c] = df[c].ffill()
                else:
                    df[c] = df[c].ffill()

                if df[c].isna().any():
                    df[c] = df[c].bfill()

                self.preprocessing_summary["missing"][str(c)] = {"missing_rate": miss_rate, "strategy": strategy}

            if self.demand_col in df.columns and pd.api.types.is_numeric_dtype(df[self.demand_col]):
                yv = df[self.demand_col].astype(float)
                q1 = float(yv.quantile(0.25))
                q3 = float(yv.quantile(0.75))
                iqr = float(q3 - q1)
                lower = float(q1 - 1.5 * iqr)
                upper = float(q3 + 1.5 * iqr)
                outlier_mask = (yv < lower) | (yv > upper)
                outlier_mask = outlier_mask.fillna(False)
                self.outlier_mask = outlier_mask
                self.preprocessing_summary["outliers"] = {
                    "method": "iqr",
                    "lower": lower,
                    "upper": upper,
                    "count": int(outlier_mask.sum()),
                    "rate": float(outlier_mask.mean()),
                }
        except Exception:
            self.preprocessing_summary = {"missing": {}, "outliers": {}}

        # 识别协变量
        self._identify_covariates(df)

        # 数据质量检查
        self._data_quality_check(df)

        self.data = df
        return df

    def get_variable_label(self, variable_name: str = None) -> str:
        """
        获取变量的中文标签

        参数:
            variable_name: 变量名，如果为None则使用self.demand_col

        返回:
            中文标签
        """
        if variable_name is None:
            variable_name = self.demand_col

        # 转换为小写进行比较（但保留原始大小写用于显示）
        var_lower = variable_name.lower()

        if str(getattr(self, "language", "en")).lower().startswith("en"):
            en_map = {
                "demand": "Demand",
                "quantity": "Quantity",
                "volume": "Volume",
                "units_sold": "Units Sold",
                "sales": "Sales",
                "revenue": "Revenue",
                "sales_revenue": "Sales Revenue",
                "value": "Value",
                "consumption": "Consumption",
            }
            for key, label in en_map.items():
                if var_lower == key.lower():
                    return label
            return variable_name

        for key, label in self.variable_labels.items():
            if var_lower == key.lower():
                return label

        # 如果没找到映射，使用变量名本身
        return variable_name

    def get_variable_label_with_unit(self, variable_name: str = None, unit: str = None) -> str:
        """
        获取带单位的变量中文标签

        参数:
            variable_name: 变量名
            unit: 单位，如'元'、'件'、'千瓦时'等

        返回:
            带单位的中文标签
        """
        label = self.get_variable_label(variable_name)

        if unit:
            return f"{label}({unit})"
        return label

    def _identify_covariates(self, df: pd.DataFrame):
        """识别协变量"""
        all_cols = set(df.columns)
        core_cols = {self.demand_col, self.time_col if isinstance(self.time_col, str) else ''}
        self.covariates = list(all_cols - core_cols - {''})

        if self.covariates:
            print(f"识别到 {len(self.covariates)} 个协变量: {self.covariates}")
        else:
            print("未识别到协变量")

    def _data_quality_check(self, df: pd.DataFrame):
        """数据质量检查"""
        print("\n" + "=" * 40)
        print("数据质量检查报告")
        print("=" * 40)

        # 检查缺失值
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_report = pd.DataFrame({
            '缺失值数量': missing,
            '缺失百分比%': missing_pct
        })

        missing_significant = missing_report[missing_report['缺失值数量'] > 0]

        if len(missing_significant) > 0:
            print("发现缺失值:")
            print(missing_significant.to_string())

            # 建议处理方法
            print("\n建议处理方法:")
            for col in missing_significant.index:
                if missing_pct[col] < 5:
                    print(f"  {col}: 缺失率{missing_pct[col]}% < 5%, 建议使用前向填充")
                elif missing_pct[col] < 30:
                    print(f"  {col}: 缺失率{missing_pct[col]}% < 30%, 建议使用插值法")
                else:
                    print(f"  {col}: 缺失率{missing_pct[col]}% >= 30%, 建议删除该列或使用模型预测")
        else:
            print("✓ 无缺失值")

        # 检查异常值（使用IQR方法）
        if self.demand_col in df.columns:
            Q1 = df[self.demand_col].quantile(0.25)
            Q3 = df[self.demand_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[self.demand_col] < lower_bound) | (df[self.demand_col] > upper_bound)]

            if len(outliers) > 0:
                print(f"\n发现 {len(outliers)} 个异常值 (IQR方法)")
                print(f"异常值范围: {lower_bound:.2f} ~ {upper_bound:.2f}")
                print(f"异常值占比: {len(outliers)/len(df)*100:.2f}%")
            else:
                print("✓ 无异常值")

        # 检查时间间隔
        if hasattr(df.index, 'freq') and df.index.freq is None:
            # 检查是否真的是不规则时间序列（不是所有NaN的情况）
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    unique_diffs = time_diffs.nunique()
                    if unique_diffs > 3:  # 多种不同的时间间隔
                        print(f"\n⚠ 警告: 时间序列不规则，检测到 {unique_diffs} 种不同时间间隔")
                        print(f"  最常见间隔: {time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 'N/A'}")
                        print("  对于不规则时间序列，某些模型可能需要数据重采样")
                    else:
                        print("\nℹ 时间序列相对规则，适合时间序列建模")
                else:
                    print("\nℹ 时间序列数据可用")

        # 时间序列分析
        self._time_series_analysis(df)

        print("=" * 40)

    def _time_series_analysis(self, df):
        """
        执行时间序列分析

        包括:
        1. 季节性分解 (趋势、季节性、残差)
        2. 平稳性检验 (ADF检验)
        3. 自相关和偏自相关分析
        4. 季节性检测
        """
        print("\n" + "=" * 40)
        print("时间序列分析")
        print("=" * 40)

        if self.demand_col not in df.columns:
            print(f"需求列 '{self.demand_col}' 不存在，跳过时间序列分析")
            return

        y = df[self.demand_col]

        # 1. 季节性分解
        decomposition = self._seasonal_decomposition(y)

        # 2. 平稳性检验
        stationarity_result = self._stationarity_test(y)

        # 3. ACF/PACF分析
        self._acf_pacf_analysis(y)

        # 4. 季节性检测
        seasonality_result = self._seasonality_detection(y)

        # 存储分析结果
        self.ts_analysis_results = {
            'decomposition': decomposition,
            'stationarity': stationarity_result,
            'seasonality': seasonality_result
        }

        print("时间序列分析完成")
        print("=" * 40)

    def _seasonal_decomposition(self, y):
        """执行季节性分解"""
        print("\n1. 季节性分解:")

        try:
            # 确定季节性周期
            if self.freq == 'D':
                period = 7  # 周季节性
            elif self.freq == 'W':
                period = 4  # 月季节性 (近似)
            elif self.freq == 'M':
                period = 12  # 年季节性
            elif self.freq == 'Q':
                period = 4  # 年季节性
            elif self.freq == 'Y':
                period = 1  # 无季节性
            else:
                period = None

            if period is not None and len(y) > period * 2:
                result = seasonal_decompose(y, model='additive', period=period)

                # 计算各个成分的贡献度
                trend_ratio = np.abs(result.trend).sum() / np.abs(y).sum() if result.trend is not None else 0
                seasonal_ratio = np.abs(result.seasonal).sum() / np.abs(y).sum() if result.seasonal is not None else 0
                residual_ratio = np.abs(result.resid).sum() / np.abs(y).sum() if result.resid is not None else 0

                print(f"  分解周期: {period}")
                print(f"  趋势成分占比: {trend_ratio:.2%}")
                print(f"  季节性成分占比: {seasonal_ratio:.2%}")
                print(f"  残差成分占比: {residual_ratio:.2%}")

                # 判断主要成分
                if seasonal_ratio > 0.3:
                    print("  ✅ 数据呈现强季节性")
                elif seasonal_ratio > 0.1:
                    print("  ⚠ 数据呈现中等季节性")
                else:
                    print("  ℹ 数据季节性较弱")

                if residual_ratio > 0.5:
                    print("  ⚠ 残差占比较高，表明存在未捕捉的模式或噪声")

                return {
                    'period': period,
                    'trend_ratio': trend_ratio,
                    'seasonal_ratio': seasonal_ratio,
                    'residual_ratio': residual_ratio,
                    'has_strong_seasonality': seasonal_ratio > 0.3
                }
            else:
                print("  数据长度不足，无法进行季节性分解")
                return None

        except Exception as e:
            print(f"  季节性分解失败: {e}")
            return None

    def _stationarity_test(self, y):
        """执行平稳性检验 (ADF检验)"""
        print("\n2. 平稳性检验 (ADF检验):")

        try:
            result = adfuller(y.dropna())

            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]

            print(f"  ADF统计量: {adf_statistic:.4f}")
            print(f"  P值: {p_value:.4f}")

            # 判断平稳性
            is_stationary = p_value < 0.05
            if is_stationary:
                print("  ✅ 数据是平稳的 (p < 0.05)")
            else:
                print("  ⚠ 数据是非平稳的 (p ≥ 0.05)")
                print("  建议进行差分处理")

            # 与临界值比较
            print("  临界值:")
            for key, value in critical_values.items():
                print(f"    {key}: {value:.4f}")

            return {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'critical_values': critical_values
            }

        except Exception as e:
            print(f"  平稳性检验失败: {e}")
            return None

    def _acf_pacf_analysis(self, y):
        """执行自相关和偏自相关分析"""
        print("\n3. 自相关(ACF)和偏自相关(PACF)分析:")

        try:
            # 计算ACF和PACF
            nlags = min(40, len(y) // 2)
            acf_values = acf(y.dropna(), nlags=nlags)
            pacf_values = pacf(y.dropna(), nlags=nlags)

            # 寻找显著滞后
            significant_acf = np.where(np.abs(acf_values) > 1.96 / np.sqrt(len(y)))[0]
            significant_pacf = np.where(np.abs(pacf_values) > 1.96 / np.sqrt(len(y)))[0]

            # 排除滞后0
            significant_acf = significant_acf[significant_acf > 0]
            significant_pacf = significant_pacf[significant_pacf > 0]

            print(f"  分析滞后数: {nlags}")

            if len(significant_acf) > 0:
                print(f"  显著ACF滞后: {significant_acf[:10].tolist()}" +
                      (f" (共{len(significant_acf)}个)" if len(significant_acf) > 10 else ""))

                # 检查季节性滞后
                if self.freq == 'D' and 7 in significant_acf:
                    print("  ✅ 检测到周度季节性 (滞后7)")
                elif self.freq == 'M' and 12 in significant_acf:
                    print("  ✅ 检测到年度季节性 (滞后12)")
            else:
                print("  无显著ACF滞后")

            if len(significant_pacf) > 0:
                print(f"  显著PACF滞后: {significant_pacf[:10].tolist()}" +
                      (f" (共{len(significant_pacf)}个)" if len(significant_pacf) > 10 else ""))
            else:
                print("  无显著PACF滞后")

            # ARIMA阶数建议
            if len(significant_acf) > 0 and len(significant_pacf) > 0:
                p = min(3, len(significant_pacf))
                q = min(3, len(significant_acf))
                print(f"  ARIMA阶数建议: p={p}, q={q} (基于显著滞后)")

            return {
                'significant_acf_lags': significant_acf.tolist(),
                'significant_pacf_lags': significant_pacf.tolist()
            }

        except Exception as e:
            print(f"  ACF/PACF分析失败: {e}")
            return None

    def _seasonality_detection(self, y):
        """检测季节性模式"""
        print("\n4. 季节性检测:")

        try:
            # 简单的季节性检测方法
            if len(y) < 14:
                print("  数据长度不足，无法进行季节性检测")
                return {'has_seasonality': False}

            # 方法1: 基于周期自相关
            if self.freq == 'D':
                test_periods = [7, 14, 30]  # 周、双周、月
            elif self.freq == 'W':
                test_periods = [4, 8, 13]  # 月、双月、季
            elif self.freq == 'M':
                test_periods = [3, 6, 12]  # 季、半年、年
            else:
                test_periods = []

            seasonality_found = False
            for period in test_periods:
                if len(y) > period * 2:
                    # 计算周期性自相关
                    autocorr = y.autocorr(lag=period)
                    if abs(autocorr) > 0.5:
                        print(f"  ✅ 检测到{period}期季节性 (自相关: {autocorr:.3f})")
                        seasonality_found = True

            if not seasonality_found:
                print("  ℹ 未检测到明显季节性模式")

            # 方法2: 基于方差分析 (简化)
            if self.freq == 'D' and len(y) >= 28:
                # 检查周内模式
                day_of_week = y.index.dayofweek if hasattr(y.index, 'dayofweek') else None
                if day_of_week is not None:
                    weekday_means = y.groupby(day_of_week).mean()
                    if weekday_means.std() / weekday_means.mean() > 0.2:
                        print("  ✅ 检测到周内模式变化")
                        seasonality_found = True

            return {
                'has_seasonality': seasonality_found,
                'suggested_periods': test_periods
            }

        except Exception as e:
            print(f"  季节性检测失败: {e}")
            return {'has_seasonality': False}

    def prepare_data(self):
        """准备训练和测试数据"""
        if self.data is None:
            raise ValueError("请先加载数据")

        y_full = self.data[self.demand_col].astype(float)
        self.y = y_full

        split_idx = int(len(y_full) * (1 - self.test_size))
        if len(y_full) >= 2:
            split_idx = max(1, min(split_idx, len(y_full) - 1))
        else:
            split_idx = 1
        split_time = y_full.index[split_idx]

        self.train_data = {
            'X': None,
            'y': y_full.iloc[:split_idx]
        }

        self.test_data = {
            'X': None,
            'y': y_full.iloc[split_idx:]
        }

        if self.covariates:
            X_all = self.data[self.covariates].copy()
            # 为协变量模型也添加时间特征和滞后特征
            df_temp = self.data[[self.demand_col]].copy()
            lag_features = []
            for i in range(1, 6):  # 默认加5阶滞后
                lag_name = f'lag_{i}'
                df_temp[lag_name] = df_temp[self.demand_col].shift(i)
                lag_features.append(lag_name)
            
            X_all['dayofweek'] = X_all.index.dayofweek if hasattr(X_all.index, 'dayofweek') else 0
            X_all['month'] = X_all.index.month if hasattr(X_all.index, 'month') else 0
            X_all['quarter'] = X_all.index.quarter if hasattr(X_all.index, 'quarter') else 0
            X_all['year'] = X_all.index.year if hasattr(X_all.index, 'year') else 0
            
            for lag in lag_features:
                X_all[lag] = df_temp[lag]
            
            y_all = y_full
            # Dropna for aligned index
            valid_idx = X_all.dropna().index
            X_all = X_all.loc[valid_idx]
            y_all = y_all.loc[valid_idx]
        else:
            X_all, y_all = self._create_lag_features(n_lags=5)

        self.feature_data = {'X': X_all, 'y': y_all}
        train_mask = y_all.index < split_time
        test_mask = y_all.index >= split_time
        self.train_features = {'X': X_all.loc[train_mask], 'y': y_all.loc[train_mask]}
        self.test_features = {'X': X_all.loc[test_mask], 'y': y_all.loc[test_mask]}

        print(f"数据划分完成:")
        print(f"  训练集: {len(self.train_data['y'])} 个观测值")
        print(f"  测试集: {len(self.test_data['y'])} 个观测值 ({self.test_size*100}%)")

    def _create_lag_features(self, n_lags: int = 5):
        """创建滞后特征"""
        df = self.data[[self.demand_col]].copy()
        lag_features = []
        for i in range(1, int(n_lags) + 1):
            lag_name = f'lag_{i}'
            df[lag_name] = df[self.demand_col].shift(i)
            lag_features.append(lag_name)

        # 始终添加时间特征（帮助模型理解周期）
        df['dayofweek'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        df['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 0
        df['year'] = df.index.year if hasattr(df.index, 'year') else 0
        lag_features.extend(['dayofweek', 'month', 'quarter', 'year'])

        df = df.dropna()
        X = df[lag_features].copy()
        y = df[self.demand_col].astype(float).copy()
        return X, y

    def run_baseline_models(self):
        """
        Step 2: 建立时间序列基准

        拟合以下经典模型作为分析基础:
        1. Naïve / Seasonal Naïve: 作为最简单的基准
        2. 移动平均线 (MA): 自动选择最佳窗口长度
        3. 指数平滑 (ETS): 根据数据特征自动选择 Simple, Holt 或 Holt-Winters
        4. ARIMA / SARIMA: 通过 AIC 自动确定阶数
        """
        print("\n" + "=" * 60)
        print("Step 2: 建立时间序列基准模型")
        print("=" * 60)

        if self.train_data is None:
            self.prepare_data()

        y_train = self.train_data['y']
        y_test = self.test_data['y']

        def _safe_mape(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                m = np.abs((y_true - y_pred) / y_true)
                m = np.where(np.isfinite(m), m, 1.0)
            return float(np.mean(m) * 100)

        def _metrics(y_true, y_pred):
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAPE": _safe_mape(y_true, y_pred),
            }

        # 1. Naïve 模型
        print("\n1. 拟合Naïve模型...")
        naive_forecast = y_train.iloc[-1]
        naive_predictions = np.full(len(y_test), naive_forecast)

        self.baseline_models['naive'] = {
            'predictions': naive_predictions,
            'fitted': y_train.shift(1).values,
            'model': None,
            'params': {'strategy': 'last_observation'}
        }

        # 2. Seasonal Naïve (如果数据有季节性)
        print("2. 拟合Seasonal Naïve模型...")
        # 简单实现: 使用上一个周期的同期值
        if len(y_train) >= 7:  # 假设至少一周数据用于日频数据
            seasonal_period = 7 if self.freq == 'D' else 4 if self.freq == 'W' else 12 if self.freq == 'M' else 1
            seasonal_period = int(max(1, seasonal_period))
            seasonal_naive_predictions = np.array(
                [float(y_train.iloc[-seasonal_period + (i % seasonal_period)]) for i in range(int(len(y_test)))],
                dtype=float,
            )

            self.baseline_models['seasonal_naive'] = {
                'predictions': seasonal_naive_predictions,
                'fitted': y_train.shift(seasonal_period).values,
                'model': None,
                'params': {'seasonal_period': int(seasonal_period)}
            }

        # 3. 移动平均 (MA)
        print("3. 拟合移动平均(MA)模型...")
        self._fit_moving_average(y_train, y_test)

        # 4. 指数平滑 (ETS)
        print("4. 拟合指数平滑(ETS)模型...")
        self._fit_exponential_smoothing(y_train, y_test)

        # 5. ARIMA/SARIMA
        print("5. 拟合ARIMA/SARIMA模型...")
        self._fit_arima(y_train, y_test)

        # 评估基准模型
        print("\n基准模型评估:")
        print("-" * 40)
        for model_name, result in self.baseline_models.items():
            predictions = result['predictions']
            if len(predictions) == len(y_test):
                result['holdout_metrics'] = _metrics(y_test, predictions)
                hm = result['holdout_metrics']
                print(f"{model_name:20s} MAE: {hm['MAE']:.2f}, RMSE: {hm['RMSE']:.2f}, MAPE: {hm['MAPE']:.2f}%")

        print("=" * 60)

    def _fit_moving_average(self, y_train, y_test):
        """拟合移动平均模型"""
        windows = [3, 5, 7, 10, 14, 21, 28]
        windows = [w for w in windows if w < len(y_train)]
        if not windows:
            return

        # 简单“AI选择”：在训练集尾部做一段验证，选择验证误差最小的窗口
        val_len = max(1, int(len(y_train) * 0.2))
        sub_train = y_train.iloc[:-val_len] if len(y_train) > val_len + 5 else y_train.iloc[:-1]
        val_y = y_train.iloc[len(sub_train):]

        def _ma_forecast(series, window, horizon):
            history = [float(v) for v in series.astype(float).values]
            out = []
            for _ in range(int(horizon)):
                tail = history[-int(window):] if len(history) >= int(window) else history
                if not tail:
                    out.append(float("nan"))
                    history.append(float("nan"))
                    continue
                pred = float(np.mean(tail))
                out.append(pred)
                history.append(pred)
            return np.asarray(out, dtype=float)

        best = (None, float("inf"))
        for window in windows:
            if window >= len(sub_train):
                continue
            preds_val = _ma_forecast(sub_train, window, len(val_y))
            mae = float(mean_absolute_error(val_y, preds_val))
            if mae < best[1]:
                best = (window, mae)

        best_window = best[0] or windows[0]
        preds_test = _ma_forecast(y_train, best_window, len(y_test))

        # in-sample fitted：用前window的均值预测当前点（shift避免泄露）
        fitted = y_train.rolling(window=best_window).mean().shift(1).values

        self.baseline_models['moving_average'] = {
            'predictions': preds_test,
            'fitted': fitted,
            'model': None,
            'params': {'window': int(best_window), 'selection': 'validation_mae'},
        }

    def _fit_exponential_smoothing(self, y_train, y_test):
        """拟合指数平滑模型"""
        try:
            def _seasonal_periods_default():
                if self.freq == "D":
                    return 7
                if self.freq == "W":
                    return 52
                if self.freq == "M":
                    return 12
                if self.freq == "Q":
                    return 4
                return 7

            sp = None
            try:
                seas = getattr(self, "ts_analysis_results", {}) or {}
                seas = seas.get("seasonality") or {}
                if seas.get("has_seasonality") and seas.get("suggested_periods"):
                    sp = int(seas.get("suggested_periods")[0])
            except Exception:
                sp = None
            if not sp:
                sp = _seasonal_periods_default()
            candidates = [
                ("ets_simple", {"trend": None, "seasonal": None, "seasonal_periods": None}),
                ("ets_holt", {"trend": "add", "seasonal": None, "seasonal_periods": None}),
            ]
            if sp and len(y_train) >= 2 * sp:
                candidates.append(("ets_holt_winters", {"trend": "add", "seasonal": "add", "seasonal_periods": int(sp)}))

            for model_key, params in candidates:
                try:
                    model = ExponentialSmoothing(
                        y_train,
                        trend=params["trend"],
                        seasonal=params["seasonal"],
                        seasonal_periods=params["seasonal_periods"],
                    )
                    fitted_model = model.fit(optimized=True)
                    predictions = fitted_model.forecast(len(y_test))
                    fitted_vals = np.asarray(getattr(fitted_model, "fittedvalues", np.full(len(y_train), np.nan)), dtype=float)
                    fitted_params = dict(getattr(fitted_model, "params", {}) or {})
                    self.baseline_models[model_key] = {
                        "predictions": np.asarray(predictions, dtype=float),
                        "fitted": fitted_vals,
                        "model": None,
                        "params": {
                            "config": params,
                            "fitted_params": fitted_params,
                            "aic": float(getattr(fitted_model, "aic", np.nan)) if getattr(fitted_model, "aic", None) is not None else None,
                        },
                    }
                except Exception:
                    continue

        except Exception as e:
            print(f"指数平滑模型拟合失败: {e}")

    def _fit_arima(self, y_train, y_test):
        """拟合ARIMA模型"""
        try:
            if str(os.getenv("FORECASTPRO_SKIP_ARIMA", "")).strip().lower() in {"1", "true", "yes"}:
                return

            def _seasonal_periods_default():
                if self.freq == "D":
                    return 7
                if self.freq == "W":
                    return 52
                if self.freq == "M":
                    return 12
                if self.freq == "Q":
                    return 4
                return None

            fast = str(os.getenv("FORECASTPRO_FAST", "")).strip().lower() in {"1", "true", "yes"}
            maxiter_env = os.getenv("FORECASTPRO_ARIMA_MAXITER")
            try:
                maxiter = int(maxiter_env) if maxiter_env is not None else (25 if fast else 80)
            except Exception:
                maxiter = 25 if fast else 80

            if fast:
                orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
            else:
                orders = [(p, d, q) for p in range(0, 6) for d in range(0, 3) for q in range(0, 6)]

            sp = None
            try:
                seas = getattr(self, "ts_analysis_results", {}) or {}
                seas = seas.get("seasonality") or {}
                if seas.get("has_seasonality") and seas.get("suggested_periods"):
                    sp = int(seas.get("suggested_periods")[0])
            except Exception:
                sp = None
            if not sp:
                sp = _seasonal_periods_default()

            use_seasonal = (not fast) and bool(sp) and int(sp) > 1 and len(y_train) >= 2 * int(sp)
            seasonal_orders = [(0, 0, 0, 0)]
            if use_seasonal:
                seasonal_orders = [(P, D, Q, int(sp)) for P in (0, 1, 2) for D in (0, 1) for Q in (0, 1, 2)]

            best = {"aic": float("inf"), "order": None, "seasonal_order": None, "fitted": None, "pred": None, "params": None}
            max_seconds = os.getenv("FORECASTPRO_ARIMA_MAX_SECONDS")
            try:
                max_seconds = float(max_seconds) if max_seconds is not None else (8.0 if fast else 25.0)
            except Exception:
                max_seconds = 8.0 if fast else 25.0
            t0 = time.monotonic()

            for order in orders:
                for sorder in seasonal_orders:
                    if (time.monotonic() - t0) > max_seconds:
                        break
                    try:
                        if fast:
                            model = ARIMA(y_train, order=order)
                            fitted_model = model.fit(method_kwargs={"maxiter": maxiter})
                        else:
                            model = SARIMAX(
                                y_train,
                                order=order,
                                seasonal_order=sorder if use_seasonal else (0, 0, 0, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            fitted_model = model.fit(disp=False, maxiter=maxiter)
                        aic = float(getattr(fitted_model, "aic", np.inf))
                        if not np.isfinite(aic) or aic >= best["aic"]:
                            continue
                        pred = fitted_model.forecast(len(y_test))
                        fitted_vals = np.asarray(getattr(fitted_model, "fittedvalues", np.full(len(y_train), np.nan)), dtype=float)
                        best = {
                            "aic": aic,
                            "order": order,
                            "seasonal_order": sorder if use_seasonal else (0, 0, 0, 0),
                            "fitted": fitted_vals,
                            "pred": np.asarray(pred, dtype=float),
                            "params": dict(getattr(fitted_model, "params", {}) or {}),
                        }
                    except Exception:
                        continue
                if (time.monotonic() - t0) > max_seconds:
                    break

            if best["pred"] is not None:
                self.baseline_models['arima'] = {
                    'predictions': best["pred"],
                    'fitted': best["fitted"],
                    'model': None,
                    'params': {
                        'order': best["order"],
                        'seasonal_order': best["seasonal_order"] if use_seasonal else None,
                        'aic': best["aic"],
                        'fitted_params': best["params"],
                        'selection': 'aic_grid_search',
                    }
                }

        except Exception as e:
            print(f"ARIMA模型拟合失败: {e}")

    def run_advanced_models(self):
        """
        Step 3: 高级模型探索与选择

        在基准之上，利用AI推理选择并运行以下至少一类高级模型:
        1. 回归分析: 基于滞后需求和协变量的OLS或Lasso/Ridge回归
        2. 机器学习: XGBoost、Random Forest，需配合时间感知交叉验证
        3. 深度学习/基础模型: LSTM、Temporal Fusion Transformer或TimeGPT
        4. 混合模型: 如使用ML修正ARIMA的残差
        """
        print("\n" + "=" * 60)
        print("Step 3: 高级模型探索与选择")
        print("=" * 60)

        if self.train_data is None:
            self.prepare_data()

        n_total = int(len(self.data)) if self.data is not None else int(len(self.train_data["y"]) + len(self.test_data["y"]))
        if n_total < 30:
            print("数据量 n < 30：跳过高级模型（过拟合风险极高）")
            return

        X_train = getattr(self, "train_features", None)
        X_test = getattr(self, "test_features", None)
        if not X_train or not X_test:
            print("高级模型不可用: 缺少特征数据")
            return

        X_train = X_train.get("X")
        y_train = self.train_features.get("y") if hasattr(self, "train_features") else None
        X_test = X_test.get("X")
        y_test = self.test_features.get("y") if hasattr(self, "test_features") else None

        if X_train is None or y_train is None or X_test is None or y_test is None:
            print("高级模型不可用: 特征数据为空")
            return
        if len(X_train) < 5 or len(X_test) < 1:
            print("高级模型不可用: 数据量过少")
            return

        # 1. 回归分析
        print("\n1. 拟合回归模型...")
        self._fit_regression_models(X_train, y_train, X_test, y_test)

        # 2. 机器学习模型
        print("\n2. 拟合机器学习模型...")
        self._fit_ml_models(X_train, y_train, X_test, y_test)

        # 评估高级模型
        print("\n高级模型评估:")
        print("-" * 40)
        for model_name, result in self.advanced_models.items():
            predictions = result['predictions']

            if len(predictions) == len(y_test):
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                # 避免除零错误：当y_test为0时，使用一个小值或跳过
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_array = np.abs((y_test - predictions) / y_test)
                    # 将无限大和NaN值替换为0或一个大的惩罚值
                    mape_array = np.where(np.isfinite(mape_array), mape_array, 1.0)  # 100%误差
                    mape = np.mean(mape_array) * 100

                result['metrics'] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }

                # 检查过拟合风险
                if 'train_score' in result and 'test_score' in result:
                    train_score = result['train_score']
                    test_score = result['test_score']

                    if train_score - test_score > 0.1:  # 训练集性能明显优于测试集
                        result['overfitting_risk'] = '高'
                    elif train_score - test_score > 0.05:
                        result['overfitting_risk'] = '中'
                    else:
                        result['overfitting_risk'] = '低'

                print(f"{model_name:20s} MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        print("=" * 60)

    def _fit_regression_models(self, X_train, y_train, X_test, y_test):
        """拟合回归模型"""
        def _predict_no_leakage(model, X_cols, horizon):
            lag_cols = [c for c in X_cols if isinstance(c, str) and c.startswith("lag_")]
            
            if not lag_cols:
                return np.asarray(model.predict(X_test), dtype=float)

            try:
                n_lags = max(int(str(c).split("_", 1)[1]) for c in lag_cols)
            except Exception:
                n_lags = 0
                
            if n_lags <= 0:
                return np.asarray(model.predict(X_test), dtype=float)

            hist = [float(v) for v in self.train_data["y"].astype(float).values]
            last_known = {}
            for c in X_cols:
                if isinstance(c, str) and c.startswith("lag_"):
                    continue
                try:
                    if self.covariates and c in self.covariates:
                        raw_val = self.data[c].iloc[-1]
                    else:
                        raw_val = X_test[c].iloc[-1] if not X_test.empty else 0.0
                    
                    if isinstance(raw_val, pd.Series):
                        raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                    last_known[c] = float(raw_val)
                except Exception:
                    last_known[c] = 0.0

            idx = self.test_data["y"].index
            preds = []
            
            last_date = self.train_data["y"].index[-1] if not self.train_data["y"].empty else self.data.index[-1]
            for i in range(int(horizon)):
                ts = idx[i] if i < len(idx) else None
                if ts is None:
                    if self.freq == "D":
                        last_date += pd.Timedelta(days=1)
                    elif self.freq == "W":
                        last_date += pd.Timedelta(weeks=1)
                    elif self.freq == "M":
                        last_date += pd.DateOffset(months=1)
                    elif self.freq == "Q":
                        last_date += pd.DateOffset(months=3)
                    else:
                        last_date += pd.Timedelta(days=1)
                    ts_dt = pd.to_datetime(last_date)
                else:
                    ts_dt = pd.to_datetime(ts)
                    last_date = ts_dt
                
                row = {f"lag_{k}": hist[-k] for k in range(1, n_lags + 1)}
                
                if "dayofweek" in X_cols:
                    row["dayofweek"] = float(ts_dt.dayofweek)
                if "month" in X_cols:
                    row["month"] = float(ts_dt.month)
                if "quarter" in X_cols:
                    row["quarter"] = float(ts_dt.quarter)
                if "year" in X_cols:
                    row["year"] = float(ts_dt.year)
                        
                for c in X_cols:
                    if c in row: # 已经作为滞后或时间特征处理过了
                        continue
                    v = last_known.get(c, 0.0)
                    if ts is not None:
                        try:
                            # 尝试从原始数据中获取真实特征值（除了滞后项）
                            # 注意：如果是时间特征(dayofweek等)，它们本来就在 self.data 中
                            if self.covariates and c in self.covariates:
                                raw_val = self.data.loc[ts, c]
                            else:
                                if ts in X_test.index:
                                    raw_val = X_test.loc[ts, c]
                                else:
                                    raw_val = v
                            
                            if isinstance(raw_val, pd.Series):
                                raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                            v = float(raw_val)
                        except Exception:
                            pass
                    row[c] = v
                
                # 保证顺序一致
                ordered_row = {c: float(row.get(c, 0.0)) for c in X_cols}
                X_row = pd.DataFrame([ordered_row])
                
                # 对于线性模型等不支持特征名称缺失或顺序错误的，确保列名一致
                X_row = X_row[X_cols]
                
                y_next = float(model.predict(X_row)[0])
                preds.append(y_next)
                hist.append(y_next)
            return np.asarray(preds, dtype=float)

        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1)
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = _predict_no_leakage(model, X_train.columns, len(y_test))

                self.advanced_models[name] = {
                    'predictions': predictions,
                    'fitted': model.predict(X_train),
                    'model': model,
                    'params': model.get_params(),
                    'train_score': model.score(X_train, y_train),
                    'test_score': float(r2_score(np.asarray(y_test.values, dtype=float), np.asarray(predictions, dtype=float)))
                }

            except Exception as e:
                print(f"  {name} 拟合失败: {e}")

    def _fit_ml_models(self, X_train, y_train, X_test, y_test):
        """拟合机器学习模型"""
        def _predict_no_leakage(model, X_cols, horizon):
            lag_cols = [c for c in X_cols if isinstance(c, str) and c.startswith("lag_")]
            
            if not lag_cols:
                return np.asarray(model.predict(X_test), dtype=float)

            try:
                n_lags = max(int(str(c).split("_", 1)[1]) for c in lag_cols)
            except Exception:
                n_lags = 0
                
            if n_lags <= 0:
                return np.asarray(model.predict(X_test), dtype=float)

            hist = [float(v) for v in self.train_data["y"].astype(float).values]
            last_known = {}
            for c in X_cols:
                if isinstance(c, str) and c.startswith("lag_"):
                    continue
                try:
                    if self.covariates and c in self.covariates:
                        raw_val = self.data[c].iloc[-1]
                    else:
                        raw_val = X_test[c].iloc[-1] if not X_test.empty else 0.0
                    
                    if isinstance(raw_val, pd.Series):
                        raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                    last_known[c] = float(raw_val)
                except Exception:
                    last_known[c] = 0.0

            idx = self.test_data["y"].index
            preds = []
            
            last_date = self.train_data["y"].index[-1] if not self.train_data["y"].empty else self.data.index[-1]
            for i in range(int(horizon)):
                ts = idx[i] if i < len(idx) else None
                if ts is None:
                    if self.freq == "D":
                        last_date += pd.Timedelta(days=1)
                    elif self.freq == "W":
                        last_date += pd.Timedelta(weeks=1)
                    elif self.freq == "M":
                        last_date += pd.DateOffset(months=1)
                    elif self.freq == "Q":
                        last_date += pd.DateOffset(months=3)
                    else:
                        last_date += pd.Timedelta(days=1)
                    ts_dt = pd.to_datetime(last_date)
                else:
                    ts_dt = pd.to_datetime(ts)
                    last_date = ts_dt
                
                row = {f"lag_{k}": hist[-k] for k in range(1, n_lags + 1)}
                
                if "dayofweek" in X_cols:
                    row["dayofweek"] = float(ts_dt.dayofweek)
                if "month" in X_cols:
                    row["month"] = float(ts_dt.month)
                if "quarter" in X_cols:
                    row["quarter"] = float(ts_dt.quarter)
                if "year" in X_cols:
                    row["year"] = float(ts_dt.year)
                        
                for c in X_cols:
                    if c in row: # 已经作为滞后或时间特征处理过了
                        continue
                    v = last_known.get(c, 0.0)
                    if ts is not None:
                        try:
                            # 尝试从原始数据中获取真实特征值（除了滞后项）
                            # 注意：如果是时间特征(dayofweek等)，它们本来就在 self.data 中
                            if self.covariates and c in self.covariates:
                                raw_val = self.data.loc[ts, c]
                            else:
                                if ts in X_test.index:
                                    raw_val = X_test.loc[ts, c]
                                else:
                                    raw_val = v
                            
                            if isinstance(raw_val, pd.Series):
                                raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                            v = float(raw_val)
                        except Exception:
                            pass
                    row[c] = v
                
                # 保证顺序一致
                ordered_row = {c: float(row.get(c, 0.0)) for c in X_cols}
                X_row = pd.DataFrame([ordered_row])
                
                # 对于线性模型等不支持特征名称缺失或顺序错误的，确保列名一致
                X_row = X_row[X_cols]
                
                y_next = float(model.predict(X_row)[0])
                preds.append(y_next)
                hist.append(y_next)
            return np.asarray(preds, dtype=float)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)

        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed
            )

            # 使用时间序列交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                rf_model.fit(X_train_cv, y_train_cv)
                cv_score = rf_model.score(X_val_cv, y_val_cv)
                cv_scores.append(cv_score)

            # 在整个训练集上训练
            rf_model.fit(X_train, y_train)
            predictions = _predict_no_leakage(rf_model, X_train.columns, len(y_test))

            self.advanced_models['random_forest'] = {
                'predictions': predictions,
                'fitted': rf_model.predict(X_train),
                'model': rf_model,
                'params': rf_model.get_params(),
                'train_score': rf_model.score(X_train, y_train),
                'test_score': float(r2_score(np.asarray(y_test.values, dtype=float), np.asarray(predictions, dtype=float))),
                'cv_mean_score': np.mean(cv_scores)
            }

        except Exception as e:
            print(f"  Random Forest 拟合失败: {e}")

        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed
            )

            xgb_model.fit(X_train, y_train)
            predictions = _predict_no_leakage(xgb_model, X_train.columns, len(y_test))

            self.advanced_models['xgboost'] = {
                'predictions': predictions,
                'fitted': xgb_model.predict(X_train),
                'model': xgb_model,
                'params': xgb_model.get_params(),
                'train_score': xgb_model.score(X_train, y_train),
                'test_score': float(r2_score(np.asarray(y_test.values, dtype=float), np.asarray(predictions, dtype=float)))
            }

        except Exception as e:
            print(f"  XGBoost 拟合失败: {e}")

    def _fit_dl_models(self, X_train, y_train, X_test, y_test):
        """拟合深度学习模型"""
        # 简化实现 - 实际应用中需要更复杂的网络架构
        try:
            # 简单LSTM模型
            if len(X_train.shape) == 2:
                # 重塑数据为LSTM输入格式 [samples, timesteps, features]
                n_features = X_train.shape[1]

                # 创建简单的神经网络
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])

                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )

                # 训练模型
                model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )

                predictions = _predict_no_leakage(model, X_train.columns, len(y_test))

                self.advanced_models['neural_network'] = {
                    'predictions': predictions,
                    'model': model,
                    'train_score': model.evaluate(X_train, y_train, verbose=0)[1],
                    'test_score': float(r2_score(np.asarray(y_test.values, dtype=float), np.asarray(predictions, dtype=float))),
                }

        except Exception as e:
            print(f"  深度学习模型拟合失败: {e}")

    def _fit_hybrid_models(self, X_train, y_train, X_test, y_test):
        """拟合混合模型"""
        # 混合模型: ARIMA + 机器学习残差修正
        try:
            # 使用ARIMA作为基础模型
            if 'arima' in self.baseline_models:
                arima_predictions = self.baseline_models['arima']['predictions']

                # 计算ARIMA残差
                if len(arima_predictions) == len(y_test):
                    # 这里简化实现 - 实际需要更复杂的残差建模
                    residuals = y_test.values - arima_predictions

                    # 使用随机森林修正残差
                    rf_residual = RandomForestRegressor(
                        n_estimators=50,
                        random_state=self.random_seed
                    )

                    # 需要为残差模型准备特征
                    # 这里简化: 使用原始特征
                    rf_residual.fit(X_train, y_train - y_train.mean())
                    residual_predictions = rf_residual.predict(X_test)

                    # 混合预测 = ARIMA预测 + 残差修正
                    hybrid_predictions = arima_predictions + residual_predictions

                    self.advanced_models['arima_rf_hybrid'] = {
                        'predictions': hybrid_predictions,
                        'fitted': None,
                        'model': (self.baseline_models['arima'], rf_residual),
                        'params': {
                            'base': 'arima',
                            'residual_model': rf_residual.get_params(),
                        },
                        'type': 'hybrid'
                    }

        except Exception as e:
            print(f"  混合模型拟合失败: {e}")

    def evaluate_models(self):
        """
        Step 4: 模型评估与诊断

        留出测试 (Hold-out): 使用最后20%的观测值作为测试集
        统一指标: 报告所有模型的MAE, RMSE和MAPE
        警示标识: 必须明确指出高级模型中是否存在过拟合或数据泄露的风险
        """
        print("\n" + "=" * 60)
        print("Step 4: 模型评估与诊断")
        print("=" * 60)

        if not self.baseline_models and not self.advanced_models:
            print("警告: 没有已训练的模型，请先运行基准和高级模型")
            return

        y_test = self.test_data['y']
        y_train_baseline = self.train_data['y']
        y_train_advanced = None
        try:
            y_train_advanced = self.train_features.get("y") if hasattr(self, "train_features") else None
        except Exception:
            y_train_advanced = None

        # 收集所有模型结果
        all_models = {}
        all_models.update(self.baseline_models)
        all_models.update(self.advanced_models)

        # 计算指标
        evaluation_results = []

        model_details = {}
        for model_name, result in all_models.items():
            predictions = result['predictions']

            if len(predictions) != len(y_test):
                print(f"警告: {model_name} 预测长度不匹配")
                continue

            # 计算指标
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_array = np.abs((y_test - predictions) / y_test)
                mape_array = np.where(np.isfinite(mape_array), mape_array, 1.0)
                mape = np.mean(mape_array) * 100

            resid_pvalue = None
            resid_white_noise = None
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                resid = np.asarray(y_test.values, dtype=float) - np.asarray(predictions, dtype=float)
                resid = resid[np.isfinite(resid)]
                if resid.size >= 8:
                    lag = int(min(10, resid.size - 1))
                    lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
                    resid_pvalue = float(lb["lb_pvalue"].iloc[-1])
                    resid_white_noise = bool(resid_pvalue > 0.05)
            except Exception:
                resid_pvalue = None
                resid_white_noise = None

            in_sample_metrics = None
            fitted = result.get('fitted')
            if fitted is not None:
                try:
                    fitted_arr = np.asarray(fitted, dtype=float)
                    y_ref = y_train_baseline
                    if model_name in self.advanced_models and y_train_advanced is not None:
                        y_ref = y_train_advanced
                    y_arr = np.asarray(y_ref.values, dtype=float)
                    n = min(len(fitted_arr), len(y_arr))
                    yt = y_arr[:n]
                    yp = fitted_arr[:n]
                    mask = np.isfinite(yt) & np.isfinite(yp)
                    if int(mask.sum()) >= 5:
                        in_sample_metrics = {
                            "MAE": float(mean_absolute_error(yt[mask], yp[mask])),
                            "RMSE": float(np.sqrt(mean_squared_error(yt[mask], yp[mask]))),
                            "MAPE": float(np.mean(np.where(np.isfinite(np.abs((yt[mask] - yp[mask]) / yt[mask])), np.abs((yt[mask] - yp[mask]) / yt[mask]), 1.0)) * 100),
                        }
                except Exception:
                    in_sample_metrics = None

            # 检查过拟合风险
            overfitting_risk = '低'
            if model_name in self.advanced_models:
                if 'overfitting_risk' in self.advanced_models[model_name]:
                    overfitting_risk = self.advanced_models[model_name]['overfitting_risk']
                elif 'train_score' in result and 'test_score' in result:
                    train_score = result['train_score']
                    test_score = result['test_score']

                    if train_score - test_score > 0.1:
                        overfitting_risk = '高'
                    elif train_score - test_score > 0.05:
                        overfitting_risk = '中'

            evaluation_results.append({
                'model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'MAE_test': float(mae),
                'RMSE_test': float(rmse),
                'MAPE_test': float(mape),
                'MAE_train': in_sample_metrics["MAE"] if in_sample_metrics else None,
                'RMSE_train': in_sample_metrics["RMSE"] if in_sample_metrics else None,
                'MAPE_train': in_sample_metrics["MAPE"] if in_sample_metrics else None,
                'in_sample_MAE': in_sample_metrics["MAE"] if in_sample_metrics else None,
                'in_sample_RMSE': in_sample_metrics["RMSE"] if in_sample_metrics else None,
                'in_sample_MAPE': in_sample_metrics["MAPE"] if in_sample_metrics else None,
                'resid_lb_pvalue': resid_pvalue,
                'resid_white_noise': resid_white_noise,
                'overfitting_risk': overfitting_risk,
                'type': 'baseline' if model_name in self.baseline_models else 'advanced'
            })

            model_details[model_name] = {
                "type": 'baseline' if model_name in self.baseline_models else 'advanced',
                "params": result.get("params"),
                "in_sample": in_sample_metrics,
                "holdout": {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)},
                "residual_diagnostics": {"ljung_box_pvalue": resid_pvalue, "white_noise": resid_white_noise},
                "overfitting_risk": overfitting_risk,
            }

        # 按测试集MAPE排序（若缺失则回退RMSE）
        evaluation_df = pd.DataFrame(evaluation_results)
        if "MAPE" in evaluation_df.columns and evaluation_df["MAPE"].notna().any():
            evaluation_df = evaluation_df.sort_values('MAPE')
        else:
            evaluation_df = evaluation_df.sort_values('RMSE')

        # 保存评估结果
        self.evaluation_results = evaluation_df
        self.model_results = model_details

        n_total = int(len(self.data)) if self.data is not None else int(len(y_train_baseline) + len(y_test))
        baseline_df = evaluation_df[evaluation_df["type"] == "baseline"].copy()
        advanced_df = evaluation_df[evaluation_df["type"] == "advanced"].copy()

        best_model_row = None
        selection_reason = ""
        primary_metric = "MAPE" if ("MAPE" in evaluation_df.columns and evaluation_df["MAPE"].notna().any()) else "RMSE"

        if n_total < 30:
            allowed = baseline_df[baseline_df["model"].astype(str).isin(["naive"]) | baseline_df["model"].astype(str).str.startswith("ets")]
            if len(allowed) > 0:
                best_model_row = allowed.sort_values(primary_metric).iloc[0]
                selection_reason = "n < 30：强制使用ETS或Naive（避免高级模型过拟合）"
            elif len(baseline_df) > 0:
                best_model_row = baseline_df.sort_values(primary_metric).iloc[0]
                selection_reason = "n < 30：ETS不可用时回退到最佳基线模型"
        else:
            best_baseline = baseline_df.sort_values(primary_metric).iloc[0] if len(baseline_df) > 0 else None
            best_advanced = advanced_df.sort_values(primary_metric).iloc[0] if len(advanced_df) > 0 else None

            if best_baseline is None and best_advanced is not None:
                best_model_row = best_advanced
                selection_reason = "无可用基线模型，选择最佳高级模型"
            elif best_baseline is not None and best_advanced is None:
                best_model_row = best_baseline
                selection_reason = "无可用高级模型，选择最佳基线模型"
            elif best_baseline is not None and best_advanced is not None:
                score_b = float(best_baseline[primary_metric])
                score_a = float(best_advanced[primary_metric])
                if score_a < score_b:
                    best_model_row = best_advanced
                    selection_reason = f"高级模型在测试集上的{primary_metric}表现更优"
                else:
                    best_model_row = best_baseline
                    selection_reason = f"基线模型在测试集上的{primary_metric}表现更优或相当"

        if best_model_row is None:
            best_model_row = evaluation_df.iloc[0]
            selection_reason = "缺少可用模型对比，回退到RMSE最小模型"

        self.best_model = str(best_model_row['model'])
        self.best_model_selection_reason = selection_reason

        # 打印评估报告
        print(f"\n模型性能排名 (按{primary_metric}升序):")
        print("-" * 80)
        print(evaluation_df.to_string(index=False))

        print(f"\n最佳模型: {self.best_model}")
        try:
            print(f"测试集{primary_metric}: {float(best_model_row[primary_metric]):.4f}")
        except Exception:
            print(f"测试集RMSE: {float(best_model_row['RMSE']):.4f}")
        if selection_reason:
            print(f"选择逻辑: {selection_reason}")

        # 过拟合风险报告
        high_risk_models = evaluation_df[evaluation_df['overfitting_risk'] == '高']
        if len(high_risk_models) > 0:
            print("\n⚠ 过拟合高风险模型:")
            for _, row in high_risk_models.iterrows():
                print(f"  {row['model']}: RMSE={float(row['RMSE']):.4f}")

        print("=" * 60)

        return evaluation_df

    def generate_forecast(
        self,
        periods: int = 4,
        forecast_method: str = "auto",
        seasonal_periods: Optional[int] = None,
        seasonal: Optional[bool] = None,
    ):
        """
        生成未来预测

        参数:
            periods: 预测周期数
            forecast_method: 预测方法，可选 {"auto", "trend", "ets", "seasonal_ets"}
            seasonal_periods: 季节周期（如日频7=周季节性，月频12=年季节性）
            seasonal: 是否启用季节性（当 forecast_method="ets" 时可用于启用/关闭季节项）
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        if periods < 1:
            raise ValueError("periods 必须 >= 1")

        y = self.data[self.demand_col].astype(float)
        if len(y) < 2:
            raise ValueError("数据量过少，无法生成预测")

        def _future_dates():
            last_date = self.data.index[-1]
            try:
                idx = pd.date_range(start=last_date, periods=periods + 1, freq=self.freq)
                return idx[1:]
            except Exception:
                return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

        def _default_seasonal_periods():
            if self.freq == "D":
                return 7
            if self.freq == "W":
                return 52
            if self.freq == "M":
                return 12
            if self.freq == "Q":
                return 4
            return None

        method = (forecast_method or "auto").lower().strip()
        if method in {"auto", "best"}:
            if isinstance(self.best_model, str) and (self.advanced_models or {}).get(self.best_model) is not None:
                method = self.best_model
            if self.best_model in {"seasonal_ets", "ets_holt_winters"}:
                method = "seasonal_ets"
            elif self.best_model in {"ets", "exponential_smoothing", "ets_simple", "ets_holt"} or (
                isinstance(self.best_model, str) and self.best_model.startswith("ets_")
            ):
                method = "ets"
            elif self.best_model in {"arima"}:
                method = "arima"
            elif self.best_model in {"naive", "seasonal_naive", "moving_average"}:
                method = str(self.best_model)
            else:
                method = "trend"

        if method in {"llm", "deepseek_llm", "llm_deepseek"}:
            return self.generate_forecast_with_llm(periods=int(periods))

        if method == "seasonal_ets":
            seasonal = True if seasonal is None else bool(seasonal)
        elif method == "ets":
            seasonal = seasonal if seasonal is None else bool(seasonal)
        else:
            seasonal = False

        if seasonal_periods is None and seasonal:
            try:
                seas = getattr(self, "ts_analysis_results", {}) or {}
                seas = seas.get("seasonality") or {}
                if seas.get("has_seasonality") and seas.get("suggested_periods"):
                    seasonal_periods = int(seas.get("suggested_periods")[0])
            except Exception:
                seasonal_periods = None
            if seasonal_periods is None:
                seasonal_periods = _default_seasonal_periods()

        if method in (self.advanced_models or {}):
            if getattr(self, "feature_data", None) is None or self.feature_data.get("X") is None:
                raise ValueError("高级模型未来预测需要特征矩阵，请先准备数据并训练高级模型")

            model_obj = self.advanced_models.get(method, {}).get("model")
            if model_obj is None:
                raise ValueError(f"高级模型 {method} 不可用")

            try:
                X_full = self.feature_data.get("X")
                y_full_feat = self.feature_data.get("y")
                if X_full is not None and y_full_feat is not None and len(X_full) == len(y_full_feat):
                    model_obj.fit(X_full, y_full_feat)
            except Exception:
                pass

            X_cols = list(self.feature_data["X"].columns)
            lag_cols = [c for c in X_cols if isinstance(c, str) and c.startswith("lag_")]
            last_known_features = {}
            for c in X_cols:
                if isinstance(c, str) and c.startswith("lag_"):
                    continue
                try:
                    if getattr(self, "covariates", None) and c in self.covariates:
                        raw_val = self.data[c].iloc[-1]
                    else:
                        raw_val = self.data[c].iloc[-1] if c in self.data.columns else 0.0
                        
                    if isinstance(raw_val, pd.Series):
                        raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                    last_known_features[c] = float(raw_val)
                except Exception:
                    last_known_features[c] = 0.0

            n_lags = 0
            if lag_cols:
                try:
                    n_lags = max(int(str(c).split("_", 1)[1]) for c in lag_cols)
                except Exception:
                    n_lags = 0

            preds = []
            if n_lags <= 0:
                try:
                    X_last = self.data[X_cols].iloc[[-1]]
                    y_next = float(model_obj.predict(X_last)[0])
                    preds = [y_next for _ in range(int(periods))]
                except Exception:
                    raise ValueError("高级模型未来预测失败：缺少可用的滞后特征，且无法使用最后一行特征直接预测")
            else:
                if len(y) < n_lags + 1:
                    raise ValueError("数据量不足以进行高级模型未来预测")

                history_vals = [float(v) for v in y.iloc[-n_lags:].values]
                current_date = pd.to_datetime(y.index[-1])
                
                for _ in range(int(periods)):
                    if self.freq == "D":
                        current_date += pd.Timedelta(days=1)
                    elif self.freq == "W":
                        current_date += pd.Timedelta(weeks=1)
                    elif self.freq == "M":
                        current_date += pd.DateOffset(months=1)
                    elif self.freq == "Q":
                        current_date += pd.DateOffset(months=3)
                    else:
                        current_date += pd.Timedelta(days=1)

                    row = {f"lag_{i}": history_vals[-i] for i in range(1, n_lags + 1)}
                    
                    # 更新时间相关特征
                    if "dayofweek" in X_cols:
                        row["dayofweek"] = float(current_date.dayofweek)
                    if "month" in X_cols:
                        row["month"] = float(current_date.month)
                    if "quarter" in X_cols:
                        row["quarter"] = float(current_date.quarter)
                    if "year" in X_cols:
                        row["year"] = float(current_date.year)
                        
                    for k, v in last_known_features.items():
                        if k not in row:
                            row[k] = v
                        
                    # 保证特征顺序和类型正确
                    ordered_row = {c: float(row.get(c, 0.0)) for c in X_cols}
                    X_row = pd.DataFrame([ordered_row])
                    X_row = X_row[X_cols]
                    
                    y_next = float(model_obj.predict(X_row)[0])
                    preds.append(y_next)
                    history_vals.append(y_next)

            fitted = self.advanced_models.get(method, {}).get("fitted")
            sigma = float(y.std())
            if fitted is not None:
                try:
                    yref = self.train_features.get("y") if hasattr(self, "train_features") and self.train_features is not None else self.train_data["y"]
                    yt = np.asarray(yref.values, dtype=float)
                    yp = np.asarray(fitted, dtype=float)
                    n = min(len(yt), len(yp))
                    resid = yt[:n] - yp[:n]
                    resid = resid[np.isfinite(resid)]
                    if resid.size > 1:
                        sigma = float(np.std(resid))
                except Exception:
                    pass

            z = 1.96
            lower = [p - z * sigma for p in preds]
            upper = [p + z * sigma for p in preds]

            res = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": preds,
                "lower_bound": lower,
                "upper_bound": upper,
                "model": method,
                "params": {"type": "advanced", "model_params": self.advanced_models.get(method, {}).get("params")},
            }
            self.forecast_results = res
            return res

        if method in {"naive", "seasonal_naive", "moving_average"}:
            last_date = self.data.index[-1]
            future_dates = [d.to_pydatetime() for d in _future_dates()]
            if method == "naive":
                last_value = float(y.iloc[-1])
                fc = [last_value] * periods
                sigma = float(y.std())
                try:
                    yt = self.train_data["y"].astype(float) if self.train_data and self.train_data.get("y") is not None else None
                    fit = (self.baseline_models or {}).get("naive", {}).get("fitted")
                    if yt is not None and fit is not None:
                        fit_arr = np.asarray(fit, dtype=float)
                        n = min(len(yt), len(fit_arr))
                        resid = np.asarray(yt.values[:n], dtype=float) - fit_arr[:n]
                        resid = resid[np.isfinite(resid)]
                        if resid.size > 1:
                            sigma = float(np.std(resid))
                except Exception:
                    pass
                lower = [v - 1.96 * sigma for v in fc]
                upper = [v + 1.96 * sigma for v in fc]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": fc,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "naive",
                    "params": {"strategy": "last_observation"},
                }
                return self.forecast_results

            if method == "seasonal_naive":
                sp = seasonal_periods or _default_seasonal_periods() or 1
                sp = int(max(sp, 1))
                vals = []
                for i in range(periods):
                    vals.append(float(y.iloc[-sp + (i % sp)]))
                sigma = float(y.std())
                try:
                    yt = self.train_data["y"].astype(float) if self.train_data and self.train_data.get("y") is not None else None
                    fit = (self.baseline_models or {}).get("seasonal_naive", {}).get("fitted")
                    if yt is not None and fit is not None:
                        fit_arr = np.asarray(fit, dtype=float)
                        n = min(len(yt), len(fit_arr))
                        resid = np.asarray(yt.values[:n], dtype=float) - fit_arr[:n]
                        resid = resid[np.isfinite(resid)]
                        if resid.size > 1:
                            sigma = float(np.std(resid))
                except Exception:
                    pass
                lower = [v - 1.96 * sigma for v in vals]
                upper = [v + 1.96 * sigma for v in vals]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": vals,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "seasonal_naive",
                    "params": {"seasonal_period": sp},
                }
                return self.forecast_results

            if method == "moving_average":
                window = None
                try:
                    window = int(self.baseline_models.get("moving_average", {}).get("params", {}).get("window"))
                except Exception:
                    window = None
                if not window:
                    window = int(min(14, max(3, len(y) // 10)))
                last_ma = float(y.rolling(window=window).mean().iloc[-1])
                fc = [last_ma] * periods
                sigma = float(y.std())
                try:
                    yt = self.train_data["y"].astype(float) if self.train_data and self.train_data.get("y") is not None else None
                    fit = (self.baseline_models or {}).get("moving_average", {}).get("fitted")
                    if yt is not None and fit is not None:
                        fit_arr = np.asarray(fit, dtype=float)
                        n = min(len(yt), len(fit_arr))
                        resid = np.asarray(yt.values[:n], dtype=float) - fit_arr[:n]
                        resid = resid[np.isfinite(resid)]
                        if resid.size > 1:
                            sigma = float(np.std(resid))
                except Exception:
                    pass
                lower = [v - 1.96 * sigma for v in fc]
                upper = [v + 1.96 * sigma for v in fc]
                self.forecast_results = {
                    "dates": future_dates,
                    "forecast": fc,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "model": "moving_average",
                    "params": {"window": window},
                }
                return self.forecast_results

        if method in {"arima"}:
            order = None
            seasonal_order = None
            try:
                arima_params = self.baseline_models.get("arima", {}).get("params", {}) or {}
                order = tuple(arima_params.get("order")) if arima_params.get("order") is not None else None
                seasonal_order = tuple(arima_params.get("seasonal_order")) if arima_params.get("seasonal_order") is not None else None
            except Exception:
                order = None
                seasonal_order = None
            if order is None:
                order = (1, 1, 1)
            if seasonal_order is None:
                seasonal_order = (0, 0, 0, 0)
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(periods).astype(float)
            resid = (y - fitted.fittedvalues).dropna()
            sigma = float(resid.std()) if len(resid) > 1 else float(y.std())
            z = 1.96
            lower = forecast - z * sigma
            upper = forecast + z * sigma
            self.forecast_results = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": forecast.tolist(),
                "lower_bound": lower.tolist(),
                "upper_bound": upper.tolist(),
                "model": "arima",
                "params": {"order": order, "seasonal_order": seasonal_order, "aic": float(getattr(fitted, "aic", np.nan))},
            }
            return self.forecast_results

        if method in {"ets", "seasonal_ets"}:
            variant = None
            try:
                if self.evaluation_results is not None and len(self.evaluation_results) > 0 and "model" in self.evaluation_results.columns:
                    candidates = ["ets_simple", "ets_holt", "ets_holt_winters", "seasonal_ets", "ets"]
                    dfv = self.evaluation_results[self.evaluation_results["model"].isin(candidates)]
                    if len(dfv) > 0 and "MAPE" in dfv.columns:
                        variant = str(dfv.sort_values("MAPE").iloc[0]["model"])
            except Exception:
                variant = None

            if seasonal and (seasonal_periods is None or seasonal_periods < 2):
                seasonal = False
                seasonal_periods = None

            if seasonal and len(y) < 2 * seasonal_periods:
                seasonal = False
                seasonal_periods = None

            if method == "seasonal_ets":
                seasonal = True if seasonal is None else bool(seasonal)
            elif seasonal is None:
                seasonal = variant in {"ets_holt_winters", "seasonal_ets"}

            if seasonal and (seasonal_periods is None or seasonal_periods < 2):
                seasonal_periods = _default_seasonal_periods()

            if seasonal and seasonal_periods is not None and len(y) < 2 * int(seasonal_periods):
                seasonal = False
                seasonal_periods = None

            if seasonal:
                print(f"\n使用ETS(季节性)生成 {periods} 期预测，seasonal_periods={seasonal_periods}")
                model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
            else:
                trend = "add" if variant in {"ets_holt"} else None
                print(f"\n使用ETS(非季节性)生成 {periods} 期预测")
                model = ExponentialSmoothing(y, trend=trend, seasonal=None)

            fitted = model.fit()
            forecast = fitted.forecast(periods).astype(float)
            resid = (y - fitted.fittedvalues).dropna()
            sigma = float(resid.std()) if len(resid) > 1 else float(y.std())
            z = 1.96
            lower = forecast - z * sigma
            upper = forecast + z * sigma

            self.forecast_results = {
                "dates": [d.to_pydatetime() for d in _future_dates()],
                "forecast": forecast.tolist(),
                "lower_bound": lower.tolist(),
                "upper_bound": upper.tolist(),
                "model": "ets",
                "params": {"variant": variant, "seasonal": seasonal, "seasonal_periods": seasonal_periods, "trend": "add" if seasonal or variant in {"ets_holt"} else None},
            }
            print(f"未来 {periods} 期预测生成完成")
            return self.forecast_results

        if self.best_model is None:
            print("警告: 没有最佳模型，请先运行模型评估")

        if method not in {"trend"}:
            method = "trend"

        model_label = self.best_model if self.best_model is not None else "trend"
        print(f"\n使用趋势外推生成 {periods} 期预测 (参考模型: {model_label})")

        trend = (y.iloc[-1] - y.iloc[0]) / len(y)
        last_value = y.iloc[-1]
        future_forecast = [float(last_value + trend * i) for i in range(1, periods + 1)]

        std_dev = float(y.std())
        lower_bound = [f - 1.96 * std_dev for f in future_forecast]
        upper_bound = [f + 1.96 * std_dev for f in future_forecast]

        self.forecast_results = {
            "dates": [d.to_pydatetime() for d in _future_dates()],
            "forecast": future_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": model_label,
            "params": {"method": "trend"},
        }

        print(f"未来 {periods} 期预测生成完成")
        return self.forecast_results

    def generate_report(self, save_to_disk: bool = True, use_llm: bool = False):
        """
        Step 5: 输出管理报告

        生成一份非技术经理也能读懂的报告，包含:
        1. 预测图表: 包含历史需求、拟合值及未来至少4个周期的预测值（含95%置信区间）
        2. 对比表: 按测试集MAPE（或RMSE）对模型进行排名，高亮推荐模型
        3. 通俗总结: 解释胜出模型的原因，并描述未来的需求高峰、趋势和季节性
        4. 行动建议: 给出两项关于库存目标、人员配备或采购计划的具体建议
        """
        print("\n" + "=" * 60)
        print("Step 5: 生成管理报告")
        print("=" * 60)

        if self.evaluation_results is None:
            print("警告: 请先运行模型评估")
            return

        if self.forecast_results is None:
            self.generate_forecast(periods=4)

        import os

        is_en = str(getattr(self, "language", "en")).lower().startswith("en")

        def _t(zh_str: str, en_str: str) -> str:
            return en_str if is_en else zh_str

        try:
            dataset_name = os.path.basename(str(self.data_path or "")).strip() or _t("数据集", "Dataset")
        except Exception:
            dataset_name = _t("数据集", "Dataset")

        try:
            override = getattr(self, "dataset_name_override", None)
            if override and str(override).strip():
                dataset_name = str(override).strip()
        except Exception:
            pass

        dataset_theme = None
        data_description = None
        try:
            dataset_theme = getattr(self, "dataset_theme", None)
        except Exception:
            dataset_theme = None
        try:
            data_description = getattr(self, "data_description", None)
        except Exception:
            data_description = None

        best_metrics = None
        try:
            best_metrics = self.evaluation_results[self.evaluation_results['model'] == self.best_model].to_dict('records')[0]
        except Exception:
            best_metrics = None

        start_date = None
        end_date = None
        try:
            if self.data is not None and len(self.data) > 0:
                start_date = pd.to_datetime(self.data.index.min())
                end_date = pd.to_datetime(self.data.index.max())
        except Exception:
            start_date = None
            end_date = None

        split_date = None
        try:
            if self.test_data and self.test_data.get("y") is not None and len(self.test_data["y"]) > 0:
                split_date = pd.to_datetime(self.test_data["y"].index[0])
            elif self.train_data and self.train_data.get("y") is not None and len(self.train_data["y"]) > 0:
                split_date = pd.to_datetime(self.train_data["y"].index[-1])
        except Exception:
            split_date = None

        freq_map = {"D": _t("每日", "Daily"), "W": _t("每周", "Weekly"), "M": _t("每月", "Monthly"), "Q": _t("每季度", "Quarterly"), "Y": _t("每年", "Yearly")}
        freq_label = freq_map.get(str(self.freq or "").upper(), str(self.freq))

        missing_summary = {}
        key_stats = {}
        trend_summary = {}
        seasonality_summary = {}
        try:
            if self.data is not None and self.demand_col in self.data.columns:
                y_series = self.data[self.demand_col].astype(float)
                missing_summary = {
                    "missing_values_total": int(self.data.isna().sum().sum()),
                    "missing_values_target": int(y_series.isna().sum()),
                    "missing_values_by_column": {k: int(v) for k, v in self.data.isna().sum().to_dict().items()},
                }
                key_stats = {
                    "mean": float(y_series.mean()),
                    "std": float(y_series.std()),
                    "min": float(y_series.min()),
                    "max": float(y_series.max()),
                }
                if len(y_series) > 1:
                    avg_change = float(y_series.diff().dropna().mean())
                    avg_level = float(y_series.mean()) if float(y_series.mean()) != 0 else None
                    trend_summary = {
                        "avg_change_per_period": avg_change,
                        "avg_change_pct_of_mean": (avg_change / avg_level * 100.0) if avg_level is not None else None,
                    }
        except Exception:
            missing_summary = {}
            key_stats = {}
            trend_summary = {}

        try:
            tsr = getattr(self, "ts_analysis_results", None) or {}
            seas = tsr.get("seasonality") or {}
            decomp = tsr.get("decomposition") or {}
            seasonality_summary = {
                "has_seasonality": bool(seas.get("has_seasonality")) if seas else None,
                "suggested_periods": seas.get("suggested_periods") if seas else None,
                "seasonal_strength": decomp.get("seasonal_ratio") if decomp else None,
                "trend_strength": decomp.get("trend_ratio") if decomp else None,
            }
        except Exception:
            seasonality_summary = {}

        y_full = self.data[self.demand_col].astype(float) if self.data is not None else None
        tail_n = len(y_full) if y_full is not None else 0
        y_tail = y_full.iloc[-tail_n:] if y_full is not None and len(y_full) > 0 else None

        fitted_tail = None
        try:
            fitted = None
            if isinstance(self.best_model, str):
                if (self.baseline_models or {}).get(self.best_model) is not None:
                    fitted = (self.baseline_models or {}).get(self.best_model, {}).get("fitted")
                    idx = self.train_data["y"].index if self.train_data and self.train_data.get("y") is not None else None
                elif (self.advanced_models or {}).get(self.best_model) is not None:
                    fitted = (self.advanced_models or {}).get(self.best_model, {}).get("fitted")
                    idx = self.train_features.get("y").index if hasattr(self, "train_features") and self.train_features.get("y") is not None else None
                else:
                    idx = None

                if fitted is not None and idx is not None and y_tail is not None:
                    fit_map = {}
                    fitted_list = list(fitted)
                    n = min(len(idx), len(fitted_list))
                    for i in range(n):
                        try:
                            v = float(fitted_list[i])
                            fit_map[idx[i]] = v if pd.notna(v) and float("inf") != abs(v) else None
                        except Exception:
                            fit_map[idx[i]] = None
                    fitted_tail = [fit_map.get(d, None) for d in y_tail.index]
        except Exception:
            fitted_tail = None

        chart_data = None
        try:
            if y_tail is not None:
                test_pred = None
                test_idx = None
                try:
                    if isinstance(self.best_model, str):
                        if (self.baseline_models or {}).get(self.best_model) is not None:
                            test_pred = (self.baseline_models or {}).get(self.best_model, {}).get("predictions")
                            test_idx = self.test_data["y"].index if self.test_data and self.test_data.get("y") is not None else None
                        elif (self.advanced_models or {}).get(self.best_model) is not None:
                            test_pred = (self.advanced_models or {}).get(self.best_model, {}).get("predictions")
                            if hasattr(self, "test_features") and self.test_features is not None and self.test_features.get("y") is not None:
                                test_idx = self.test_features.get("y").index
                            else:
                                test_idx = self.test_data["y"].index if self.test_data and self.test_data.get("y") is not None else None
                except Exception:
                    test_pred = None
                    test_idx = None

                test_pred_aligned = None
                test_actual_aligned = None
                test_dates = None
                try:
                    if test_pred is not None and test_idx is not None and self.test_data and self.test_data.get("y") is not None:
                        preds_list = list(np.asarray(test_pred, dtype=float))
                        idx_list = list(pd.to_datetime(test_idx))
                        actual_idx = list(pd.to_datetime(self.test_data["y"].index))
                        actual_vals = list(np.asarray(self.test_data["y"].astype(float).values, dtype=float))

                        pred_map = {}
                        n = min(len(idx_list), len(preds_list))
                        for i in range(n):
                            pred_map[idx_list[i]] = float(preds_list[i])

                        actual_map = {}
                        for i in range(len(actual_idx)):
                            actual_map[actual_idx[i]] = float(actual_vals[i])

                        test_dates = actual_idx
                        test_actual_aligned = [actual_map.get(d, None) for d in test_dates]
                        test_pred_aligned = [pred_map.get(d, None) for d in test_dates]
                except Exception:
                    test_pred_aligned = None
                    test_actual_aligned = None
                    test_dates = None

                combined_dates = []
                combined_forecast = []
                combined_lower = []
                combined_upper = []
                try:
                    future = self.forecast_results or {}
                    fd = list(pd.to_datetime(future.get("dates") or []))
                    fc = list(np.asarray(future.get("forecast") or [], dtype=float))
                    lo = list(np.asarray(future.get("lower_bound") or [], dtype=float)) if future.get("lower_bound") is not None else []
                    up = list(np.asarray(future.get("upper_bound") or [], dtype=float)) if future.get("upper_bound") is not None else []

                    if test_dates and test_pred_aligned:
                        combined_dates = list(test_dates) + list(fd)
                        combined_forecast = list(test_pred_aligned) + list(fc)
                        combined_lower = [None] * len(test_dates) + (list(lo) if lo else [None] * len(fc))
                        combined_upper = [None] * len(test_dates) + (list(up) if up else [None] * len(fc))
                    else:
                        combined_dates = list(fd)
                        combined_forecast = list(fc)
                        combined_lower = list(lo) if lo else [None] * len(fc)
                        combined_upper = list(up) if up else [None] * len(fc)
                except Exception:
                    combined_dates = []
                    combined_forecast = []
                    combined_lower = []
                    combined_upper = []

                chart_data = {
                    "history": {
                        "dates": [d.to_pydatetime() for d in y_tail.index],
                        "actual": [float(v) for v in y_tail.values],
                        "fitted_best": fitted_tail,
                    },
                    "split_date": split_date.to_pydatetime() if split_date is not None else None,
                    "test": {
                        "dates": [d.to_pydatetime() for d in (test_dates or [])],
                        "actual": test_actual_aligned,
                        "predicted_best": test_pred_aligned,
                    },
                    "combined_forecast": {
                        "dates": [d.to_pydatetime() for d in combined_dates],
                        "forecast": combined_forecast,
                        "lower_bound": combined_lower,
                        "upper_bound": combined_upper,
                    },
                    "future": self.forecast_results,
                }
        except Exception:
            chart_data = None

        plotly_payload = None
        try:
            import plotly.graph_objects as go

            if chart_data is not None and chart_data.get("history") and chart_data.get("future"):
                hd = chart_data["history"]["dates"]
                ha = chart_data["history"]["actual"]
                hf = chart_data["history"].get("fitted_best") or []
                cd = chart_data.get("combined_forecast", {}).get("dates") or []
                cf = chart_data.get("combined_forecast", {}).get("forecast") or []
                clo = chart_data.get("combined_forecast", {}).get("lower_bound") or []
                cup = chart_data.get("combined_forecast", {}).get("upper_bound") or []
                split_dt = chart_data.get("split_date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hd, y=ha, mode="lines", name="历史需求", line=dict(color="#0B1F3B", width=2)))
                if hf:
                    fig.add_trace(
                        go.Scatter(
                            x=hd,
                            y=hf,
                            mode="lines",
                            name="拟合值（获胜模型）",
                            line=dict(color="#FF8C00", width=2, dash="dash"),
                        )
                    )
                if cd and cf:
                    fig.add_trace(go.Scatter(x=cd, y=cf, mode="lines", name="预测值（测试期+未来）", line=dict(color="#2E8B57", width=3)))
                if cd and clo and cup:
                    fig.add_trace(go.Scatter(x=cd, y=cup, mode="lines", line=dict(width=0), showlegend=False, name="95%上限"))
                    fig.add_trace(
                        go.Scatter(
                            x=cd,
                            y=clo,
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(120,120,120,0.25)",
                            line=dict(width=0),
                            showlegend=True,
                            name="95%预测区间",
                        )
                    )

                try:
                    if split_dt is not None and hd and ha:
                        y_min = float(np.nanmin(np.asarray([x for x in ha if x is not None], dtype=float)))
                        y_max = float(np.nanmax(np.asarray([x for x in ha if x is not None], dtype=float)))
                        fig.add_trace(
                            go.Scatter(
                                x=[split_dt, split_dt],
                                y=[y_min, y_max],
                                mode="lines",
                                name="训练/测试分割",
                                line=dict(color="rgba(0,0,0,0.35)", width=2, dash="dash"),
                            )
                        )
                except Exception:
                    pass

                fig.update_layout(
                    title=f"需求预测 — {self.best_model} — {dataset_name}",
                    xaxis_title="时间周期",
                    yaxis_title=str(self.demand_col),
                    legend=dict(orientation="h"),
                    margin=dict(l=40, r=20, t=50, b=40),
                )

                code = (
                    "import plotly.graph_objects as go\n"
                    "from datetime import datetime\n\n"
                    "hd = report['chart_data']['history']['dates']\n"
                    "ha = report['chart_data']['history']['actual']\n"
                    "hf = report['chart_data']['history'].get('fitted_best')\n"
                    "cd = report['chart_data']['combined_forecast']['dates']\n"
                    "cf = report['chart_data']['combined_forecast']['forecast']\n"
                    "clo = report['chart_data']['combined_forecast']['lower_bound']\n"
                    "cup = report['chart_data']['combined_forecast']['upper_bound']\n"
                    "split_dt = report['chart_data'].get('split_date')\n\n"
                    "fig = go.Figure()\n"
                    "fig.add_trace(go.Scatter(x=hd, y=ha, mode='lines', name='历史需求'))\n"
                    "if hf:\n"
                    "    fig.add_trace(go.Scatter(x=hd, y=hf, mode='lines', name='拟合值（获胜模型）', line=dict(dash='dash')))\n"
                    "fig.add_trace(go.Scatter(x=cd, y=cf, mode='lines', name='预测值（测试期+未来）'))\n"
                    "fig.add_trace(go.Scatter(x=cd, y=cup, mode='lines', line=dict(width=0), showlegend=False))\n"
                    "fig.add_trace(go.Scatter(x=cd, y=clo, mode='lines', fill='tonexty', name='95%预测区间', line=dict(width=0)))\n"
                    "if split_dt:\n"
                    "    fig.add_vline(x=split_dt, line_dash='dash')\n"
                    "fig.show()\n"
                )
                plotly_payload = {"figure": fig.to_dict(), "code": code}
        except Exception:
            plotly_payload = None

        # 生成报告内容
        uncertainty_flags = []
        try:
            n_obs = int(len(self.data)) if self.data is not None else 0
            if n_obs < 24:
                uncertainty_flags.append("数据集观测值少于24个，季节性相关结论可能不可靠。")
        except Exception:
            pass
        try:
            if best_metrics and best_metrics.get("MAPE") is not None and float(best_metrics.get("MAPE")) > 30:
                uncertainty_flags.append("MAPE > 30%，预测不确定性较高，建议保守规划并尽快用新数据复核。")
        except Exception:
            pass

        forecast_interpretation = {}
        actionable_recommendations = []
        try:
            fr = self.forecast_results or {}
            fd = [pd.to_datetime(d) for d in (fr.get("dates") or [])]
            fc = [float(v) for v in (fr.get("forecast") or [])]
            lo = [float(v) for v in (fr.get("lower_bound") or [])] if fr.get("lower_bound") is not None else [None] * len(fc)
            up = [float(v) for v in (fr.get("upper_bound") or [])] if fr.get("upper_bound") is not None else [None] * len(fc)

            if fd and fc:
                peak_i = int(np.nanargmax(np.asarray(fc, dtype=float)))
                low_i = int(np.nanargmin(np.asarray(fc, dtype=float)))
                peak_date = fd[peak_i]
                low_date = fd[low_i]
                peak_val = float(fc[peak_i])
                low_val = float(fc[low_i])
                peak_up = float(up[peak_i]) if up and up[peak_i] is not None else None
                low_lo = float(lo[low_i]) if lo and lo[low_i] is not None else None
                avg_fc = float(np.mean(np.asarray(fc, dtype=float)))
                interval_width = None
                try:
                    widths = []
                    for i in range(len(fc)):
                        if lo and up and lo[i] is not None and up[i] is not None:
                            widths.append(float(up[i]) - float(lo[i]))
                    interval_width = float(np.mean(widths)) if widths else None
                except Exception:
                    interval_width = None

                last_actual = None
                try:
                    if y_full is not None and len(y_full) > 0:
                        last_actual = float(y_full.iloc[-1])
                except Exception:
                    last_actual = None

                delta = None
                delta_pct = None
                try:
                    if last_actual is not None:
                        delta = avg_fc - last_actual
                        delta_pct = (delta / last_actual * 100.0) if last_actual != 0 else None
                except Exception:
                    delta = None
                    delta_pct = None

                forecast_interpretation = {
                    "avg_forecast": avg_fc,
                    "avg_vs_last_delta": delta,
                    "avg_vs_last_delta_pct": delta_pct,
                    "peak": {"date": peak_date.to_pydatetime(), "forecast": peak_val, "upper_95": peak_up},
                    "trough": {"date": low_date.to_pydatetime(), "forecast": low_val, "lower_95": low_lo},
                    "avg_interval_width": interval_width,
                }

                if peak_up is None:
                    peak_up = peak_val
                if low_lo is None:
                    low_lo = low_val

                actionable_recommendations = self._actionable_recommendations_from_forecast(forecast_interpretation) or []
        except Exception:
            forecast_interpretation = {}
            actionable_recommendations = []

        model_notes = {}
        try:
            dfc = self.evaluation_results.copy()
            primary_metric = "MAPE" if ("MAPE" in dfc.columns and dfc["MAPE"].notna().any()) else ("RMSE" if "RMSE" in dfc.columns else None)
            if primary_metric is not None:
                dfc = dfc.sort_values(primary_metric)
            best_score = None
            try:
                if len(dfc) > 0 and primary_metric is not None and dfc.iloc[0].get(primary_metric) is not None:
                    best_score = float(dfc.iloc[0][primary_metric])
            except Exception:
                best_score = None
            for _, row in dfc.iterrows():
                name = str(row.get("model"))
                score = row.get(primary_metric) if primary_metric is not None else None
                reasons = []
                try:
                    if row.get("overfitting_risk") == "高":
                        reasons.append("过拟合风险较高")
                except Exception:
                    pass
                try:
                    if row.get("resid_white_noise") is not None and not bool(row.get("resid_white_noise")):
                        reasons.append("残差可能仍包含结构性信息")
                except Exception:
                    pass
                try:
                    if best_score is not None and score is not None and primary_metric is not None:
                        gap = float(score) - float(best_score)
                        if gap > 0:
                            suffix = "%" if primary_metric == "MAPE" else ""
                            reasons.append(f"相对最佳模型误差更高（{primary_metric}差约 {gap:.4f}{suffix}）")
                except Exception:
                    pass
                if not reasons and name != str(self.best_model):
                    reasons.append("对该数据的趋势/波动刻画不足")
                if reasons:
                    model_notes[name] = "；".join(reasons)
        except Exception:
            model_notes = {}

        chart_table = []
        try:
            cd = chart_data.get("combined_forecast", {}).get("dates") if chart_data else None
            cf = chart_data.get("combined_forecast", {}).get("forecast") if chart_data else None
            clo = chart_data.get("combined_forecast", {}).get("lower_bound") if chart_data else None
            cup = chart_data.get("combined_forecast", {}).get("upper_bound") if chart_data else None
            test_dates = chart_data.get("test", {}).get("dates") if chart_data else None
            test_actual = chart_data.get("test", {}).get("actual") if chart_data else None
            if cd and cf:
                actual_map = {}
                if test_dates and test_actual:
                    for i in range(min(len(test_dates), len(test_actual))):
                        actual_map[pd.to_datetime(test_dates[i])] = test_actual[i]
                for i in range(min(len(cd), len(cf))):
                    dt = pd.to_datetime(cd[i])
                    chart_table.append(
                        {
                            "period": dt.to_pydatetime(),
                            "history_actual": actual_map.get(dt, None),
                            "fitted": None,
                            "forecast": cf[i],
                            "lower_95": clo[i] if clo and i < len(clo) else None,
                            "upper_95": cup[i] if cup and i < len(cup) else None,
                        }
                    )
        except Exception:
            chart_table = []

        cover = {
            "title": f"需求预测报告 — {dataset_name}",
            "generated_by": "由人工智能预测系统生成",
            "dataset": dataset_name,
            "theme": dataset_theme,
            "description": data_description,
            "time_range": {
                "start": start_date.to_pydatetime() if start_date is not None else None,
                "end": end_date.to_pydatetime() if end_date is not None else None,
                "frequency": freq_label,
            },
        }

        executive_summary = {
            "analyzed_data": f"对数据集 {dataset_name} 的时间序列（目标变量：{self.demand_col}）进行建模与评估。" + (f" 数据主题：{dataset_theme}。" if dataset_theme else "") + (f" 数据描述：{data_description}" if data_description else ""),
            "winning_model": {
                "name": self.best_model,
                "reason": getattr(self, "best_model_selection_reason", None) or "在相同测试集划分下，综合误差指标表现最佳。",
            },
            "top_business_recommendations": [r.get("action") for r in actionable_recommendations[:2]] if actionable_recommendations else [],
        }

        data_overview = {
            "summary": {
                "observations": len(self.data) if self.data is not None else 0,
                "start": start_date.to_pydatetime() if start_date is not None else None,
                "end": end_date.to_pydatetime() if end_date is not None else None,
                "frequency": freq_label,
                "missing": missing_summary,
            },
            "key_statistics": key_stats,
            "trend": trend_summary,
            "seasonality": seasonality_summary,
        }

        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_name': dataset_name,
            'cover': cover,
            'executive_summary': executive_summary,
            'data_overview': data_overview,
            'uncertainty_flags': uncertainty_flags,
            'data_summary': {
                'total_observations': len(self.data) if self.data is not None else 0,
                'training_period': len(self.train_data['y']) if self.train_data else 0,
                'testing_period': len(self.test_data['y']) if self.test_data else 0,
                'frequency': self.freq,
                'demand_variable': self.demand_col
            },
            'data_quality': getattr(self, "preprocessing_summary", None),
            'best_model': {
                'name': self.best_model,
                'metrics': best_metrics,
                'selection_reason': getattr(self, "best_model_selection_reason", None),
            },
            'model_comparison': self.evaluation_results.to_dict('records'),
            'model_comparison_notes': model_notes,
            'model_details': self.model_results if hasattr(self, 'model_results') else None,
            'forecast_summary': self.forecast_results if hasattr(self, 'forecast_results') else None,
            'chart_data': chart_data,
            'chart_table': chart_table,
            'plotly': plotly_payload,
            'ts_analysis': self.ts_analysis_results if hasattr(self, 'ts_analysis_results') else None,
            'model_selection': {
                'summary': self._generate_model_selection_summary(),
                'risks': self._generate_risk_flags(),
            },
            'insights': self._generate_insights(),
            'recommendations': self._generate_recommendations(),
            'actionable_recommendations': actionable_recommendations,
            'forecast_interpretation': forecast_interpretation,
            'future_by_method': getattr(self, 'future_by_method', {})
        }

        if bool(use_llm):
            try:
                llm_sections = self._llm_generate_report_sections(self._clean_for_json(report))
                report["llm_sections"] = llm_sections
                if llm_sections.get("executive_summary_text"):
                    report["executive_summary"]["llm_text"] = llm_sections.get("executive_summary_text")
                if llm_sections.get("section5_plain_text"):
                    report["forecast_interpretation"]["llm_text"] = llm_sections.get("section5_plain_text")
                if llm_sections.get("actionable_recommendations"):
                    report["actionable_recommendations"] = llm_sections.get("actionable_recommendations")
                if llm_sections.get("model_comparison_commentary"):
                    report["model_comparison_notes"] = llm_sections.get("model_comparison_commentary")
            except Exception as e:
                report["llm_sections_error"] = str(e)

        def _fmt_dt(x):
            try:
                if x is None:
                    return "—"
                return pd.to_datetime(x).strftime("%Y-%m-%d")
            except Exception:
                return str(x)

        def _fmt_num(x, nd=3, pct=False):
            try:
                if x is None:
                    return "—"
                v = float(x)
                s = f"{v:.{int(nd)}f}"
                return f"{s}%" if pct else s
            except Exception:
                return "—"

        def _build_pages(rep: Dict[str, Any]) -> List[Dict[str, Any]]:
            cover_ = rep.get("cover") or {}
            ov = rep.get("data_overview") or {}
            summ = (ov.get("summary") or {})
            ks = (ov.get("key_statistics") or {})
            trend = (ov.get("trend") or {})
            seas = (ov.get("seasonality") or {})
            exec_ = rep.get("executive_summary") or {}
            best_ = (rep.get("best_model") or {})
            best_name = str(best_.get("name") or "—")
            best_reason = str(best_.get("selection_reason") or (exec_.get("winning_model") or {}).get("reason") or "—")
            chart = rep.get("chart_data") or {}
            interp = rep.get("forecast_interpretation") or {}
            recs = rep.get("actionable_recommendations") or []
            comp = rep.get("model_comparison") or []
            details = rep.get("model_details") or {}
            flags_ = rep.get("uncertainty_flags") or []

            comp_sorted = list(comp) if isinstance(comp, list) else []
            try:
                comp_sorted.sort(key=lambda r: float(r.get("MAPE")) if r is not None and r.get("MAPE") is not None else float("inf"))
            except Exception:
                try:
                    comp_sorted.sort(key=lambda r: float(r.get("RMSE")) if r is not None and r.get("RMSE") is not None else float("inf"))
                except Exception:
                    pass

            def _metric_rows(rows):
                out = []
                for r in (rows or []):
                    if not isinstance(r, dict):
                        continue
                    out.append(
                        {
                            "模型": str(r.get("model") or ""),
                            "类型": str(r.get("type") or ""),
                            "MAE(训练)": r.get("MAE_train") if r.get("MAE_train") is not None else r.get("in_sample_MAE"),
                            "RMSE(训练)": r.get("RMSE_train") if r.get("RMSE_train") is not None else r.get("in_sample_RMSE"),
                            "MAPE(训练)": r.get("MAPE_train") if r.get("MAPE_train") is not None else r.get("in_sample_MAPE"),
                            "MAE(测试)": r.get("MAE_test") if r.get("MAE_test") is not None else r.get("MAE"),
                            "RMSE(测试)": r.get("RMSE_test") if r.get("RMSE_test") is not None else r.get("RMSE"),
                            "MAPE(测试)": r.get("MAPE_test") if r.get("MAPE_test") is not None else r.get("MAPE"),
                            "过拟合风险": str(r.get("overfitting_risk") or ""),
                            "残差白噪声": "是" if r.get("resid_white_noise") is True else ("否" if r.get("resid_white_noise") is False else ""),
                        }
                    )
                return out

            metric_table = _metric_rows(comp_sorted)

            def _model_param_text(model_key: str) -> str:
                d = details.get(model_key) if isinstance(details, dict) else None
                if not isinstance(d, dict):
                    return "参数：—"
                p = d.get("params")
                if p is None:
                    return "参数：—"
                try:
                    s = json.dumps(p, ensure_ascii=False)
                    if len(s) > 240:
                        s = s[:240] + "…"
                    return f"参数：{s}"
                except Exception:
                    return f"参数：{str(p)}"

            classical_order = ["naive", "seasonal_naive", "moving_average", "ets", "arima", "sarima"]
            classical_models = []
            for mk in classical_order:
                hit = None
                for r in comp_sorted:
                    if isinstance(r, dict) and str(r.get("model") or "") == mk:
                        hit = r
                        break
                if hit is None:
                    continue
                classical_models.append(
                    {
                        "模型": mk,
                        "MAE(训练)": hit.get("MAE_train") if hit.get("MAE_train") is not None else hit.get("in_sample_MAE"),
                        "RMSE(训练)": hit.get("RMSE_train") if hit.get("RMSE_train") is not None else hit.get("in_sample_RMSE"),
                        "MAPE(训练)": hit.get("MAPE_train") if hit.get("MAPE_train") is not None else hit.get("in_sample_MAPE"),
                        "MAE(测试)": hit.get("MAE_test") if hit.get("MAE_test") is not None else hit.get("MAE"),
                        "RMSE(测试)": hit.get("RMSE_test") if hit.get("RMSE_test") is not None else hit.get("RMSE"),
                        "MAPE(测试)": hit.get("MAPE_test") if hit.get("MAPE_test") is not None else hit.get("MAPE"),
                        "参数摘要": _model_param_text(mk),
                    }
                )

            advanced_candidates = []
            for r in comp_sorted:
                if isinstance(r, dict) and str(r.get("type") or "") == "advanced":
                    advanced_candidates.append(r)
            advanced_models = []
            for r in advanced_candidates[:3]:
                mk = str(r.get("model") or "")
                advanced_models.append(
                    {
                        "模型": mk,
                        "MAE(训练)": r.get("MAE_train") if r.get("MAE_train") is not None else r.get("in_sample_MAE"),
                        "RMSE(训练)": r.get("RMSE_train") if r.get("RMSE_train") is not None else r.get("in_sample_RMSE"),
                        "MAPE(训练)": r.get("MAPE_train") if r.get("MAPE_train") is not None else r.get("in_sample_MAPE"),
                        "MAE(测试)": r.get("MAE_test") if r.get("MAE_test") is not None else r.get("MAE"),
                        "RMSE(测试)": r.get("RMSE_test") if r.get("RMSE_test") is not None else r.get("RMSE"),
                        "MAPE(测试)": r.get("MAPE_test") if r.get("MAPE_test") is not None else r.get("MAPE"),
                        "过拟合风险": str(r.get("overfitting_risk") or ""),
                        "参数摘要": _model_param_text(mk),
                    }
                )

            chart_table = rep.get("chart_table", [])
            forecast_rows = []
            for row in chart_table[:36]:
                if isinstance(row, dict):
                    forecast_rows.append({
                        "Period" if is_en else "时期": _fmt_dt(row.get("period")),
                        "History" if is_en else "历史数据": _fmt_num(row.get("history_actual")),
                        "Fitted" if is_en else "拟合值": _fmt_num(row.get("fitted")),
                        "Forecast" if is_en else "预测值": _fmt_num(row.get("forecast")),
                        "95% Lower" if is_en else "95% 下限": _fmt_num(row.get("lower_95")),
                        "95% Upper" if is_en else "95% 上限": _fmt_num(row.get("upper_95"))
                    })

            comparison_chart_data = []
            for r in comp_sorted:
                if isinstance(r, dict):
                    comparison_chart_data.append({
                        "label": str(r.get("model") or ""),
                        "value": r.get("MAPE_test") if r.get("MAPE_test") is not None else r.get("MAPE")
                    })

            split_dt = chart.get("split_date")
            title = cover_.get("title") or _t(f"需求预测报告 — {rep.get('dataset_name') or '数据集'}", f"Demand Forecast Report — {rep.get('dataset_name') or 'Dataset'}")
            gen_line = cover_.get("generated_by") or _t("由人工智能预测系统生成", "Generated by AI Forecasting System")

            demand_var = str((rep.get("data_summary") or {}).get("demand_variable") or self.demand_col)
            freq_line = str((cover_.get("time_range") or {}).get("frequency") or (summ.get("frequency") or "—"))
            date_range_line = _t(f"{_fmt_dt(summ.get('start'))} 至 {_fmt_dt(summ.get('end'))}", f"{_fmt_dt(summ.get('start'))} to {_fmt_dt(summ.get('end'))}")

            exec_text = exec_.get("llm_text") or ""
            if not exec_text:
                exec_text = _t(
                    f"本报告对数据集“{rep.get('dataset_name') or '数据集'}”的时间序列需求（目标变量：{demand_var}）进行建模评估，并生成未来预测与运营建议。推荐模型为 {best_name}，原因：{best_reason}",
                    f"This report evaluates time-series demand models for dataset '{rep.get('dataset_name') or 'Dataset'}' (Target: {demand_var}) and generates forecasts & operational recommendations. Recommended model: {best_name}, reason: {best_reason}"
                )

            interp_text = interp.get("llm_text") or ""
            if not interp_text:
                peak = interp.get("peak") or {}
                trough = interp.get("trough") or {}
                interp_text = _t(
                    f"趋势：未来平均预测值约 {_fmt_num(interp.get('avg_forecast'))}，相对最近一期变化约 {_fmt_num(interp.get('avg_vs_last_delta'))}"
                    f"（{_fmt_num(interp.get('avg_vs_last_delta_pct'), nd=2, pct=True)}）。"
                    f"峰值：{_fmt_dt(peak.get('date'))}，点预测 {_fmt_num(peak.get('forecast'))}，95%上限 {_fmt_num(peak.get('upper_95'))}。"
                    f"低谷：{_fmt_dt(trough.get('date'))}，点预测 {_fmt_num(trough.get('forecast'))}，95%下限 {_fmt_num(trough.get('lower_95'))}。"
                    f"不确定性：平均区间宽度约 {_fmt_num(interp.get('avg_interval_width'))}。",
                    
                    f"Trend: Future avg forecast ~{_fmt_num(interp.get('avg_forecast'))}, delta vs last ~{_fmt_num(interp.get('avg_vs_last_delta'))} "
                    f"({_fmt_num(interp.get('avg_vs_last_delta_pct'), nd=2, pct=True)}). "
                    f"Peak: {_fmt_dt(peak.get('date'))}, point forecast {_fmt_num(peak.get('forecast'))}, 95% upper {_fmt_num(peak.get('upper_95'))}. "
                    f"Trough: {_fmt_dt(trough.get('date'))}, point forecast {_fmt_num(trough.get('forecast'))}, 95% lower {_fmt_num(trough.get('lower_95'))}. "
                    f"Uncertainty: Avg interval width ~{_fmt_num(interp.get('avg_interval_width'))}."
                )

            chart_desc = _t(
                "图表包含：历史需求（深蓝色实线）、获胜模型拟合值（橙色虚线）、预测值（绿色实线，覆盖测试期+未来≥4期）、"
                "95%预测区间（灰色阴影）、训练/测试分割线（竖向虚线）。X轴为时间周期，Y轴为需求单位。"
                f"标题：需求预测 — {best_name} — {rep.get('dataset_name') or '数据集'}。",
                
                "Chart includes: Historical Demand (dark blue solid), Best Model Fitted (orange dashed), Forecast (green solid, test + future ≥4 periods), "
                "95% Interval (gray area), Train/Test Split (vertical dashed). X-axis: Time, Y-axis: Demand Units. "
                f"Title: Demand Forecast — {best_name} — {rep.get('dataset_name') or 'Dataset'}."
            )

            pages = [
                {
                    "page_no": 1,
                    "title": _t("封面页", "Cover Page"),
                    "blocks": [
                        {"type": "h1", "text": title},
                        {"type": "p", "text": _t("课程：MGT550 Operations Management", "Course: MGT550 Operations Management")},
                        {"type": "p", "text": _t("项目：AI‑Powered Demand Forecasting System（通用版）", "Project: AI-Powered Demand Forecasting System")},
                        {"type": "p", "text": _t(f"编制日期：{rep.get('report_date')}", f"Date: {rep.get('report_date')}")},
                        {"type": "p", "text": gen_line},
                        {"type": "p", "text": _t(f"数据集：{rep.get('dataset_name') or '数据集'}", f"Dataset: {rep.get('dataset_name') or 'Dataset'}")},
                        {"type": "p", "text": _t(f"时间段：{date_range_line} · 频率：{freq_line}", f"Period: {date_range_line} · Freq: {freq_line}")},
                        {"type": "p", "text": _t(f"目标变量：{demand_var}", f"Target Variable: {demand_var}")},
                    ],
                },
                {
                    "page_no": 2,
                    "title": _t("预测图表（前瞻性预测与区间）", "Forecast Chart (Forward-looking Forecast & Intervals)"),
                    "blocks": [
                        {"type": "p", "text": _t("图表展示：历史需求数据、拟合值以及包含至少未来 4 个时间段的 95% 预测区间前瞻性预测。", "Chart showing: Historical demand data, fitted values, and forward-looking forecasts with 95% prediction intervals for at least 4 future periods.")},
                        {"type": "chart", "title": _t("需求预测可视化", "Demand Forecast Visualization"), "chart_data_key": "chart_data"},
                        {"type": "callout", "title": _t("图表说明", "Chart Legend"), "text": chart_desc},
                    ],
                },
                {
                    "page_no": 3,
                    "title": _t("模型对比表（按误差排序）", "Model Comparison Table (Sorted by Error)"),
                    "blocks": [
                        {"type": "p", "text": _t("下表展示所有拟合模型，并已按保留集 MAPE（或 RMSE）进行排序，突出显示推荐模型。", "The table below shows all fitted models sorted by hold-out MAPE (or RMSE), with the recommended model highlighted.")},
                        {"type": "bar_chart", "title": _t("各模型测试集 MAPE 误差对比图", "Model MAPE Comparison Chart"), "chart_data": comparison_chart_data},
                        {"type": "table", "title": _t("模型对比（训练+测试）", "Model Comparison (Train + Test)"), "rows": metric_table, "highlight_model": best_name},
                        {"type": "callout", "title": _t("推荐模型", "Recommended Model"), "text": f"{best_name} ({best_reason})"},
                    ],
                },
                {
                    "page_no": 4,
                    "title": _t("通俗语言摘要（执行摘要）", "Executive Summary (Plain Language)"),
                    "blocks": [
                        {"type": "p", "text": exec_text},
                        {"type": "callout", "title": _t("业务影响总结", "Business Impact Summary"), "text": interp_text},
                    ],
                },
                {
                    "page_no": 5,
                    "title": _t("可执行运营建议", "Actionable Recommendations"),
                    "blocks": [
                        {"type": "p", "text": _t("基于预测数据，提供两项直接源自预测的可执行建议（例如库存目标、人员配置水平、再订购时机或采购计划安排）。", "Based on the forecast data, providing two actionable recommendations directly derived from the forecast (e.g., inventory targets, staffing levels, reorder timing, or procurement scheduling).")},
                        {"type": "recommendations", "items": recs[:2]},
                        {"type": "callout", "title": _t("规划原则", "Planning Principles"), "text": _t("保守规划参考 95% 上限；成本压降参考 95% 下限；区间越宽越需预留缓冲。", "For conservative planning, refer to the 95% upper bound; for cost reduction, refer to the 95% lower bound. Wider intervals require larger buffers.")},
                    ],
                },
                {
                    "page_no": 6,
                    "title": _t("数据剖析与探索性分析", "Data Profiling & Exploratory Analysis"),
                    "blocks": [
                        {
                            "type": "bullets",
                            "items": [
                                _t(f"时间范围：{date_range_line}", f"Time Range: {date_range_line}"),
                                _t(f"观测值数量：{summ.get('observations') if summ.get('observations') is not None else '—'}", f"Observations: {summ.get('observations') if summ.get('observations') is not None else '—'}"),
                                _t(f"缺失值（目标列）：{(summ.get('missing') or {}).get('missing_values_target', '—')}", f"Missing Values (Target): {(summ.get('missing') or {}).get('missing_values_target', '—')}"),
                                _t(f"均值/标准差：{_fmt_num(ks.get('mean'))} / {_fmt_num(ks.get('std'))}", f"Mean/Std Dev: {_fmt_num(ks.get('mean'))} / {_fmt_num(ks.get('std'))}"),
                            ],
                        },
                        {
                            "type": "bullets",
                            "items": [
                                _t(f"趋势（平均每期变化）：{_fmt_num(trend.get('avg_change_per_period'))}（约 {_fmt_num(trend.get('avg_change_pct_of_mean'), nd=2, pct=True)} 相对均值）", f"Trend (Avg change/period): {_fmt_num(trend.get('avg_change_per_period'))} (~{_fmt_num(trend.get('avg_change_pct_of_mean'), nd=2, pct=True)} of mean)") if trend else _t("趋势：—", "Trend: —"),
                                _t(f"季节性：{'存在' if seas.get('has_seasonality') else ('不明显' if seas.get('has_seasonality') is False else '—')}（建议周期：{seas.get('suggested_periods') or '—'}）", f"Seasonality: {'Yes' if seas.get('has_seasonality') else ('Not obvious' if seas.get('has_seasonality') is False else '—')} (Suggested periods: {seas.get('suggested_periods') or '—'})"),
                            ],
                        },
                    ],
                },
                {
                    "page_no": 7,
                    "title": _t("经典基线模型参数（Baseline Models）", "Classical Baseline Models"),
                    "blocks": [
                        {"type": "p", "text": _t("本页展示 Naïve/季节Naïve/移动平均/ETS/ARIMA(SARIMA) 等基线模型的参数摘要与训练/测试指标。", "This page shows parameter summaries and train/test metrics for baseline models like Naïve, Seasonal Naïve, MA, ETS, and ARIMA.")},
                        {"type": "table", "title": _t("基线模型指标", "Baseline Model Metrics"), "rows": classical_models},
                    ],
                },
                {
                    "page_no": 8,
                    "title": _t("高级 AI 预测模型（Advanced Models）", "Advanced AI Forecast Models"),
                    "blocks": [
                        {"type": "p", "text": _t("本页展示至少一种高级方法（如树模型/回归/深度学习/混合模型），并说明特征、验证与数据泄露控制。", "This page shows at least one advanced method (e.g., Tree Models/Regression/Deep Learning/Hybrid) and explains features, validation, and data leakage control.")},
                        {"type": "table", "title": _t("高级模型指标", "Advanced Model Metrics"), "rows": advanced_models},
                    ],
                },
                {
                    "page_no": 9,
                    "title": _t("业务语言深度洞察", "In-depth Business Insights"),
                    "blocks": [
                        {"type": "bullets", "items": (rep.get("insights") or [])[:6]},
                    ],
                },
                {
                    "page_no": 10,
                    "title": _t("结论与局限性", "Conclusion & Limitations"),
                    "blocks": [
                        {"type": "p", "text": _t("本系统提供端到端预测能力：数据剖析→模型拟合→对比评估→未来预测→运营建议，适用于通用业务需求时间序列。", "This system provides end-to-end forecasting: Data Profiling → Model Fitting → Evaluation → Future Forecast → Operations Recommendations.")},
                        {"type": "bullets", "items": flags_ if flags_ else [_t("局限性：外部冲击（促销/政策/供应中断）可能导致误差上升；建议滚动更新预测并监控误差。", "Limitations: External shocks (promotions/policies/supply disruptions) may increase errors; rolling updates and error monitoring are recommended.")]},
                        {"type": "bullets", "items": [_t("后续改进：引入外部协变量（价格/营销/节假日/天气等）与更稳健的概率预测校准；自动化监控与重训。", "Future Improvements: Incorporate external covariates (price/marketing/holidays/weather) and robust probabilistic forecast calibration; automated monitoring and retraining.")]},
                    ],
                },
                {
                    "page_no": 11,
                    "title": _t("预测数据明细 (附录)", "Forecast Data Details (Appendix)"),
                    "blocks": [
                        {"type": "p", "text": _t("下表展示了最近的历史数据以及未来的预测数据明细（最多显示 36 期）。", "The table below displays the most recent historical data and detailed future forecast data (up to 36 periods).")},
                        {"type": "table", "title": _t("详细数据表", "Detailed Data Table"), "rows": forecast_rows},
                    ],
                },
            ]
            return pages

        report["template"] = {"name": "mgt550_en_11_pages", "pages": 11, "language": "en-US"} if is_en else {"name": "mgt550_cn_11_pages", "pages": 11, "language": "zh-CN"}
        report["pages"] = _build_pages(report)

        self.report = report

        # 打印报告
        self._print_report(report)

        if save_to_disk:
            self._save_report(report)

        print("\n管理报告生成完成!")
        print("=" * 60)

        # 返回清理后的报告以确保JSON可序列化
        return self._clean_for_json(report)

    def _generate_insights(self):
        """生成业务洞察"""
        is_en = str(getattr(self, "language", "en")).lower().startswith("en")
        def _t(zh_str: str, en_str: str) -> str:
            return en_str if is_en else zh_str
        insights = []

        # 获取变量中文标签
        label = self.get_variable_label()

        # 分析趋势
        if self.data is not None:
            y = self.data[self.demand_col]

            # 趋势分析
            if len(y) > 1:
                growth_rate = ((y.iloc[-1] - y.iloc[0]) / y.iloc[0]) * 100
                if growth_rate > 0:
                    insights.append(_t(f"{label}呈现增长趋势，整体增长率为 {growth_rate:.1f}%", f"{label} shows an upward trend, with total growth of {growth_rate:.1f}%"))
                elif growth_rate < 0:
                    insights.append(_t(f"{label}呈现下降趋势，整体下降率为 {abs(growth_rate):.1f}%", f"{label} shows a downward trend, with total decline of {abs(growth_rate):.1f}%"))
                else:
                    insights.append(_t(f"{label}保持稳定，无明显增长或下降趋势", f"{label} remains stable with no clear upward or downward trend"))

            # 季节性分析 (简化)
            if len(y) >= 14 and self.freq == 'D':  # 至少两周数据
                weekly_pattern = y.rolling(window=7).mean()
                if weekly_pattern.std() > y.std() * 0.1:
                    insights.append(_t("数据呈现明显的周度季节性模式", "The series shows a clear weekly seasonality pattern"))

            # 波动性分析
            volatility = y.pct_change().std() * 100
            if volatility > 20:
                prof = self._recommendation_profile()
                if prof.get("allow_inventory"):
                    insights.append(_t(f"{label}波动性较高 ({volatility:.1f}%)，建议增加安全库存或缩短补货周期", f"{label} has high volatility ({volatility:.1f}%). Consider increasing safety stock or shortening replenishment cycles"))
                else:
                    insights.append(_t(f"{label}波动性较高 ({volatility:.1f}%)，建议增加资源缓冲或缩短滚动复核周期", f"{label} has high volatility ({volatility:.1f}%). Consider adding resource buffers or shortening the rolling review cycle"))
            elif volatility < 5:
                prof = self._recommendation_profile()
                if prof.get("allow_inventory"):
                    insights.append(_t(f"{label}波动性较低 ({volatility:.1f}%)，库存/补货节奏相对稳定", f"{label} has low volatility ({volatility:.1f}%). Inventory/replenishment rhythm is relatively stable"))
                else:
                    insights.append(_t(f"{label}波动性较低 ({volatility:.1f}%)，运营节奏相对稳定", f"{label} has low volatility ({volatility:.1f}%). Operational rhythm is relatively stable"))

        # 时间序列分析洞察
        if hasattr(self, 'ts_analysis_results') and self.ts_analysis_results:
            ts_results = self.ts_analysis_results

            # 平稳性洞察
            if ts_results.get('stationarity'):
                stationarity = ts_results['stationarity']
                if not stationarity['is_stationary']:
                    insights.append(_t("数据非平稳，建议进行差分处理以提高模型稳定性", "The series is non-stationary; consider differencing to improve model stability"))
                else:
                    insights.append(_t("数据平稳，适合直接建模", "The series appears stationary and is suitable for direct modeling"))

            # 季节性洞察
            if ts_results.get('decomposition'):
                decomposition = ts_results['decomposition']
                if decomposition and decomposition.get('has_strong_seasonality'):
                    insights.append(_t("数据呈现强季节性模式，建议使用季节性模型", "Strong seasonality detected; consider seasonal models"))
                elif decomposition and decomposition.get('seasonal_ratio', 0) > 0.1:
                    insights.append(_t("数据呈现中等季节性，可考虑季节性调整", "Moderate seasonality detected; consider seasonal adjustment"))

            # 季节性检测洞察
            if ts_results.get('seasonality'):
                seasonality = ts_results['seasonality']
                if seasonality.get('has_seasonality'):
                    insights.append(_t("检测到明显的季节性模式", "A clear seasonal pattern is detected"))

        # 模型洞察
        if self.best_model:
            insights.append(_t(f"最佳预测模型为 {self.best_model}，在测试集上表现最优", f"The best forecasting model is {self.best_model}, achieving the best performance on the test set"))

            if 'arima' in self.best_model:
                insights.append(_t("ARIMA模型对时间序列的自相关结构捕捉较好", "ARIMA captures the autocorrelation structure of the time series effectively"))
            elif 'random_forest' in self.best_model or 'xgboost' in self.best_model:
                insights.append(_t("树模型能够捕捉复杂的非线性关系", "Tree-based models can capture complex nonlinear relationships"))
            elif 'naive' in self.best_model or 'moving_average' in self.best_model:
                insights.append(_t(f"简单模型表现最佳，表明{label}模式相对简单稳定", f"Simple models perform best, suggesting {label} follows a relatively stable and simple pattern"))

        # 预测洞察
        if hasattr(self, 'forecast_results'):
            forecast_values = self.forecast_results['forecast']
            if len(forecast_values) >= 2:
                forecast_growth = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
                if forecast_growth > 5:
                    insights.append(_t(f"未来{label}预计增长 {forecast_growth:.1f}%，建议提前准备", f"Future {label} is expected to increase by {forecast_growth:.1f}%; consider preparing in advance"))
                elif forecast_growth < -5:
                    insights.append(_t(f"未来{label}预计下降 {abs(forecast_growth):.1f}%，建议调整生产计划", f"Future {label} is expected to decrease by {abs(forecast_growth):.1f}%; consider adjusting production planning"))

        return insights

    def _generate_recommendations(self):
        """生成行动建议"""
        recommendations = []

        # 获取变量中文标签
        label = self.get_variable_label()

        # 基于预测的建议
        if hasattr(self, 'forecast_results'):
            avg_forecast = np.mean(self.forecast_results['forecast'])
            current_level = self.data[self.demand_col].iloc[-1] if self.data is not None else avg_forecast
            prof = self._recommendation_profile()

            if avg_forecast > current_level * 1.1:
                if prof.get("allow_inventory"):
                    recommendations.append(
                        f"预计未来{label}将增加约 {(avg_forecast/current_level-1)*100:.1f}%，"
                        f"建议提高备货/补货目标并增加安全库存（例如提升到当前水平的110%-120%）"
                    )
                else:
                    recommendations.append(
                        f"预计未来{label}将增加约 {(avg_forecast/current_level-1)*100:.1f}%，"
                        f"建议提前规划资源/预算上限，并预留弹性缓冲（例如提升到当前水平的110%-120%）"
                    )
            elif avg_forecast < current_level * 0.9:
                if prof.get("allow_inventory"):
                    recommendations.append(
                        f"预计未来{label}将下降约 {(1-avg_forecast/current_level)*100:.1f}%，"
                        f"建议下调订货/补货节奏，避免积压成本上升"
                    )
                else:
                    recommendations.append(
                        f"预计未来{label}将下降约 {(1-avg_forecast/current_level)*100:.1f}%，"
                        f"建议下调可变投入并优化节奏，避免资源闲置"
                    )
            else:
                recommendations.append(
                    f"{label}保持相对稳定，建议维持当前策略，"
                    "但保持对市场变化的敏感度"
                )

        # 基于模型性能的建议
        if self.evaluation_results is not None:
            best_mape = self.evaluation_results.iloc[0]['MAPE']

            if best_mape < 10:
                if prof.get("allow_inventory"):
                    recommendations.append(
                        f"模型预测准确度高 (MAPE={best_mape:.1f}%)，"
                        "可以依赖模型预测进行更精细的补货/库存与运营计划"
                    )
                else:
                    recommendations.append(
                        f"模型预测准确度高 (MAPE={best_mape:.1f}%)，"
                        "可以依赖模型预测进行更精细的运营节奏与资源配置"
                    )
            elif best_mape < 20:
                recommendations.append(
                    f"模型预测准确度中等 (MAPE={best_mape:.1f}%)，"
                    "建议结合业务经验和模型预测进行决策，并保持一定缓冲"
                )
            else:
                recommendations.append(
                    f"模型预测误差较大 (MAPE={best_mape:.1f}%)，"
                    "建议谨慎使用模型预测，更多依赖历史经验和市场分析"
                )

        # 确保至少有2条建议
        if len(recommendations) < 2:
            recommendations.append(
                "建议建立定期(如每周)的预测复核机制，"
                "根据实际销售数据持续优化预测模型"
            )
            recommendations.append(
                "考虑引入外部数据源(如天气、节假日、促销活动)，"
                "进一步提升预测准确性"
            )

        return recommendations[:2]  # 返回前2条最重要的建议

    def _generate_model_selection_summary(self):
        try:
            if self.evaluation_results is None or len(self.evaluation_results) == 0:
                return "尚未生成模型评估结果。"

            best_name = str(self.best_model or "")
            row_df = self.evaluation_results[self.evaluation_results["model"].astype(str) == best_name]
            best_row = row_df.iloc[0].to_dict() if len(row_df) > 0 else self.evaluation_results.iloc[0].to_dict()

            rmse = best_row.get("RMSE")
            in_rmse = best_row.get("in_sample_RMSE")
            gap = None
            try:
                if rmse is not None and in_rmse is not None:
                    gap = float(rmse) - float(in_rmse)
            except Exception:
                gap = None

            model_type = "高级模型" if (self.advanced_models or {}).get(best_name) is not None else "经典时序模型"
            parts = [f"推荐模型：{best_name}（{model_type}）"]
            if rmse is not None:
                try:
                    parts.append(f"以最后20%留出测试集为准，RMSE={float(rmse):.4f}。")
                except Exception:
                    pass
            reason = getattr(self, "best_model_selection_reason", None)
            if reason:
                parts.append(f"决策逻辑：{reason}。")

            if gap is not None:
                if gap > 0:
                    parts.append("训练集与测试集误差存在差距，需关注泛化与过拟合风险。")
                else:
                    parts.append("训练集与测试集误差相近，泛化表现较稳定。")

            if "arima" in best_name.lower():
                parts.append("该模型能更好刻画序列的自相关结构，适合存在惯性/周期波动的数据。")
            elif "ets" in best_name.lower() or "exponential" in best_name.lower():
                parts.append("该模型对趋势与（可能的）季节性有较好刻画，适合平滑且可解释的需求序列。")
            elif "naive" in best_name.lower() or "moving_average" in best_name.lower():
                parts.append("简单基线模型已足够，说明序列模式相对稳定、复杂模型收益有限。")
            elif "random_forest" in best_name.lower() or "xgboost" in best_name.lower():
                parts.append("该模型能捕捉非线性关系，但需重点关注过拟合与数据泄漏风险，并优先检查残差是否接近白噪声。")

            return " ".join([p for p in parts if p])
        except Exception:
            return "模型选择解释生成失败。"

    def _generate_risk_flags(self):
        risks = []
        try:
            if self.evaluation_results is not None and len(self.evaluation_results) > 0:
                high = self.evaluation_results[self.evaluation_results.get("overfitting_risk") == "高"] if "overfitting_risk" in self.evaluation_results.columns else None
                if high is not None and len(high) > 0:
                    names = [str(x) for x in high["model"].tolist()[:6]]
                    risks.append("过拟合风险较高的模型：" + "、".join(names))
        except Exception:
            pass

        try:
            best_name = str(self.best_model or "")
            if (self.advanced_models or {}).get(best_name) is not None:
                risks.append("高级模型未来预测默认将外生特征取最后已知值并递推滞后项；若未来外生变量变化较大，预测可能偏离。")
        except Exception:
            pass

        if not risks:
            risks.append("未发现明显的过拟合/数据泄漏风险信号，但仍建议定期用新数据复核模型表现。")
        return risks

    def _print_report(self, report):
        """打印报告到控制台"""
        print("\n" + "=" * 80)
        print("FORECASTPRO 管理报告")
        print("=" * 80)

        print(f"\n报告日期: {report['report_date']}")
        print(f"数据概览: {report['data_summary']['total_observations']} 个观测值，"
              f"频率: {report['data_summary']['frequency']}")

        print(f"\n📊 最佳模型: {report['best_model']['name']}")
        best_metrics = report['best_model'].get('metrics') or {}
        if best_metrics:
            try:
                print(f"   - 均方根误差 (RMSE): {float(best_metrics['RMSE']):.4f}")
            except Exception:
                pass
            try:
                print(f"   - 平均绝对误差 (MAE): {float(best_metrics['MAE']):.4f}")
            except Exception:
                pass
            try:
                print(f"   - 平均绝对百分比误差 (MAPE): {float(best_metrics['MAPE']):.2f}%")
            except Exception:
                pass
            if best_metrics.get('overfitting_risk') is not None:
                print(f"   - 过拟合风险: {best_metrics['overfitting_risk']}")
        if report['best_model'].get('selection_reason'):
            print(f"   - 选择逻辑: {report['best_model']['selection_reason']}")

        print("\n🏆 模型性能排名:")
        print("-" * 80)
        comparison_df = pd.DataFrame(report['model_comparison'])
        cols = [c for c in ['model', 'RMSE', 'MAE', 'MAPE', 'resid_white_noise', 'overfitting_risk', 'type'] if c in comparison_df.columns]
        print(comparison_df[cols].sort_values('RMSE').to_string(index=False) if 'RMSE' in comparison_df.columns else comparison_df.to_string(index=False))

        # 时间序列分析结果
        if report.get('ts_analysis'):
            print("\n📈 时间序列分析:")
            print("-" * 40)
            ts_results = report['ts_analysis']

            if ts_results.get('stationarity'):
                stat = ts_results['stationarity']
                if stat:
                    stationarity_status = "平稳" if stat.get('is_stationary') else "非平稳"
                    print(f"  平稳性: {stationarity_status} (p值: {stat.get('p_value', 0):.4f})")

            if ts_results.get('decomposition'):
                decomp = ts_results['decomposition']
                if decomp:
                    print(f"  季节性分解:")
                    print(f"    - 趋势成分占比: {decomp.get('trend_ratio', 0):.2%}")
                    print(f"    - 季节性成分占比: {decomp.get('seasonal_ratio', 0):.2%}")
                    print(f"    - 残差成分占比: {decomp.get('residual_ratio', 0):.2%}")

            if ts_results.get('seasonality'):
                season = ts_results['seasonality']
                if season:
                    seasonality_status = "有季节性" if season.get('has_seasonality') else "无显著季节性"
                    print(f"  季节性检测: {seasonality_status}")

        print("\n🔮 业务洞察:")
        print("-" * 40)
        for i, insight in enumerate(report['insights'], 1):
            print(f"{i}. {insight}")

        print("\n💡 行动建议:")
        print("-" * 40)
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")

        print("\n" + "=" * 80)
        print("报告结束 - ForecastPro AI需求预测系统")
        print("=" * 80)

    def _clean_for_json(self, obj):
        """递归清理对象以便JSON序列化"""
        import pandas as pd
        import numpy as np
        from datetime import datetime as dt
        import math

        if isinstance(obj, (pd.Timestamp, dt)):
            return obj.isoformat()
        elif isinstance(obj, pd.DatetimeIndex):
            return [d.isoformat() for d in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            v = float(obj)
            return v if math.isfinite(v) else None
        elif isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj

    def _save_report(self, report, output_dir: str = "./reports"):
        """保存报告到文件"""
        import os

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 清理报告数据，确保JSON可序列化
        clean_report = self._clean_for_json(report)

        # 保存JSON报告
        json_path = os.path.join(output_dir, f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, ensure_ascii=False, indent=2)

        # 保存文本报告
        txt_path = os.path.join(output_dir, f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        cover = report.get("cover") or {}
        time_range = cover.get("time_range") or {}
        exec_sum = report.get("executive_summary") or {}
        data_overview = report.get("data_overview") or {}
        best = report.get("best_model") or {}
        best_metrics = best.get("metrics") or {}
        chart_table = report.get("chart_table") or []
        notes = report.get("model_comparison_notes") or {}
        interp = report.get("forecast_interpretation") or {}
        recs = report.get("actionable_recommendations") or []
        flags = report.get("uncertainty_flags") or []

        def _fmt_dt(v):
            try:
                if v is None:
                    return ""
                return pd.to_datetime(v).strftime("%Y-%m-%d")
            except Exception:
                return str(v)

        def _fmt_num(v, nd=3):
            try:
                if v is None:
                    return ""
                return f"{float(v):.{nd}f}"
            except Exception:
                return str(v)

        train_end = None
        test_start = None
        try:
            if self.train_data and self.train_data.get("y") is not None and len(self.train_data["y"]) > 0:
                train_end = self.train_data["y"].index[-1]
            if self.test_data and self.test_data.get("y") is not None and len(self.test_data["y"]) > 0:
                test_start = self.test_data["y"].index[0]
        except Exception:
            train_end = None
            test_start = None

        def _table_lines(rows, limit=200):
            header = "| 时期 | 历史数据 | 拟合值 | 预测值 | 95%下限 | 95%上限 |"
            sep = "|---|---:|---:|---:|---:|---:|"
            lines = [header, sep]
            for r in rows[: int(limit)]:
                lines.append(
                    "| "
                    + _fmt_dt(r.get("period"))
                    + " | "
                    + _fmt_num(r.get("history_actual"))
                    + " | "
                    + _fmt_num(r.get("fitted"))
                    + " | "
                    + _fmt_num(r.get("forecast"))
                    + " | "
                    + _fmt_num(r.get("lower_95"))
                    + " | "
                    + _fmt_num(r.get("upper_95"))
                    + " |"
                )
            return "\n".join(lines)

        cover_title = cover.get("title") or f"需求预测报告 — {report.get('dataset_name') or '数据集'}"
        dataset_name = cover.get("dataset") or (report.get("dataset_name") or "数据集")
        start_s = _fmt_dt(time_range.get("start"))
        end_s = _fmt_dt(time_range.get("end"))
        freq_s = time_range.get("frequency") or str(report.get("data_summary", {}).get("frequency") or "")

        analyzed_data = exec_sum.get("analyzed_data") or ""
        winning_reason = (exec_sum.get("winning_model") or {}).get("reason") or best.get("selection_reason") or ""
        top_recs = exec_sum.get("top_business_recommendations") or []

        ds = data_overview.get("summary") or {}
        ks = data_overview.get("key_statistics") or {}
        tr = data_overview.get("trend") or {}
        seas = data_overview.get("seasonality") or {}

        trend_line = ""
        if tr:
            trend_line = f"每周期平均变化约 {_fmt_num(tr.get('avg_change_per_period'))}（约占均值 {_fmt_num(tr.get('avg_change_pct_of_mean'), nd=2)}%）。"

        season_line = ""
        if seas:
            hs = seas.get("has_seasonality")
            if hs is True:
                season_line = f"检测到季节性；建议周期: {seas.get('suggested_periods') or ''}。"
            elif hs is False:
                season_line = "未检测到显著季节性。"

        interp_lines = []
        if interp:
            peak = interp.get("peak") or {}
            trough = interp.get("trough") or {}
            interp_lines.append(f"趋势：未来平均预测值约 {_fmt_num(interp.get('avg_forecast'))}。")
            if interp.get("avg_vs_last_delta") is not None:
                interp_lines.append(f"相对最近一期实际值变化约 {_fmt_num(interp.get('avg_vs_last_delta'))}（{_fmt_num(interp.get('avg_vs_last_delta_pct'), nd=2)}%）。")
            interp_lines.append(f"峰值需求：{_fmt_dt(peak.get('date'))}，点预测 {_fmt_num(peak.get('forecast'))}，95%上限 {_fmt_num(peak.get('upper_95'))}。")
            interp_lines.append(f"低谷期：{_fmt_dt(trough.get('date'))}，点预测 {_fmt_num(trough.get('forecast'))}，95%下限 {_fmt_num(trough.get('lower_95'))}。")
            if interp.get("avg_interval_width") is not None:
                interp_lines.append(f"不确定性：平均区间宽度约 {_fmt_num(interp.get('avg_interval_width'))}。")

        notes_lines = []
        for k, v in notes.items():
            if k and v:
                notes_lines.append(f"- {k}: {v}")

        rec_block = []
        for i, r in enumerate(recs[:2], 1):
            rec_block.append(f"建议 {i}：{r.get('title','')}")
            rec_block.append(f"- 发现：{r.get('finding','')}")
            rec_block.append(f"- 行动：{r.get('action','')}")
            rec_block.append(f"- 理由：{r.get('reason','')}")

        flag_block = "\n".join([f"- {x}" for x in flags]) if flags else "- 无"

        report_text = "\n".join(
            [
                f"{cover_title}",
                "",
                f"编制日期：{report.get('report_date')}（{cover.get('generated_by') or ''}）",
                f"数据集：{dataset_name}，时间段：{start_s} 至 {end_s}，频率：{freq_s}",
                "",
                "第 1 部分：执行摘要",
                f"- 分析了哪些数据：{analyzed_data}",
                f"- 哪个模型胜出以及原因：{best.get('name') or ''}。{winning_reason}",
                "- 两项最重要的商业建议：",
                "\n".join([f"  - {x}" for x in top_recs]) if top_recs else "  - （无）",
                "",
                "第 2 节：数据概览",
                f"- 观测值：{ds.get('observations', '')}",
                f"- 时间段：{start_s} 至 {end_s}",
                f"- 频率：{freq_s}",
                f"- 缺失值（目标列）：{(ds.get('missing') or {}).get('missing_values_target','')}",
                f"- 关键统计：均值 {_fmt_num(ks.get('mean'))}，标准差 {_fmt_num(ks.get('std'))}，最小值 {_fmt_num(ks.get('min'))}，最大值 {_fmt_num(ks.get('max'))}",
                f"- 趋势与季节性：{trend_line} {season_line}".strip(),
                "",
                "第 3 节：预测图表",
                "图表包含：历史需求（深蓝实线）、拟合值（橙色虚线）、预测值（绿色实线，测试期+未来）、95%预测区间（灰色阴影）、训练/测试分割线（虚线）。",
                f"训练/测试划分日期：训练集截止 {_fmt_dt(train_end)}；测试集起始 {_fmt_dt(test_start)}。",
                "",
                _table_lines(chart_table, limit=200) if chart_table else "（无可用表格数据）",
                "",
                "第 4 节：模型比较表",
                self.evaluation_results.to_string(index=False),
                "",
                "排名较低模型可能原因（简要）：",
                "\n".join(notes_lines) if notes_lines else "- （无）",
                "",
                "数据泄露与过拟合检查说明：",
                "- 对所有模型使用同一测试集划分，不改变测试集范围。",
                "- 测试集观测数据不参与训练与调参；仅用于评估与绘图展示。",
                "",
                "第 5 节：预测解读",
                "\n".join([f"- {x}" for x in interp_lines]) if interp_lines else "- （无）",
                "",
                "第 6 节：可操作的商业建议",
                "\n".join(rec_block) if rec_block else "（无）",
                "",
                "第 7 节：技术附录（可选）",
                "指标公式（用于核对）：",
                "- MAE = (1/n) × Σ|实际值 - 预测值|",
                "- RMSE = √[(1/n) × Σ（实际值 - 预测值）²]",
                "- MAPE = (1/n) × Σ|（实际值 - 预测值）/ 实际值| × 100",
                "",
                "不确定性提示：",
                flag_block,
                "",
            ]
        )

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n报告已保存到:")
        print(f"  JSON: {json_path}")
        print(f"  TXT: {txt_path}")

    def export_to_excel(self, future_by_method: dict = None, output_dir: str = "./reports/excel_exports"):
        """
        将时间序列预测的训练集、测试集、未来预测数据导出为Excel文件，分为三张Sheet。
        作为后端数据检查使用。
        """
        import os
        import pandas as pd
        import numpy as np
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = os.path.join(output_dir, f"forecast_data_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                def _remove_tz(series_or_index):
                    if hasattr(series_or_index, 'dt') and getattr(series_or_index.dt, 'tz', None) is not None:
                        return series_or_index.dt.tz_localize(None)
                    if hasattr(series_or_index, 'tz') and series_or_index.tz is not None:
                        return series_or_index.tz_localize(None)
                    return series_or_index

                if self.train_data is not None and 'y' in self.train_data:
                    train_idx = _remove_tz(pd.to_datetime(self.train_data['y'].index))
                    train_df = pd.DataFrame({'Actual': self.train_data['y'].values}, index=train_idx)

                    train_idx_adv = None
                    try:
                        if hasattr(self, "train_features") and self.train_features is not None and self.train_features.get("y") is not None:
                            train_idx_adv = _remove_tz(pd.to_datetime(self.train_features["y"].index))
                    except Exception:
                        train_idx_adv = None

                    for name in (self.baseline_models or {}):
                        res = self.baseline_models[name]
                        if 'fitted' in res and res['fitted'] is not None:
                            try:
                                fitted = np.asarray(res['fitted'], dtype=float)
                                s = pd.Series(fitted, index=train_idx[: len(fitted)])
                                train_df[f'Fitted_{name}'] = s.reindex(train_df.index).values
                            except Exception:
                                pass
                    for name in (self.advanced_models or {}):
                        res = self.advanced_models[name]
                        if 'fitted' in res and res['fitted'] is not None:
                            try:
                                fitted = np.asarray(res['fitted'], dtype=float)
                                idx = train_idx_adv if train_idx_adv is not None else train_idx
                                s = pd.Series(fitted, index=idx[: len(fitted)])
                                train_df[f'Fitted_{name}'] = s.reindex(train_df.index).values
                            except Exception:
                                pass

                    train_df.reset_index(names='Date').to_excel(writer, sheet_name='训练集 (Training)', index=False)

                if self.test_data is not None and 'y' in self.test_data:
                    test_idx = _remove_tz(pd.to_datetime(self.test_data['y'].index))
                    test_df = pd.DataFrame({'Actual': self.test_data['y'].values}, index=test_idx)

                    test_idx_adv = None
                    try:
                        if hasattr(self, "test_features") and self.test_features is not None and self.test_features.get("y") is not None:
                            test_idx_adv = _remove_tz(pd.to_datetime(self.test_features["y"].index))
                    except Exception:
                        test_idx_adv = None

                    for name in (self.baseline_models or {}):
                        res = self.baseline_models[name]
                        if 'predictions' in res and res['predictions'] is not None:
                            try:
                                preds = np.asarray(res['predictions'], dtype=float)
                                s = pd.Series(preds, index=test_idx[: len(preds)])
                                test_df[f'Predicted_{name}'] = s.reindex(test_df.index).values
                            except Exception:
                                pass
                    for name in (self.advanced_models or {}):
                        res = self.advanced_models[name]
                        if 'predictions' in res and res['predictions'] is not None:
                            try:
                                preds = np.asarray(res['predictions'], dtype=float)
                                idx = test_idx_adv if test_idx_adv is not None else test_idx
                                s = pd.Series(preds, index=idx[: len(preds)])
                                test_df[f'Predicted_{name}'] = s.reindex(test_df.index).values
                            except Exception:
                                pass

                    test_df.reset_index(names='Date').to_excel(writer, sheet_name='测试集 (Testing)', index=False)

                future_df = pd.DataFrame()
                # 如果传入了 future_by_method (从API端构建)
                if future_by_method:
                    # 找出所有 future dates
                    all_dates = []
                    for m, res in future_by_method.items():
                        if res and 'dates' in res:
                            all_dates = res['dates']
                            break
                    
                    if all_dates:
                        future_df['Date'] = _remove_tz(pd.to_datetime(all_dates))
                        for m, res in future_by_method.items():
                            if res and 'forecast' in res:
                                try:
                                    future_df[f'Forecast_{m}'] = res['forecast']
                                    if 'lower_bound' in res:
                                        future_df[f'Lower_{m}'] = res['lower_bound']
                                    if 'upper_bound' in res:
                                        future_df[f'Upper_{m}'] = res['upper_bound']
                                except Exception:
                                    pass
                else:
                    # 如果只有 self.forecast_results
                    if hasattr(self, 'forecast_results') and self.forecast_results:
                        res = self.forecast_results
                        if 'dates' in res:
                            future_df['Date'] = _remove_tz(pd.to_datetime(res['dates']))
                            model_name = res.get('model', 'auto')
                            if 'forecast' in res:
                                future_df[f'Forecast_{model_name}'] = res['forecast']
                            if 'lower_bound' in res:
                                future_df[f'Lower_{model_name}'] = res['lower_bound']
                            if 'upper_bound' in res:
                                future_df[f'Upper_{model_name}'] = res['upper_bound']

                if not future_df.empty:
                    future_df.to_excel(writer, sheet_name='未来预测期 (Future)', index=False)

            print(f"数据检查Excel已生成: {excel_path}")
        except Exception as e:
            print(f"生成Excel失败: {e}")

    def export_best_excel(self, periods: int = 4, output_dir: str = "./reports/excel_best"):
        import os
        import numpy as np
        import pandas as pd
        from datetime import datetime

        if self.data is None or self.train_data is None or self.test_data is None:
            raise ValueError("请先加载数据并完成数据划分")

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_name = str(self.best_model or "best")
        excel_path = os.path.join(output_dir, f"forecast_best_{best_name}_{ts}.xlsx")

        if self.forecast_results is None or int(periods) != int(len((self.forecast_results or {}).get("forecast") or [])):
            try:
                self.generate_forecast(periods=int(periods), forecast_method="auto")
            except Exception:
                pass

        y_full = self.data[self.demand_col].astype(float)
        train_idx = self.train_data["y"].index
        test_idx = self.test_data["y"].index

        def _aligned_series(values, idx_values, idx_target):
            if values is None or idx_values is None or idx_target is None:
                return [None] * int(len(idx_target))
            try:
                v = list(values)
                idxv = list(idx_values)
            except Exception:
                return [None] * int(len(idx_target))
            n = min(len(v), len(idxv))
            m = {}
            for i in range(n):
                try:
                    x = float(v[i])
                    if pd.notna(x) and float("inf") != abs(x):
                        m[idxv[i]] = x
                    else:
                        m[idxv[i]] = None
                except Exception:
                    m[idxv[i]] = None
            return [m.get(t, None) for t in idx_target]

        fitted = None
        fitted_idx = None
        preds = None
        preds_idx = None
        try:
            if (self.baseline_models or {}).get(best_name) is not None:
                fitted = (self.baseline_models or {}).get(best_name, {}).get("fitted")
                preds = (self.baseline_models or {}).get(best_name, {}).get("predictions")
                fitted_idx = train_idx
                preds_idx = test_idx
            elif (self.advanced_models or {}).get(best_name) is not None:
                fitted = (self.advanced_models or {}).get(best_name, {}).get("fitted")
                preds = (self.advanced_models or {}).get(best_name, {}).get("predictions")
                if hasattr(self, "train_features") and self.train_features is not None and self.train_features.get("y") is not None:
                    fitted_idx = self.train_features["y"].index
                else:
                    fitted_idx = train_idx
                if hasattr(self, "test_features") and self.test_features is not None and self.test_features.get("y") is not None:
                    preds_idx = self.test_features["y"].index
                else:
                    preds_idx = test_idx
        except Exception:
            fitted = None
            preds = None
            fitted_idx = train_idx
            preds_idx = test_idx

        train_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(train_idx),
                "Actual": self.train_data["y"].astype(float).values,
                "Fitted": _aligned_series(fitted, fitted_idx, train_idx),
            }
        )
        test_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(test_idx),
                "Actual": self.test_data["y"].astype(float).values,
                "Predicted": _aligned_series(preds, preds_idx, test_idx),
            }
        )

        future_df = pd.DataFrame()
        try:
            fr = self.forecast_results or {}
            future_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(fr.get("dates") or []),
                    "Forecast": fr.get("forecast") or [],
                    "Lower_95": fr.get("lower_bound") or [],
                    "Upper_95": fr.get("upper_bound") or [],
                }
            )
        except Exception:
            future_df = pd.DataFrame()

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            train_df.to_excel(writer, sheet_name="Training", index=False)
            test_df.to_excel(writer, sheet_name="Testing", index=False)
            if not future_df.empty:
                future_df.to_excel(writer, sheet_name="Future", index=False)

        self.best_excel_path = excel_path
        return excel_path

    def export_models_summary_excel(self, periods: int = 4, output_dir: str = "./reports"):
        import os
        import json
        import numpy as np
        import pandas as pd
        from datetime import datetime

        if self.evaluation_results is None or len(self.evaluation_results) == 0:
            raise ValueError("请先完成模型评估（evaluate_models）")

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(output_dir, f"forecast_models_summary_{ts}.xlsx")

        def _pick(d, keys):
            for k in keys:
                if isinstance(d, dict) and k in d and d[k] is not None:
                    return d[k]
            return None

        def _params_brief(model_name: str):
            model_name = str(model_name or "")
            p = None
            try:
                p = (self.model_results or {}).get(model_name, {}).get("params")
            except Exception:
                p = None

            if not isinstance(p, dict):
                return ""

            if model_name == "naive":
                return f"strategy={_pick(p, ['strategy'])}"
            if model_name == "seasonal_naive":
                return f"seasonal_period={_pick(p, ['seasonal_period'])}"
            if model_name == "moving_average":
                sel = _pick(p, ["selection"])
                if sel is not None:
                    return f"window={_pick(p, ['window'])}, selection={sel}"
                return f"window={_pick(p, ['window'])}"

            if model_name.startswith("ets"):
                cfg = p.get("config") if isinstance(p.get("config"), dict) else {}
                fp = p.get("fitted_params") if isinstance(p.get("fitted_params"), dict) else {}
                parts = []
                if cfg:
                    if cfg.get("trend") is not None:
                        parts.append(f"trend={cfg.get('trend')}")
                    if cfg.get("seasonal") is not None:
                        parts.append(f"seasonal={cfg.get('seasonal')}")
                    if cfg.get("seasonal_periods") is not None:
                        parts.append(f"sp={cfg.get('seasonal_periods')}")
                for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal"]:
                    if k in fp and fp[k] is not None:
                        try:
                            label = "alpha" if k == "smoothing_level" else "beta" if k == "smoothing_trend" else "gamma" if k == "smoothing_seasonal" else k
                            parts.append(f"{label}={float(fp[k]):.4f}")
                        except Exception:
                            parts.append(f"{k}={fp[k]}")
                for k in ["initial_level", "initial_trend"]:
                    if k in fp and fp[k] is not None:
                        try:
                            parts.append(f"{k}={float(fp[k]):.4f}")
                        except Exception:
                            parts.append(f"{k}={fp[k]}")
                if fp.get("initial_seasons") is not None:
                    try:
                        arr = list(fp.get("initial_seasons"))
                        show = [float(x) for x in arr[: min(3, len(arr))]]
                        parts.append(f"initial_seasons={show}")
                    except Exception:
                        try:
                            parts.append(f"initial_seasons={fp.get('initial_seasons')}")
                        except Exception:
                            pass
                return ", ".join(parts)

            if model_name == "arima":
                order = p.get("order")
                seasonal_order = p.get("seasonal_order")
                aic = p.get("aic")
                parts = []
                if order is not None:
                    parts.append(f"order={tuple(order) if isinstance(order, (list, tuple)) else order}")
                if seasonal_order is not None:
                    parts.append(f"seasonal_order={tuple(seasonal_order) if isinstance(seasonal_order, (list, tuple)) else seasonal_order}")
                if aic is not None:
                    try:
                        parts.append(f"aic={float(aic):.2f}")
                    except Exception:
                        parts.append(f"aic={aic}")
                return ", ".join(parts)

            if model_name in {"ridge_regression", "lasso_regression"}:
                a = _pick(p, ["alpha"])
                return f"alpha={a}" if a is not None else ""

            if model_name == "random_forest":
                parts = []
                for k in ["n_estimators", "max_depth", "random_state"]:
                    v = _pick(p, [k])
                    if v is not None:
                        parts.append(f"{k}={v}")
                return ", ".join(parts)

            if model_name == "xgboost":
                parts = []
                for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state"]:
                    v = _pick(p, [k])
                    if v is not None:
                        parts.append(f"{k}={v}")
                return ", ".join(parts)

            try:
                return json.dumps(p, ensure_ascii=False, default=str)
            except Exception:
                return str(p)

        def _aligned_series(values, idx_values, idx_target):
            if values is None or idx_values is None or idx_target is None:
                return [None] * int(len(idx_target))
            try:
                v = list(values)
                idxv = list(idx_values)
            except Exception:
                return [None] * int(len(idx_target))
            n = min(len(v), len(idxv))
            m = {}
            for i in range(n):
                try:
                    x = float(v[i])
                    if pd.notna(x) and float("inf") != abs(x):
                        m[idxv[i]] = x
                    else:
                        m[idxv[i]] = None
                except Exception:
                    m[idxv[i]] = None
            return [m.get(t, None) for t in idx_target]

        train_idx = self.train_data["y"].index if self.train_data and self.train_data.get("y") is not None else None
        test_idx = self.test_data["y"].index if self.test_data and self.test_data.get("y") is not None else None

        training_matrix = None
        testing_matrix = None
        if train_idx is not None:
            training_matrix = pd.DataFrame({"Date": pd.to_datetime(train_idx), "Actual": self.train_data["y"].astype(float).values})
        if test_idx is not None:
            testing_matrix = pd.DataFrame({"Date": pd.to_datetime(test_idx), "Actual": self.test_data["y"].astype(float).values})

        rows = []
        for _, r in self.evaluation_results.iterrows():
            model = str(r.get("model"))
            detail = (self.model_results or {}).get(model, {}) if hasattr(self, "model_results") else {}
            holdout = detail.get("holdout") if isinstance(detail.get("holdout"), dict) else {}
            ins = detail.get("in_sample") if isinstance(detail.get("in_sample"), dict) else {}
            resid = detail.get("residual_diagnostics") if isinstance(detail.get("residual_diagnostics"), dict) else {}

            try:
                if training_matrix is not None:
                    fitted = None
                    fitted_idx = None
                    if (self.baseline_models or {}).get(model) is not None:
                        fitted = (self.baseline_models or {}).get(model, {}).get("fitted")
                        fitted_idx = train_idx
                    elif (self.advanced_models or {}).get(model) is not None:
                        fitted = (self.advanced_models or {}).get(model, {}).get("fitted")
                        if hasattr(self, "train_features") and self.train_features is not None and self.train_features.get("y") is not None:
                            fitted_idx = self.train_features["y"].index
                        else:
                            fitted_idx = train_idx
                    training_matrix[model] = _aligned_series(fitted, fitted_idx, train_idx)
            except Exception:
                pass

            try:
                if testing_matrix is not None:
                    preds = None
                    preds_idx = None
                    if (self.baseline_models or {}).get(model) is not None:
                        preds = (self.baseline_models or {}).get(model, {}).get("predictions")
                        preds_idx = test_idx
                    elif (self.advanced_models or {}).get(model) is not None:
                        preds = (self.advanced_models or {}).get(model, {}).get("predictions")
                        if hasattr(self, "test_features") and self.test_features is not None and self.test_features.get("y") is not None:
                            preds_idx = self.test_features["y"].index
                        else:
                            preds_idx = test_idx
                    testing_matrix[model] = _aligned_series(preds, preds_idx, test_idx)
            except Exception:
                pass

            rows.append(
                {
                    "model": model,
                    "type": str(r.get("type") or ""),
                    "RMSE_holdout": float(holdout.get("RMSE")) if holdout.get("RMSE") is not None else float(r.get("RMSE")),
                    "MAE_holdout": float(holdout.get("MAE")) if holdout.get("MAE") is not None else float(r.get("MAE")),
                    "MAPE_holdout": float(holdout.get("MAPE")) if holdout.get("MAPE") is not None else float(r.get("MAPE")),
                    "RMSE_train": float(ins.get("RMSE")) if ins.get("RMSE") is not None else None,
                    "MAE_train": float(ins.get("MAE")) if ins.get("MAE") is not None else None,
                    "MAPE_train": float(ins.get("MAPE")) if ins.get("MAPE") is not None else None,
                    "resid_white_noise": resid.get("white_noise"),
                    "ljung_box_pvalue": resid.get("ljung_box_pvalue"),
                    "overfitting_risk": str(r.get("overfitting_risk") or ""),
                    "params": _params_brief(model),
                    "is_best": bool(str(model) == str(self.best_model)),
                    "best_selection_reason": getattr(self, "best_model_selection_reason", None) if str(model) == str(self.best_model) else None,
                }
            )

        df = pd.DataFrame(rows)
        try:
            df = df.sort_values(["is_best", "RMSE_holdout"], ascending=[False, True])
        except Exception:
            pass

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Models", index=False)
            if training_matrix is not None:
                training_matrix.to_excel(writer, sheet_name="TrainingPreds", index=False)
            if testing_matrix is not None:
                testing_matrix.to_excel(writer, sheet_name="TestingPreds", index=False)

        self.models_summary_excel_path = excel_path
        return excel_path


    def export_combined_excel(self, periods: int = 4, output_dir: str = "./reports"):
        import os
        import pandas as pd
        from datetime import datetime

        if self.data is None or self.train_data is None or self.test_data is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"forecast_report_{ts}.xlsx")

        y_full = self.data[self.demand_col].astype(float)
        train_idx = self.train_data["y"].index
        test_idx = self.test_data["y"].index

        last_date = self.data.index[-1]
        try:
            future_idx = pd.date_range(start=last_date, periods=int(periods) + 1, freq=self.freq)[1:]
        except Exception:
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(periods), freq="D")

        all_idx = train_idx.union(test_idx).union(future_idx)
        df_out = pd.DataFrame(index=all_idx)
        df_out["Actual"] = y_full

        df_out["Set"] = ""
        df_out.loc[train_idx, "Set"] = "Train"
        df_out.loc[test_idx, "Set"] = "Test"
        df_out.loc[future_idx, "Set"] = "Future"

        best_model = getattr(self, "best_model", None)
        if best_model:
            fitted = None
            if (self.baseline_models or {}).get(best_model) is not None:
                fitted = self.baseline_models[best_model].get("fitted")
            elif (self.advanced_models or {}).get(best_model) is not None:
                fitted = self.advanced_models[best_model].get("fitted")
            
            if fitted is not None:
                n = min(len(train_idx), len(fitted))
                df_out.loc[train_idx[:n], f"Fitted ({best_model})"] = fitted[:n]

            preds = None
            if (self.baseline_models or {}).get(best_model) is not None:
                preds = self.baseline_models[best_model].get("predictions")
            elif (self.advanced_models or {}).get(best_model) is not None:
                preds = self.advanced_models[best_model].get("predictions")
            
            if preds is not None:
                n = min(len(test_idx), len(preds))
                df_out.loc[test_idx[:n], f"Test_Predicted ({best_model})"] = preds[:n]
        
        fc_res = getattr(self, "forecast_results", None)
        if fc_res and fc_res.get("forecast"):
            fc = fc_res["forecast"]
            n = min(len(future_idx), len(fc))
            df_out.loc[future_idx[:n], f"Future_Forecast ({fc_res.get('model', 'auto')})"] = fc[:n]
            if fc_res.get("lower_bound"):
                df_out.loc[future_idx[:n], "Future_Lower_Bound"] = fc_res["lower_bound"][:n]
            if fc_res.get("upper_bound"):
                df_out.loc[future_idx[:n], "Future_Upper_Bound"] = fc_res["upper_bound"][:n]

        try:
            df_out.reset_index(names=[self.time_col]).to_excel(filepath, index=False)
            print(f"Combined excel report generated at {filepath}")
        except Exception as e:
            print(f"Failed to generate combined excel report: {e}")

    def export_method_folders(self, periods: int = 4, output_root: str = "./reports/method_exports", include_png: bool = True):
        import os
        import re
        import numpy as np
        import pandas as pd
        from datetime import datetime

        if self.data is None or self.train_data is None or self.test_data is None:
            raise ValueError("请先加载数据并完成数据划分")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(output_root, ts)
        os.makedirs(root, exist_ok=True)

        y_full = self.data[self.demand_col].astype(float)
        train_idx = self.train_data["y"].index
        test_idx = self.test_data["y"].index

        def _safe_name(name: str) -> str:
            s = str(name or "").strip()
            if not s:
                return "unknown"
            s = re.sub(r"[^\w\-\.\(\)\s]+", "_", s, flags=re.UNICODE)
            s = re.sub(r"\s+", "_", s).strip("_")
            return s[:80] if len(s) > 80 else s

        def _aligned_series(values, idx_values, idx_target):
            if values is None or idx_values is None or idx_target is None:
                return [None] * int(len(idx_target))
            try:
                v = list(values)
                idxv = list(idx_values)
            except Exception:
                return [None] * int(len(idx_target))
            n = min(len(v), len(idxv))
            m = {}
            for i in range(n):
                try:
                    x = float(v[i])
                    if pd.notna(x) and float("inf") != abs(x):
                        m[idxv[i]] = x
                    else:
                        m[idxv[i]] = None
                except Exception:
                    m[idxv[i]] = None
            return [m.get(t, None) for t in idx_target]

        def _future_forecast_for_model(model_key: str):
            mk = str(model_key)
            last_date = self.data.index[-1]
            try:
                future_idx = pd.date_range(start=last_date, periods=int(periods) + 1, freq=self.freq)[1:]
            except Exception:
                future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(periods), freq="D")

            if mk in {"naive", "seasonal_naive", "moving_average"}:
                sigma = float(y_full.std())
                if mk == "naive":
                    fit = y_full.shift(1).values
                    resid = np.asarray(y_full.values, dtype=float) - np.asarray(fit, dtype=float)
                    resid = resid[np.isfinite(resid)]
                    if resid.size > 1:
                        sigma = float(np.std(resid))
                    fc = [float(y_full.iloc[-1])] * int(periods)
                    z = 1.96
                    return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}

                if mk == "seasonal_naive":
                    sp = 1
                    try:
                        sp = int(self.baseline_models.get("seasonal_naive", {}).get("params", {}).get("seasonal_period", 1))
                    except Exception:
                        sp = 1
                    sp = int(max(sp, 1))
                    fit = y_full.shift(sp).values
                    resid = np.asarray(y_full.values, dtype=float) - np.asarray(fit, dtype=float)
                    resid = resid[np.isfinite(resid)]
                    if resid.size > 1:
                        sigma = float(np.std(resid))
                    fc = [float(y_full.iloc[-sp + (i % sp)]) for i in range(int(periods))]
                    z = 1.96
                    return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}

                window = None
                try:
                    window = int(self.baseline_models.get("moving_average", {}).get("params", {}).get("window"))
                except Exception:
                    window = None
                if not window:
                    window = int(min(14, max(3, len(y_full) // 10)))
                fit = y_full.rolling(window=window).mean().shift(1).values
                resid = np.asarray(y_full.values, dtype=float) - np.asarray(fit, dtype=float)
                resid = resid[np.isfinite(resid)]
                if resid.size > 1:
                    sigma = float(np.std(resid))
                history = [float(v) for v in y_full.values]
                fc = []
                for _ in range(int(periods)):
                    tail = history[-int(window):] if len(history) >= int(window) else history
                    pred = float(np.mean(tail))
                    fc.append(pred)
                    history.append(pred)
                z = 1.96
                return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}

            if mk == "arima":
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    params = (self.baseline_models or {}).get("arima", {}).get("params", {}) or {}
                    order = tuple(params.get("order") or (1, 1, 1))
                    seasonal_order = tuple(params.get("seasonal_order") or (0, 0, 0, 0))
                    model = SARIMAX(
                        y_full,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    fc = fitted.forecast(int(periods)).astype(float).tolist()
                    resid = (y_full - fitted.fittedvalues).dropna().astype(float).values
                    resid = resid[np.isfinite(resid)]
                    sigma = float(np.std(resid)) if resid.size > 1 else float(y_full.std())
                    z = 1.96
                    return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}
                except Exception:
                    return None

            if mk.startswith("ets"):
                try:
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing

                    seas = getattr(self, "ts_analysis_results", {}) or {}
                    seas = seas.get("seasonality") or {}
                    sp = None
                    try:
                        if seas.get("has_seasonality") and seas.get("suggested_periods"):
                            sp = int(seas.get("suggested_periods")[0])
                    except Exception:
                        sp = None
                    if not sp:
                        sp = 7 if self.freq == "D" else 52 if self.freq == "W" else 12 if self.freq == "M" else 4 if self.freq == "Q" else None

                    trend = None
                    seasonal = None
                    seasonal_periods = None
                    if mk in {"ets_holt"}:
                        trend = "add"
                    if mk in {"ets_holt_winters"}:
                        trend = "add"
                        seasonal = "add"
                        seasonal_periods = int(sp) if sp and int(sp) > 1 else None
                        if seasonal_periods is not None and len(y_full) < 2 * seasonal_periods:
                            seasonal = None
                            seasonal_periods = None

                    model = ExponentialSmoothing(y_full, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                    fitted = model.fit(optimized=True)
                    fc = fitted.forecast(int(periods)).astype(float).tolist()
                    resid = (y_full - fitted.fittedvalues).dropna().astype(float).values
                    resid = resid[np.isfinite(resid)]
                    sigma = float(np.std(resid)) if resid.size > 1 else float(y_full.std())
                    z = 1.96
                    return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}
                except Exception:
                    return None

            if (self.advanced_models or {}).get(mk) is not None:
                try:
                    if getattr(self, "feature_data", None) is None or self.feature_data.get("X") is None or self.feature_data.get("y") is None:
                        raise ValueError("feature_data not found")
                    X_full = self.feature_data["X"]
                    y_feat = self.feature_data["y"].astype(float)
                    model_obj = (self.advanced_models or {}).get(mk, {}).get("model")
                    if model_obj is None:
                        raise ValueError("model_obj not found")
                    model_obj.fit(X_full, y_feat)

                    fitted = np.asarray(model_obj.predict(X_full), dtype=float)
                    resid = np.asarray(y_feat.values, dtype=float) - fitted
                    resid = resid[np.isfinite(resid)]
                    sigma = float(np.std(resid)) if resid.size > 1 else float(y_full.std())

                    X_cols = list(X_full.columns)
                    lag_cols = [c for c in X_cols if isinstance(c, str) and c.startswith("lag_")]
                    try:
                        n_lags = max(int(str(c).split("_", 1)[1]) for c in lag_cols) if lag_cols else 0
                    except Exception:
                        n_lags = 0

                    if n_lags <= 0:
                        X_last = X_full.iloc[[-1]]
                        y_next = float(model_obj.predict(X_last)[0])
                        fc = [y_next for _ in range(int(periods))]
                    else:
                        history_vals = [float(v) for v in y_full.iloc[-n_lags:].values]
                        last_known = {}
                        for c in X_cols:
                            if isinstance(c, str) and c.startswith("lag_"):
                                continue
                            try:
                                raw_val = self.data[c].iloc[-1]
                                if isinstance(raw_val, pd.Series):
                                    raw_val = raw_val.iloc[0] if len(raw_val) > 0 else 0.0
                                last_known[c] = float(raw_val)
                            except Exception:
                                last_known[c] = 0.0

                        fc = []
                        # 对于时间特征，我们也需要递推
                        current_date = pd.to_datetime(last_date)
                        for _ in range(int(periods)):
                            if self.freq == "D":
                                current_date += pd.Timedelta(days=1)
                            elif self.freq == "W":
                                current_date += pd.Timedelta(weeks=1)
                            elif self.freq == "M":
                                current_date += pd.DateOffset(months=1)
                            elif self.freq == "Q":
                                current_date += pd.DateOffset(months=3)
                            else:
                                current_date += pd.Timedelta(days=1)

                            row = {f"lag_{i}": history_vals[-i] for i in range(1, n_lags + 1)}
                            row.update(last_known)
                            
                            # 更新时间相关特征
                            if "dayofweek" in X_cols:
                                row["dayofweek"] = float(current_date.dayofweek)
                            if "month" in X_cols:
                                row["month"] = float(current_date.month)
                            if "quarter" in X_cols:
                                row["quarter"] = float(current_date.quarter)
                            if "year" in X_cols:
                                row["year"] = float(current_date.year)

                            # 确保所有特征在X_cols中的顺序正确，并且类型正确
                            ordered_row = {c: float(row.get(c, 0.0)) for c in X_cols}
                            X_row = pd.DataFrame([ordered_row])
                            y_next = float(model_obj.predict(X_row)[0])
                            fc.append(y_next)
                            history_vals.append(y_next)

                    z = 1.96
                    return {"dates": [d.to_pydatetime() for d in future_idx], "forecast": fc, "lower_bound": [p - z * sigma for p in fc], "upper_bound": [p + z * sigma for p in fc]}
                except Exception as e:
                    print(f"Error in future forecast for {mk}: {e}")
                    raise

            raise ValueError(f"Method {mk} not recognized or failed to forecast")

        model_keys = []
        try:
            if self.evaluation_results is not None and "model" in self.evaluation_results.columns:
                model_keys = [str(x) for x in self.evaluation_results["model"].tolist()]
        except Exception:
            model_keys = []
        if not model_keys:
            model_keys = sorted(set(list((self.baseline_models or {}).keys()) + list((self.advanced_models or {}).keys())))

        for mk in model_keys:
            folder = os.path.join(root, _safe_name(mk))
            os.makedirs(folder, exist_ok=True)

            fit = None
            fit_idx = None
            try:
                if (self.baseline_models or {}).get(mk) is not None:
                    fit = (self.baseline_models or {}).get(mk, {}).get("fitted")
                    fit_idx = train_idx
                elif (self.advanced_models or {}).get(mk) is not None:
                    fit = (self.advanced_models or {}).get(mk, {}).get("fitted")
                    if hasattr(self, "train_features") and self.train_features is not None and self.train_features.get("y") is not None:
                        fit_idx = self.train_features["y"].index
                    else:
                        fit_idx = train_idx
            except Exception:
                fit = None
                fit_idx = train_idx

            preds = None
            pred_idx = None
            try:
                if (self.baseline_models or {}).get(mk) is not None:
                    preds = (self.baseline_models or {}).get(mk, {}).get("predictions")
                    pred_idx = test_idx
                elif (self.advanced_models or {}).get(mk) is not None:
                    preds = (self.advanced_models or {}).get(mk, {}).get("predictions")
                    if hasattr(self, "test_features") and self.test_features is not None and self.test_features.get("y") is not None:
                        pred_idx = self.test_features["y"].index
                    else:
                        pred_idx = test_idx
            except Exception:
                preds = None
                pred_idx = test_idx

            fitted_aligned = _aligned_series(fit, fit_idx, train_idx)
            preds_aligned = _aligned_series(preds, pred_idx, test_idx)
            future = _future_forecast_for_model(mk)

            train_df = pd.DataFrame({"Date": pd.to_datetime(train_idx), "Actual": self.train_data["y"].astype(float).values, "Fitted": fitted_aligned})
            test_df = pd.DataFrame({"Date": pd.to_datetime(test_idx), "Actual": self.test_data["y"].astype(float).values, "Predicted": preds_aligned})

            full_idx = self.data.index
            fitted_full = _aligned_series(fit, fit_idx, full_idx)
            preds_full = _aligned_series(preds, pred_idx, full_idx)
            backtest_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(full_idx),
                    "Actual": y_full.astype(float).values,
                    "Fitted_Train": [np.nan if v is None else float(v) for v in fitted_full],
                    "Predicted_Test": [np.nan if v is None else float(v) for v in preds_full],
                }
            )

            future_df = pd.DataFrame()
            if future is not None:
                future_df = pd.DataFrame(
                    {
                        "Date": pd.to_datetime(future.get("dates") or []),
                        "Forecast": future.get("forecast") or [],
                        "Lower_95": future.get("lower_bound") or [],
                        "Upper_95": future.get("upper_bound") or [],
                    }
                )

            params_rows = []
            try:
                detail = (getattr(self, "model_results", {}) or {}).get(str(mk), {}) or {}
                params_obj = detail.get("params")
                mtype = detail.get("type")
                params_rows.append({"key": "model", "value": str(mk)})
                if mtype:
                    params_rows.append({"key": "type", "value": str(mtype)})

                if isinstance(params_obj, dict):
                    if str(mk) == "moving_average":
                        if params_obj.get("window") is not None:
                            params_rows.append({"key": "window", "value": str(params_obj.get("window"))})
                        if params_obj.get("selection") is not None:
                            params_rows.append({"key": "selection", "value": str(params_obj.get("selection"))})
                    elif str(mk) == "seasonal_naive":
                        if params_obj.get("seasonal_period") is not None:
                            params_rows.append({"key": "seasonal_period", "value": str(params_obj.get("seasonal_period"))})
                    elif str(mk) == "naive":
                        if params_obj.get("strategy") is not None:
                            params_rows.append({"key": "strategy", "value": str(params_obj.get("strategy"))})
                    elif str(mk).startswith("ets"):
                        cfg = params_obj.get("config") if isinstance(params_obj.get("config"), dict) else {}
                        fp = params_obj.get("fitted_params") if isinstance(params_obj.get("fitted_params"), dict) else {}
                        if cfg:
                            if cfg.get("trend") is not None:
                                params_rows.append({"key": "trend", "value": str(cfg.get("trend"))})
                            if cfg.get("seasonal") is not None:
                                params_rows.append({"key": "seasonal", "value": str(cfg.get("seasonal"))})
                            if cfg.get("seasonal_periods") is not None:
                                params_rows.append({"key": "seasonal_periods", "value": str(cfg.get("seasonal_periods"))})
                        alpha = fp.get("smoothing_level")
                        beta = fp.get("smoothing_trend")
                        gamma = fp.get("smoothing_seasonal")
                        if alpha is not None:
                            params_rows.append({"key": "alpha", "value": str(alpha)})
                        if beta is not None:
                            params_rows.append({"key": "beta", "value": str(beta)})
                        if gamma is not None:
                            params_rows.append({"key": "gamma", "value": str(gamma)})
                        if alpha is None and beta is None and gamma is None:
                            try:
                                res_obj = (self.baseline_models or {}).get(str(mk), {}).get("model")
                                p2 = getattr(res_obj, "params", None)
                                if hasattr(p2, "to_dict"):
                                    p2 = p2.to_dict()
                                if isinstance(p2, dict):
                                    if p2.get("smoothing_level") is not None:
                                        params_rows.append({"key": "alpha", "value": str(p2.get("smoothing_level"))})
                                    if p2.get("smoothing_trend") is not None:
                                        params_rows.append({"key": "beta", "value": str(p2.get("smoothing_trend"))})
                                    if p2.get("smoothing_seasonal") is not None:
                                        params_rows.append({"key": "gamma", "value": str(p2.get("smoothing_seasonal"))})
                                    for k in ["initial_level", "initial_trend"]:
                                        if p2.get(k) is not None:
                                            params_rows.append({"key": k, "value": str(p2.get(k))})
                                    if p2.get("initial_seasons") is not None:
                                        try:
                                            arr = list(p2.get("initial_seasons"))
                                            params_rows.append({"key": "initial_seasons_head", "value": str(arr[: min(6, len(arr))])})
                                        except Exception:
                                            params_rows.append({"key": "initial_seasons", "value": str(p2.get("initial_seasons"))})
                            except Exception:
                                pass
                        for k in ["initial_level", "initial_trend"]:
                            if fp.get(k) is not None:
                                params_rows.append({"key": k, "value": str(fp.get(k))})
                        if fp.get("initial_seasons") is not None:
                            try:
                                arr = list(fp.get("initial_seasons"))
                                params_rows.append({"key": "initial_seasons_head", "value": str(arr[: min(6, len(arr))])})
                            except Exception:
                                params_rows.append({"key": "initial_seasons", "value": str(fp.get("initial_seasons"))})

                        first_fit = None
                        first_fit_date = None
                        try:
                            for i in range(min(len(fitted_aligned), len(train_idx))):
                                v = fitted_aligned[i]
                                if v is None:
                                    continue
                                if not np.isfinite(float(v)):
                                    continue
                                first_fit = float(v)
                                first_fit_date = str(pd.to_datetime(train_idx[i]).to_pydatetime())
                                break
                        except Exception:
                            first_fit = None
                            first_fit_date = None
                        if first_fit is not None:
                            params_rows.append({"key": "first_fitted_value", "value": str(first_fit)})
                        if first_fit_date is not None:
                            params_rows.append({"key": "first_fitted_date", "value": str(first_fit_date)})
                    elif str(mk) == "arima":
                        for k in ["order", "seasonal_order", "aic", "selection"]:
                            if params_obj.get(k) is not None:
                                params_rows.append({"key": k, "value": str(params_obj.get(k))})
                    else:
                        for k in ["alpha", "n_estimators", "max_depth", "learning_rate", "random_state"]:
                            if params_obj.get(k) is not None:
                                params_rows.append({"key": k, "value": str(params_obj.get(k))})
                        try:
                            import json
                            params_rows.append({"key": "params_json", "value": json.dumps(params_obj, ensure_ascii=False, default=str)})
                        except Exception:
                            params_rows.append({"key": "params_str", "value": str(params_obj)})
            except Exception:
                params_rows = params_rows or [{"key": "model", "value": str(mk)}]

            params_df = pd.DataFrame(params_rows) if params_rows else pd.DataFrame({"key": ["model"], "value": [str(mk)]})

            excel_path = os.path.join(folder, "forecast.xlsx")
            try:
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    params_df.to_excel(writer, sheet_name="Params", index=False)
                    train_df.to_excel(writer, sheet_name="Training", index=False)
                    test_df.to_excel(writer, sheet_name="Testing", index=False)
                    backtest_df.to_excel(writer, sheet_name="Backtest", index=False)
                    if not future_df.empty:
                        future_df.to_excel(writer, sheet_name="Future", index=False)
            except Exception as e:
                print(f"方法 {mk} 导出Excel失败: {e}")

            if include_png:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(10, 4))
                    ax = fig.add_subplot(111)
                    ax.plot(self.data.index, y_full.values, color="#15222A", linewidth=2, label="Actual")
                    if any(v is not None for v in fitted_full):
                        ax.plot(self.data.index, [np.nan if v is None else float(v) for v in fitted_full], color="#8898AA", linewidth=2, linestyle="--", label="Fitted (Train)")
                    if any(v is not None for v in preds_full):
                        ax.plot(self.data.index, [np.nan if v is None else float(v) for v in preds_full], color="#6C43FF", linewidth=2.5, label="Predicted (Test)")
                    try:
                        if len(train_idx) > 0:
                            ax.axvline(pd.to_datetime(train_idx[-1]), color="rgba(16,50,66,0.35)", linestyle="--", linewidth=1.2, label="Train/Test Split")
                    except Exception:
                        pass
                    try:
                        if len(test_idx) > 0:
                            ax.axvline(pd.to_datetime(test_idx[-1]), color="rgba(108,67,255,0.55)", linestyle="-", linewidth=1.5, label="Test/Future Split")
                    except Exception:
                        pass

                    try:
                        if future is not None and future_df is not None and not future_df.empty:
                            fdates = pd.to_datetime(future_df["Date"])
                            fvals = pd.to_numeric(future_df["Forecast"], errors="coerce").astype(float).values
                            flo = pd.to_numeric(future_df["Lower_95"], errors="coerce").astype(float).values
                            fup = pd.to_numeric(future_df["Upper_95"], errors="coerce").astype(float).values
                            ax.plot(fdates, fvals, color="#6C43FF", linewidth=2.6, linestyle="-", label="Forecast (Future)")
                            ax.fill_between(fdates, flo, fup, color="#6C43FF", alpha=0.14, linewidth=0)
                    except Exception:
                        pass

                    ax.set_title(f"{mk} backtest + future (95% PI)")
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(os.path.join(folder, "chart.png"), dpi=160)
                    plt.close(fig)
                except Exception as e:
                    print(f"方法 {mk} 导出PNG失败: {e}")

        return root

    def run_full_pipeline(self, data_path: str):
        """
        运行完整预测管道

        参数:
            data_path: 数据文件路径
        """
        print("=" * 80)
        print("FORECASTPRO AI需求预测管道启动")
        print("=" * 80)

        # Step 1: 数据摄取与画像
        print("\n📊 Step 1: 数据摄取与画像")
        self.load_data(data_path)

        # Step 2: 基准模型
        print("\n📈 Step 2: 建立时间序列基准")
        self.prepare_data()
        self.run_baseline_models()

        # Step 3: 高级模型
        print("\n🤖 Step 3: 高级模型探索与选择")
        self.run_advanced_models()

        # Step 4: 模型评估
        print("\n📊 Step 4: 模型评估与诊断")
        self.evaluate_models()

        # Step 5: 生成预测
        print("\n🔮 Step 5: 生成未来预测")
        self.generate_forecast(periods=4)

        # Step 6: 管理报告
        print("\n📋 Step 6: 生成管理报告")
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("FORECASTPRO 预测管道执行完成!")
        print("=" * 80)

        return report
