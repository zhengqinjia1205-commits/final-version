# Fitting Strategies（拟合口径与数学细节）

本文定义 ForecastPro 在“回测评估（train/test）”与“未来预测（future）”两个阶段的严格口径，确保可复现实验、可检查落盘结果，并满足 **No Data Leakage**。

## 1. 符号与数据切分

设原始时间序列为 \(\{y_t\}_{t=1}^T\)，时间索引单调递增。

- 训练集：\(t = 1,\dots,T_{train}\)，其中 \(T_{train}=\lfloor 0.8T \rfloor\)
- 测试集：\(t=T_{train}+1,\dots,T\)（最后 20%）

**评估只在测试集上进行**，即使用真实 \(y_t\) 与测试期预测 \(\hat y_t\) 计算误差；未来期 \(t>T\) 没有真实值，不参与误差评估。

## 2. 无数据泄露（No Data Leakage）

测试期预测必须满足：

- 预测 \(t\) 时 **禁止** 使用任何测试集真实值 \(y_{t-1}\) 等信息参与：
  - 特征构造（例如滞后项）
  - 窗口滚动（例如移动平均）
  - 状态更新（例如递推模型）
  - 参数更新（再拟合）
- 测试期允许使用的信息仅包括：
  - 训练集历史 \(\{y_t\}_{t \le T_{train}}\)
  - 模型自身在测试期已产生的递推预测 \(\hat y_t\)

未来预测（future\_forecast）遵循默认口径：**先完成测试评估**，再用全量 \(y_{1:T}\) 重新拟合（或更新）模型，然后预测 \(h\) 期未来。

## 3. 数据预处理

### 3.1 缺失值

对每个数值列 \(x_t\)：

- 若缺失率 \(< 5\%\)：线性插值（时间索引为日期时优先 time-interpolation）
- 若缺失率 \(\ge 5\%\)：LOCF（last observation carried forward）
- 边界仍缺失时：再用 backward fill 补齐

### 3.2 异常值（IQR 标记，不删除）

仅对目标列 \(y_t\) 做 IQR 标记：

- \(Q_1 = \mathrm{quantile}_{0.25}(y)\)，\(Q_3 = \mathrm{quantile}_{0.75}(y)\)
- \(IQR = Q_3 - Q_1\)
- 下界：\(L = Q_1 - 1.5\,IQR\)，上界：\(U = Q_3 + 1.5\,IQR\)
- 若 \(y_t < L\) 或 \(y_t > U\) 则标记为异常值

默认不删除异常值，仅用于质量报告与人工核对。

## 4. Baseline 模型（回测口径）

### 4.1 Naïve

一步预测：
\[
\hat y_t = y_{t-1}
\]

- 训练期拟合（Training/Fitted）：`shift(1)`
- 测试期预测（Testing/Predicted）：从训练末值开始递推（常数延续）

### 4.2 Seasonal Naïve

设季节周期为 \(m\)：
\[
\hat y_t = y_{t-m}
\]

- 训练期拟合：`shift(m)`
- 测试期预测：仅使用训练集中最后一个季节模板循环递推（不使用测试真实值）

### 4.3 Moving Average（递推版，避免泄露）

设窗口为 \(w\)，测试期的递推预测定义为：
\[
\hat y_{t} = \frac{1}{w}\sum_{i=1}^{w} z_{t-i}
\]
其中
\[
z_{t-i}=
\begin{cases}
y_{t-i}, & t-i \le T_{train} \\\\
\hat y_{t-i}, & t-i > T_{train}
\end{cases}
\]

即：进入测试期后，窗口滚动只允许使用训练历史与已产生的预测值。

窗口 \(w\) 的选择：在训练集末端留出一段 validation（不随机打乱），比较 validation MAE，取最优窗口。

### 4.4 ETS（Holt-Winters）

使用 `statsmodels.tsa.holtwinters.ExponentialSmoothing`，通过极大似然拟合平滑参数：

- level smoothing：\(\alpha\)
- trend smoothing（若有）：\(\beta\)
- seasonal smoothing（若有）：\(\gamma\)

回测阶段：在训练集拟合后调用 `forecast(len(test))` 得到测试期预测，不在测试期用真实值更新参数或状态。

### 4.5 ARIMA / SARIMA（AIC 搜索）

在非 fast 模式下使用 AIC 网格搜索：

- \(p \in [0,5]\)
- \(d \in [0,2]\)
- \(q \in [0,5]\)

若检测到季节性且数据量满足至少两个季节周期，则启用 SARIMA：

- seasonal order \((P,D,Q,m)\) 亦做有限集合搜索（并同样以 AIC 选优）

回测阶段：训练集拟合后 `forecast(len(test))` 得到测试期预测。

## 5. 高级模型（回测口径）

高级模型使用滞后特征（例如 `lag_1 ... lag_k`）与可选协变量。关键约束：

- 特征只能使用 \(t-1\) 及以前的信息
- CV 使用 `TimeSeriesSplit(n_splits=3)`（不允许随机洗牌）
- 测试期预测若依赖 `lag_*`，则必须递推构造 lag：进入测试期后 lag 的来源是模型自身预测 \(\hat y_t\)，而不是测试真实值 \(y_t\)

## 6. 指标（Train / Hold-out）

对测试集（hold-out）：

- MAE：
\[
\mathrm{MAE}=\frac{1}{n}\sum_{t}( |y_t-\hat y_t|)
\]
- RMSE：
\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{t}(y_t-\hat y_t)^2}
\]
- MAPE：
\[
\mathrm{MAPE}= \frac{100\%}{n}\sum_{t}\left|\frac{y_t-\hat y_t}{y_t}\right|
\]
当 \(y_t=0\) 或出现非有限值时，按实现做保护处理，避免除零与 NaN 传播。

训练集（in-sample）同理，但使用训练期 fitted 值对齐计算。

## 7. 模型选择（Winner Decision Logic）

最终裁决以 **hold-out MAPE 最小**（若 MAPE 不可用则回退到 RMSE 最小）为核心。

当存在高级模型时：

- **直接对比测试集 MAPE (或 RMSE)**，选取误差最小的模型作为最佳模型。
- 模型在训练集与测试集上的表现差异（过拟合风险）仍将被计算并输出到表格（如：高/中/低），以作人工诊断参考。
- 残差 Ljung-Box 白噪声检验亦仅作为附加的诊断信息报告。
- 不再强制要求高级模型必须在误差上显著（>5%）超越基线模型且残差为白噪声。只要在测试集上表现更优，即选为最佳模型。

当样本量 \(n<30\)：为了模型稳健性，仍可能跳过高级模型，只在 ETS/Naïve 等基线模型中择优。

## 8. 预测区间（95% PI）

采用正态近似：
\[
\hat y_{T+h} \pm 1.96\,\sigma_{resid}
\]
其中 \(\sigma_{resid}\) 为训练期残差的标准差（或可用残差估计）。

## 9. 落盘检查（method_exports）

每次预测后，后端会在：

`reports/method_exports/<timestamp>/<method>/forecast.xlsx`

写入以下 sheets：

- `Params`：模型关键参数（ETS 的 alpha/beta/gamma 与初始状态；MA 的 window 等）
- `Training`：训练集 Actual + Fitted
- `Testing`：测试集 Actual + Predicted（严格无泄露递推口径）
- `Future`：未来期 Forecast + 95% PI

同目录下还会生成 `chart.png`（用于快速目视检查）。

