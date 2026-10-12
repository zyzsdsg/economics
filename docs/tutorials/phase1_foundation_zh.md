# Phase 1 教程：代码是如何工作的

本文按 "数据流" 的顺序讲解 Phase 1 的主要代码：每个函数做什么、输入输出的类型与形状、内部的计算步骤、
调用了哪些外部库、以及可能的数值或性能瓶颈。数学推导在 notebook 04/05 中，这里侧重 **理论 → 数学 → 代码** 的对应关系。

---

## 1. 数据流总览

```text
SOURCES (sources.py)
   │  scripts/data_download.py
   ▼
download_data()  ──►  raw file + manifest.json (provenance.py)
   │  readers.py
   ▼
pd.Series / pd.DataFrame (DatetimeIndex "date", attrs: units, source)
   │  build.py: 清洗 → validate_series() → 频率转换 (transforms.py)
   ▼
data/processed/macro_{daily,monthly,quarterly}.parquet + .meta.json
   │  load_dataset()
   ▼
notebooks / 模型：pct_change, log_diff → ols(), adf_test(), fit_ardl(), fit_uecm()
```

设计原则：**每一层只做一件事**。reader 只解析格式，不做经济判断；清洗规则集中在 `build.py` 并写入元数据；
parquet 只存水平值，变化率在模型代码里算。这样任何一个数字都能追溯到原始文件的 SHA-256。

---

## 2. 数据层

### 2.1 `downloader.download_data(url, file_path, *, overwrite, timeout, manifest, source_name) -> Path`

保留了你原来的签名和 "文件已存在就跳过" 的行为，新增：

| 步骤 | 代码 | 为什么 |
|---|---|---|
| 流式下载到 `.part` | `requests.get(..., stream=True)`，`iter_content(65536)` | 大文件不一次性读入内存；中断时不会留下 "看起来完整" 的文件 |
| 检查状态码 | `response.raise_for_status()` | RBNZ 返回 403 时，原版本会把 HTML 错误页保存成 `.xlsx` |
| 检查文件头 | `check_file_format(tmp, suffix)`：`.xlsx` 必须以 `PK\x03\x04` 开头（zip），`.xls` 以 `D0 CF 11 E0` 开头（OLE2） | 有的服务器出错时仍返回 200 |
| 原子替换 | `tmp.replace(file_path)` | 同一文件系统内的 rename 是原子的：要么是旧文件，要么是完整的新文件 |
| 保留旧版本 | `_archive_vintage()` 移到 `_vintages/name_YYYYMMDDTHHMMSSZ.ext` | 官方数据会修订；旧结果需要旧数据才能复现 |
| 记录溯源 | `record_file(manifest, ...)` | URL、时间、SHA-256、`ETag`/`Last-Modified` |

`try/except Exception: tmp.unlink(); raise` 保证任何失败都会清理临时文件，然后把异常继续抛出 —— 不吞掉错误。

### 2.2 `provenance.sha256_file(path) -> str`

按 1 MiB 分块读取并更新哈希（`iter(lambda: f.read(n), b"")` 是 "读到空字节串为止" 的惯用法），
所以内存占用与文件大小无关。SHA-256 改变 ⇔ 文件内容改变。

### 2.3 `readers.py`

所有 reader 返回 **DatetimeIndex 名为 `date`** 的 pandas 对象，并把单位写进 `.attrs`。

* `read_eia_xls`：EIA 文件有两行前言，`skiprows=2` 后前两列是 `Date | value`。需要 `xlrd`（旧 `.xls` 格式）。
* `read_fred_csv`：`na_values=[".", ""]` 处理两种缺失标记；`pd.to_numeric(errors="coerce")` 把剩下的非数字变 NaN。
* `read_dallasfed_igrea`：`dt.to_period("M").dt.to_timestamp()` 把任意日期映射到月初；如果映射后出现重复月份就报错，而不是悄悄合并。
* `read_worldbank_pinksheet`：用正则 `\d{4}M\d{2}` 找到第一行数据，它上面两行分别是商品名和单位。`"…"` 替换为 NaN。
* `read_rbnz_xlsx -> (data, metadata)`：在第一列搜索 `Series Id`，而不是硬编码行号。
  `np.flatnonzero(labels == "series id")` 返回所有匹配行的位置。**注意**：这个 reader 只用按文档结构构造的合成文件测试过，
  第一次读取真实 RBNZ 文件时请检查 `metadata` 输出。

### 2.4 `transforms.py`：单位的 "唯一真相"

| 函数 | 输入 → 输出 | 计算 |
|---|---|---|
| `convert_currency(price_usd, usd_per_nzd)` | 两个 Series → Series | `align(join="inner")` 后相除；汇率 ≤ 0 时报错 |
| `aggregate(series, freq, how, min_coverage=0.8)` | 日度 Series → 月/季 Series | `groupby(index.to_period(freq))`，再与该期工作日数 `len(pd.bdate_range(start, end))` 比较 |
| `pct_change(x, k, freq)` | Series → Series（%） | `100 * (x / x.shift(k) - 1)` |
| `log_diff(x, k)` | Series → Series（对数点） | `100 * np.log(x).diff(k)` |
| `pp_change(rate, k)` | Series → Series（百分点） | `rate - rate.shift(k)` |
| `require_regular(index, freq)` | 索引 → None 或 ValueError | 把索引转为 `PeriodIndex`，与 `period_range(min, max)` 比较 |

**为什么 `require_regular` 很重要？** `shift(4)` 移动的是 *行*，不是 *日期*。如果中间缺了一个季度，
`x.shift(4)` 实际比较的是 5 个季度之前的值，同比通胀就错了，而且不会有任何报错。传入 `freq=` 时函数会先检查。

性能：`aggregate` 中对每个周期调用一次 `bdate_range`，对 40 年季度数据（~160 次）可以忽略；如果用于几十年的分钟级数据就需要向量化。

### 2.5 `validation.validate_series(...) -> ValidationReport`

`ValidationReport` 是一个 dataclass，`errors` 与 `warnings` 分开：错误会让 `build_from_raw()` 失败（`raise_if_failed()`），
警告只打印。对工作日数据，长缺口和低覆盖率是 **警告**（EIA 数据确实如此，聚合时由覆盖率规则处理）；
对月/季数据，不规则间隔或日期没对齐到期初是 **错误**（它会破坏 shift 与增长率计算）。

`seasonality_test` 就是一个小 OLS：`X = [1, D2, D3, D4]`，比较受限（只有常数）与非受限模型的残差平方和：
$F = \frac{(RSS_r - RSS_u)/m}{RSS_u/(n-m-1)}$。

### 2.6 `synthetic.py`：已知真值的数据生成过程

`SyntheticParams` 是 `frozen=True` 的 dataclass（不可变，可以安全地作为默认参数共享）。
日度层用 `rng.multivariate_normal` 生成相关的油价与汇率冲击（协方差矩阵 `2×2`），季度层用 `for` 循环模拟递归方程
（ARDL 需要上一期的值，无法简单向量化；T ≈ 120，循环成本可忽略）。
tradable 通胀的真实长期传导 = `(b0 + b1) / (1 - rho_T)`，测试用它检查 ARDL 估计量。

### 2.7 `build.py`

`build_from_raw()` 依次：读取 → 校验 → 在日度层换算 NZD → 月度（EIA 月均价 / 月均汇率）→ 季度（三个月都存在才求均值）
→ 合并 IGREA 与 CPI → 对所有报告调用 `raise_if_failed()`。
`write_datasets()` 写 parquet（pyarrow）和 `.meta.json`；`load_dataset()` 读取，或在没有真实数据时回退到合成数据并标记 `data_kind`。

---

## 3. 计量层

### 3.1 `ols.ols(y, X, *, method, cov_type, hac_lags) -> OLSResult`

**输入**：`y` 形状 `(n,)`，`X` 形状 `(n, k)`（DataFrame 时保留列名）。**输出**：`OLSResult`（params、cov_params、resid…）。

计算步骤与数学的对应：

| 数学 | 代码 | 形状 |
|---|---|---|
| 检查 $\operatorname{rank}(X)=k$（假设 A2） | `np.linalg.matrix_rank(Xa)`（内部用 SVD） | 标量 |
| $X = QR$ | `np.linalg.qr(X, mode="reduced")` → LAPACK `geqrf`（Householder 反射） | Q `(n,k)`，R `(k,k)` |
| $R\hat\beta = Q'y$ | `scipy.linalg.solve_triangular(R, Q.T @ y)` 回代，$O(k^2)$ | `(k,)` |
| $\hat u = y - X\hat\beta$ | `ya - Xa @ beta` | `(n,)` |
| $(X'X)^{-1} = R^{-1}R^{-T}$ | `Rinv = solve_triangular(R, I)`；`bread = Rinv @ Rinv.T` | `(k,k)` |
| $s^2(X'X)^{-1}$ | `(ssr / (n-k)) * bread` | `(k,k)` |
| 得分 $g_t = x_t u_t$ | `X * u[:, None]`（广播：每行乘以对应残差） | `(n,k)` |
| $\hat\Gamma_l = \sum_t g_t g_{t-l}'$ | `g[l:].T @ g[:-l]` | `(k,k)` |
| 三明治 | `bread @ S @ bread` | `(k,k)` |

为什么用 $R^{-1}R^{-T}$ 而不是 `inv(X.T @ X)`：因为 $X'X = R'Q'QR = R'R$，直接用已经算好的 R，避免条件数平方。

**计算成本**：QR 为 $O(nk^2)$；HAC 的 meat 为 $O(Lnk^2)$。宏观季度数据 $n\approx130, k<20$，毫秒级。
瓶颈会出现在 Monte Carlo 中（成千上万次回归）—— 那时可以只计算需要的部分（如 notebook 03 中的 `t_stat_slope`）。

**与 statsmodels 的差异**：系数差异在 $10^{-16}$ 量级（statsmodels 用 `pinv`，即 SVD，我们用 QR）；
HAC 默认 **不** 乘以 $n/(n-k)$，与 `statsmodels` 的 `cov_type="HAC"` 默认一致 —— 这是对照测试时发现并修正的。

### 3.2 `ols.delta_method(result, func) -> (value, se)`

用中心差分数值计算梯度：$\partial g/\partial b_i \approx [g(b+h e_i) - g(b-h e_i)]/2h$，步长 `h = 1e-6 * max(1, |b_i|)`。
中心差分的截断误差是 $O(h^2)$，比前向差分的 $O(h)$ 好。然后 $\text{se} = \sqrt{\nabla g' V \nabla g}$。

### 3.3 `timeseries.adf_test(y, lags, trend, maxlag, autolag) -> ADFResult`

1. `_adf_design` 用整数索引构造回归：`dy[t-1]` 是 $\Delta y_t$，`y[t-1]` 是 $y_{t-1}$，`dy[t-1-i]` 是 $\Delta y_{t-i}$。
   第一列是 $y_{t-1}$，所以 `tvalues.iloc[0]` 就是 DF 统计量。
2. 自动选择滞后：所有候选模型从同一个起点 `start = maxlag + 1` 开始（可比的样本），按 AIC/BIC 选最小值。
3. 用选定的滞后在最大可用样本上重新估计。
4. p 值：`statsmodels.tsa.adfvalues.mackinnonp`（MacKinnon 的响应面回归），这是唯一 "借用" 的部分，因为 DF 分布没有解析形式。

### 3.4 `ardl.py`

* `ardl_design(y, x, p, q)`：用 `shift` 构建滞后列，列名 `{var}_L{k}`，再从 `start = max(p, q)` 开始截取，确保不同阶数可比。
* `ARDLResult.long_run()`：对每个 x 定义闭包 `theta(b)`（`sum(b[x_idx]) / (1 - sum(b[ar_idx]))`），交给 `delta_method`。
  注意闭包中的默认参数 `x_idx=x_idx`：Python 闭包是 **晚绑定** 的，不这样写，循环中所有闭包都会使用最后一个 `x_idx`。
* `cumulative_response(x_name, H, params=None)`：迭代差分方程计算阶梯响应；传入 `params` 即可对模拟抽取的参数复用同一函数（notebook 05 的置信带）。
* `uecm_design` / `fit_uecm`：误差修正形式。测试中验证：UECM 与 ARDL 的残差逐元素相同、长期乘数相同、参数与 `statsmodels.UECM` 相同、bounds F 与 `bounds_test().statistic` 相同。

---

## 4. 测试在检验什么

| 测试 | 检验的性质 |
|---|---|
| `test_ols_recovers_true_coefficients` | 已知 DGP 下，估计值在真值的 4 个标准误之内（5 种求解器） |
| `test_ols_matches_statsmodels`、`test_hac_matches_statsmodels` | 与成熟库数值一致（`rtol=1e-10`） |
| `test_qr_more_accurate_than_inverse_when_ill_conditioned` | 数值线性代数的核心结论 |
| `test_adf_matches_statsmodels`、`test_adf_power_and_size` | 统计量一致；平稳序列拒绝、随机游走不拒绝 |
| `test_ardl_recovers_long_run_passthrough` | 合成数据中估计的长期传导在真值的 3 个标准误之内 |
| `test_uecm_is_reparametrisation_of_ardl` | ECM 只是重新参数化 |
| `test_shift_on_gappy_index_is_refused` | 单位/频率安全 |
| `test_download_rejects_http_error_and_html`、`test_download_archives_previous_vintage` | 下载失败不留垃圾文件、旧版本不被覆盖 |
| `test_notebooks.py` | 所有 notebook 在合成数据上端到端运行 |

---

## 5. 你应该能独立回答的面试问题

1. 写出 OLS 的目标函数并推导正规方程。为什么数值计算中不直接求 $(X'X)^{-1}$？
2. 什么情况下非稳健标准误会低估真实的不确定性？Newey-West 估计量的 Bartlett 权重起什么作用？
3. 为什么把两个独立的随机游走做回归会得到 "显著" 的结果？
4. Dickey-Fuller 统计量为什么不服从 t 分布？
5. 在变化率上估计的 ARDL 中，长期乘数 $\theta$ 的经济含义是什么？它的标准误怎么得到？
6. 什么时候应该用 ECM？如果 bounds test 不拒绝，ECM 中的 "长期系数" 还有意义吗？
7. 为什么油价对通胀的 OLS 系数不能解释为 "油价供给冲击的因果效应"？偏差可能来自哪些渠道？
8. 你的数据管道如何保证别人一年后还能复现你的结果？
