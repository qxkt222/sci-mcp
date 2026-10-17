# sci-mcp

本机科学计算 MCP 服务器。把符号数学、物理单位、数值仿真、PDE、统计推断、GLM/时间序列、
信号时频分析、小波、蒙特卡洛、规划求解、高层数据分析和绘图收进一套固定的 MCP 工具里，
供 Cherry Studio 里的 agent 直接调用。

## 工具一览（65 个）

### 符号数学（sympy）
| 工具 | 作用 |
|---|---|
| `symbolic_calculus` | 求导、不定积分、极限、级数展开、化简、展开、因式分解 |
| `symbolic_solve` | 解方程/方程组；单变量返回扁平解列表，多变量返回字典列表 |
| `symbolic_evaluate` | 符号表达式数值代入并求值 |
| `dsolve_ode` | 符号 ODE 解析解（通解或带初值），支持高阶方程 |

### 向量微积分
| 工具 | 作用 |
|---|---|
| `gradient` / `divergence` / `curl` / `laplacian` | 梯度、散度、旋度（三维）、拉普拉斯算子 |

### 物理与单位（scipy.constants + pint）
| 工具 | 作用 |
|---|---|
| `physical_constant` | 查询物理常量 |
| `unit_convert` | 单位换算，支持偏移单位 |
| `unit_calculate` | 计算带单位表达式 |

### 数值仿真（numpy + scipy）
| 工具 | 作用 |
|---|---|
| `solve_ode` | 一阶 ODE 系统数值解 |
| `optimize_minimize` | 局部最小化，支持上下界 + 等式/不等式约束 |
| `global_optimize` | 全局优化：差分进化、模拟退火、basinhopping |
| `matrix_compute` | det/inv/eig/svd/solve/lstsq/matmul/rank/norm/transpose |
| `find_roots` | 数值求根（brentq / newton） |
| `fft_analysis` | 单边幅度谱与主频 |
| `interpolate_1d` | 一维插值 |
| `sparse_compute` | 稀疏矩阵：solve/eigsh/svds/matmul/transpose/norm，支持 Matrix Market、直接法/CG/GMRES |

### PDE（有限差分）
| 工具 | 作用 |
|---|---|
| `solve_heat_1d` | 一维热方程；边界支持 Dirichlet / Neumann / Robin |
| `solve_wave_1d` | 一维波动方程；边界支持 Dirichlet / Neumann / Robin |
| `solve_laplace_2d` | 二维拉普拉斯方程，Dirichlet 边界，Gauss-Seidel |

### PDE 进阶（三维 / 非线性）
| 工具 | 作用 |
|---|---|
| `solve_heat_3d` | 三维热方程，显式 FTCS（稳定性检查） |
| `solve_laplace_3d` | 三维拉普拉斯方程，向量化 Jacobi 迭代 |
| `solve_burgers_1d` | 一维 Burgers 方程（非线性对流 + 扩散） |
| `solve_heat_1d_variable` | 一维变系数热传导 alpha(x) |
| `solve_pde_1d` | 半通用一维显式 PDE：u_t = F(x,t,u,u_x,u_xx) |
| `solve_wave_3d` | 三维波动方程（显式中心差分） |
| `solve_navier_stokes_2d` | 二维不可压涡量-流函数（周期边界） |

### 统计与绘图（scipy.stats + matplotlib）
| 工具 | 作用 |
|---|---|
| `describe` | 描述统计 |
| `distribution` | 概率分布 pdf/cdf/ppf/rvs/fit |
| `curve_fit` | 任意表达式模型最小二乘拟合 + R² |
| `plot` | line/scatter/hist/bar/errorbar/contour/surface |

### 统计推断
| 工具 | 作用 |
|---|---|
| `t_test` | 单样本 / 双样本 / 配对 t 检验 |
| `chi2_test` | 卡方独立性 / 拟合优度 |
| `anova` | 单因素方差分析 |
| `correlation` | Pearson / Spearman / Kendall |
| `linear_regression` | 单变量或多变量 OLS |

### GLM 与时间序列（statsmodels）
| 工具 | 作用 |
|---|---|
| `glm` | 广义线性模型：gaussian/binomial/poisson/gamma 等 |
| `arima` | ARIMA 拟合与多步预测（含置信区间） |
| `time_series_decompose` | 趋势/季节/残差分解 |
| `stationarity_test` | ADF 单位根 / 平稳性检验 |

### 混合模型（statsmodels）
| 工具 | 作用 |
|---|---|
| `mixed_lm` | 线性混合模型：固定效应 + 随机效应，处理分组/重复测量 |
| `gee` | 广义估计方程：gaussian/binomial/poisson × 三种相关工作结构 |
| `nonlinear_mixed_lm` | 非线性混合模型（固定效应 + 随机截距，惩罚最小二乘近似） |

### 生存分析
| 工具 | 作用 |
|---|---|
| `cox_regression` | Cox 比例风险模型，含完全分离/不收敛检测 |
| `kaplan_meier` | Kaplan-Meier 生存曲线与中位生存期，支持分组 |

### 信号处理（scipy.signal）
| 工具 | 作用 |
|---|---|
| `filter_signal` | Butterworth 低通/高通/带通/带阻 |
| `convolve` | 一维卷积 |
| `power_spectrum` | Welch 功率谱密度与峰值频率 |
| `find_peaks` | 峰值检测 |
| `stft` | 短时傅里叶变换，时频矩阵 + 可选频谱图 |

### 小波（PyWavelets）
| 工具 | 作用 |
|---|---|
| `wavelet_transform` | 多层小波分解，返回各层能量与系数 |
| `wavelet_denoise` | 小波阈值去噪（通用阈值 / 自定义阈值） |

### 随机模拟与规划
| 工具 | 作用 |
|---|---|
| `monte_carlo` | 按分布抽样并计算表达式 |
| `linear_program` | 线性规划（HiGHS） |
| `integer_program` | 整数/混合整数规划（HiGHS） |

### 数据文件 IO
| 工具 | 作用 |
|---|---|
| `load_data` / `save_data` | 读写 CSV/TSV/TXT/JSON/NPY |
| `describe_file` / `plot_file` | 直接对文件列做统计或画图 |

### 高层工作流
| 工具 | 作用 |
|---|---|
| `analyze_dataset` | 读 CSV → 描述统计 + 相关矩阵 + 目标回归 + 图，一次完成 |
| `forecast` | ARIMA / 指数平滑多步预测，统一返回预测值与置信区间 |

### 兜底与健康检查
| 工具 | 作用 |
|---|---|
| `run_python` | 子进程执行任意 Python，预导入 numpy/scipy/sympy/matplotlib |
| `health_check` | 检查所有工具模块导入状态与注册函数数 |

## 启动

```powershell
cd sci-mcp
uv run sci-mcp
```

或直接用虚拟环境入口（Cherry Studio 注册时用这个）：

```text
.venv\Scripts\sci-mcp.exe
```

传输方式为 stdio，无需端口。若 `sci-mcp.exe` 正被 Cherry 占用，开发时用 `uv run --no-sync ...`。

## 输出目录

绘图、PDE/时频/小波结果（`.npz`）、`run_python` 产物默认写到：

```text
~/.sci-mcp/outputs
```

Windows 下即 `%USERPROFILE%\.sci-mcp\outputs`。

可用环境变量 `SCI_MCP_OUTPUT_DIR` 覆盖。

## 数据约定

- 长数组单次最多返回 2000 个点；超出时等间隔降采样并标注 `downsampled: true` 与原始长度。
- 完整 PDE/时频/小波结果保存为 `.npz`，返回值里带 `path`。
- 数组参数支持文件引用 `"@<path>"`（绝对路径或相对输出目录），支持 CSV/TSV/TXT/JSON/NPY；
  文本文件按行解析，一行一个值。
- `NaN` / `Infinity` 不进入 JSON：统一转成 `null`，顶层 `warnings` 说明被转换的字段。
- `load_data` 读取上限 50 MB / 1,000,000 行；`save_data`、`plot` 的文件名只允许 basename。

## 开发

```powershell
uv run --no-sync pytest -q
```

- Python ≥ 3.12；依赖由 `uv.lock` 锁定（numpy、scipy、sympy、matplotlib、pint、statsmodels、pandas、pywavelets）。
- 所有工具返回统一信封：成功 `{"ok": true, "result": ...}`，失败 `{"ok": false, "error": "..."}`。
- 工具函数不向外抛异常，错误统一走 `core.err()`。

## 扩展新工具

在 `src/sci_mcp/tools/` 新建模块，按固定写法：

```python
from ..core import ok, err

def my_tool(x: float) -> dict:
    """One-line English description shown to the model."""
    try:
        return ok(x * 2)
    except Exception as e:
        return err(f"my_tool failed: {e}")

def register(mcp) -> None:
    mcp.tool()(my_tool)
```

`server.py` 会自动发现该模块并注册，无需修改其他文件。

## 已知限制

- `run_python` 在**本机**子进程执行，不是安全沙箱；只适合可信代码。默认超时 30 秒（上限 300 秒），
  超时会杀整棵进程树；代码上限 100,000 字符；Windows 下默认 2048 MiB 内存上限
  （`SCI_MCP_SANDBOX_MEMORY_MB` 可调，16–65536）。
- PDE 覆盖一维热/波（含 Neumann/Robin、变系数）、二维拉普拉斯、三维热/拉普拉斯/波动、
  一维 Burgers、半通用一维显式 PDE、二维不可压 Navier-Stokes（周期边界）；
  仍不支持通用三维 CFD、湍流模型、复杂边界。
- 稀疏矩阵支持 solve/eigsh/svds/matmul/transpose/norm，求解器含直接法/CG/GMRES；
  超大规模仍受 nnz / 内存上限约束。
- 混合模型含线性 MixedLM、GEE，以及非线性混合模型的惩罚最小二乘近似（非完整 NLME）。
- 生存分析只做 Cox 比例风险与 Kaplan-Meier；不含竞争风险、时变协变量。
- 统计推断是经典检验 + OLS + GLM/ARIMA，不含混合模型、生存分析。
- `monte_carlo` 按独立变量抽样，不建模变量间相关性。
- Gaussian GLM 的 McFadden 伪 R² 可能略大于 1（statsmodels 似然尺度所致）。
- 指数平滑的置信区间用残差正态近似（statsmodels 无解析区间），返回里已标注。
- 符号解中的复数根和无理表达式以 sympy 字符串返回。
- `integer_program` 的 `method` 仅支持 `highs`。
