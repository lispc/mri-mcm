# MCM：多室微结构模型（扩散 MRI）

一个用于**扩散加权磁共振成像（DWI）**的**五隔室微结构模型**的 Python 实现，旨在通过小胶质细胞和星形胶质细胞不同的胞体大小及突起走向来**区分这两种细胞类型**。

> 本项目实现了 *Canals 等人（2022）* 提出的建模框架，并通过位移关联函数方法解决了球体受限扩散公式的著名数值不稳定性问题。

---

## 概述

该模型将 DWI 信号分解为五个具有生物学可解释性的隔室：

| 隔室 | 符号 | 生物物理意义 | 关键参数 |
|------|------|-------------|---------|
| **细胞内突起** | IC | 小胶质细胞突起（Watson 分布的 stick 模型） | `f_ic`、`k`、`μ` |
| **小球** | SS | 小胶质细胞胞体 | `f_ss`、`R_ss` |
| **大球** | LS | 星形胶质细胞胞体 | `f_ls`、`R_ls` |
| **细胞外空间** | EC | 细胞间扩散（各向异性张量） | `f_ec`、`d_∥`、`d_⊥` |
| **自由水** | FW | 脑脊液 / 未结合水 | `1-f_T`（`d_iso=3.0×10⁻³` 固定） |

$$
S = f_{IC} \cdot S_{IC} + f_{SS} \cdot S_{SS} + f_{LS} \cdot S_{LS} + f_{EC} \cdot S_{EC} + (1-f_T) \cdot S_{FW}
$$

**每个体素自由参数：** 12 个（体积分数、取向离散度、球体半径、EC 扩散率）

---

## 核心特性

### 🔬 生物学驱动的设计
- **Watson 分布 stick** 模型描述小胶质细胞突起的取向，集中参数为 `k`
- **两种球体尺寸**（2–6 μm 和 6–12 μm）分别对应小胶质细胞和星形胶质细胞的胞体大小
- **多 b 值、多扩散时间 Δ** 采集方案：每个体素 240 个测量点（30 方向 × 2 b 值 × 4 扩散时间）

### 🛡️ 数值稳定的球体受限扩散
经过大量测试，我们发现**经典文献公式的直接实现**（Neuman 1974、Murday-Cotts、Stepišnik）存在严重的数值问题：

| 公式 | 问题 | 现象 |
|------|------|------|
| Neuman 直接级数 | `cosh(350)` 溢出 | 典型参数下返回 `inf` |
| Murday-Cotts | 系数不匹配 | 无论参数如何，信号 ≈ 1 |
| Stepišnik | 指数相消 | 中间项出现 `NaN` |

**我们的解决方案**采用**位移关联函数**（Grebenkov 2007 框架），从解析积分的本征函数展开中推导信号。已通过蒙特卡洛随机游走模拟验证（N=20,000，误差 < 3%）。

> 📖 完整推导和踩坑记录见 [`docs/sphere_restricted_derivation.md`](docs/sphere_restricted_derivation.md)。

### ⚡ 性能优化
- **球面网格缓存**（`@lru_cache`）：消除重复的 60×60 网格生成
- **批量计算**：按唯一 `b` 值分组进行向量化计算
- **相比朴素实现 3.3 倍加速**：

| 指标 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| `mcm_signal_batch(60 点)` | 8.2 ms | 2.5 ms | **3.3×** |
| `fit_mcm(500 次评估)` | 3.5 s | 1.18 s | **3.0×** |

---

## 安装

```bash
# 克隆仓库
git clone <仓库地址>
cd mcm_project

# 安装依赖
pip install -r requirements.txt
```

**依赖：** NumPy ≥ 2.0、SciPy ≥ 1.10、Matplotlib ≥ 3.8

---

## 快速开始

### 正演模型 — 模拟 DWI 信号

```python
from src.mcm_forward import MCMParameters, make_acquisition_scheme, simulate_mcm_data

# 定义微结构参数
params = MCMParameters(
    f_ic=0.30,          # 小胶质细胞突起：30%
    k=8.0,              # 中等取向一致性
    mu=(1.0, 0.0, 0.0), # 沿 x 轴方向
    f_ss=0.10, R_ss=4.0,   # 小胶质细胞胞体：10%，4 μm
    f_ls=0.10, R_ls=8.0,   # 星形胶质细胞胞体：10%，8 μm
    f_ec=0.20,              # 细胞外空间：20%
    d_parallel_ec=1.2e-3,   # EC 平行扩散率（mm²/s）
    d_perp_ec=0.6e-3,       # EC 垂直扩散率（mm²/s）
    f_T=0.80,               # 组织水占比
)

# 创建采集方案：30 方向 × 2 个 b 值 × 4 个扩散时间
scheme = make_acquisition_scheme(n_dirs=30, b_values=(2000, 4000))

# 模拟信号（240 个测量点）
signal = simulate_mcm_data(params, **scheme, delta=5.0, Delta_list=[15, 25, 40, 60])
print(signal.shape)  # (240,)
```

### 参数拟合 — 恢复微结构

```python
from src.mcm_fit import fit_mcm_multi_start

# 多随机起点拟合观测数据
result = fit_mcm_multi_start(
    **scheme,
    delta=5.0,
    Delta_list=[15, 25, 40, 60],
    observed=signal,        # 你的归一化 DWI 数据
    n_starts=3,             # 3 组随机初值
    max_nfev=2000,          # 最大函数评估次数
    seed=42,
)

fitted = result["params"]
print(f"f_ic = {fitted.f_ic:.3f}, k = {fitted.k:.2f}, R_ss = {fitted.R_ss:.1f} μm")
```

---

## 项目结构

```
mcm_project/
├── src/                          # 核心实现
│   ├── mcm_forward.py            # 五隔室正演模型 + 模拟器
│   ├── mcm_fit.py                # 非线性最小二乘拟合
│   ├── watson_stick.py           # Watson 分布 stick 隔室
│   ├── sphere_restricted.py      # 球体受限扩散（PGSE）
│   ├── extracellular_tensor.py   # 各向异性 EC 张量隔室
│   ├── free_water.py             # 自由水（各向同性）隔室
│   └── utils.py                  # 球面采样工具（带缓存）
│
├── tests/                        # 测试套件（94 个测试，全部通过）
│   ├── test_end_to_end.py        # 完整参数恢复验证
│   ├── test_mcm_forward.py       # 正演模型正确性
│   ├── test_mcm_fit.py           # 拟合收敛性
│   ├── test_sphere_restricted.py # 球体扩散 vs 随机游走
│   ├── test_watson_stick.py      # Watson 分布极限
│   ├── test_extracellular_tensor.py
│   └── test_free_water.py
│
├── docs/                         # 文档
│   ├── design.md                 # 数学定义与参数表
│   ├── TECHNICAL_REPORT.md       # 参数恢复审计、代码质量、速度基准
│   ├── sphere_restricted_derivation.md  # 推导与公式踩坑记录
│   ├── fitting_scale_and_data.md # 数据规模、体素数、计算量估算
│   └── mcm_visual_summary.png    # 模型示意图
│
├── notebooks/                    # （Jupyter 笔记本 — 待补充）
├── scripts/
│   └── visualize_mcm.py          # 生成模型示意图
├── requirements.txt
├── README.md                     # 英文版
└── README.zh.md                  # 本文档
```

---

## 运行测试

```bash
# 运行完整测试套件
python -m pytest tests/ -v

# 带覆盖率报告
python -m pytest tests/ --cov=src --cov-report=term-missing
```

全部 94 个测试通过，包括：
- **端到端参数恢复**：真实参数在容差范围内被恢复（f_ic ±0.05、R ±1 μm、k 在 2 倍范围内）
- **球体扩散验证**：解析公式与随机游走模拟一致（误差 < 3%）
- **边界条件测试**：自由扩散极限（$R \to \infty$）、强受限极限（$R \to 0$）
- **Watson 分布极限**：$k \to 0$（各向同性）和 $k \to \infty$（单方向 stick）

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/design.md`](docs/design.md) | 完整数学定义、参数范围、采集方案细节 |
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | 代码质量审计、参数恢复结果、拟合速度基准、已知局限 |
| [`docs/sphere_restricted_derivation.md`](docs/sphere_restricted_derivation.md) | **文献公式为何失效**，以及位移关联函数方法如何解决 |
| [`docs/fitting_scale_and_data.md`](docs/fitting_scale_and_data.md) | 数据规模（240 点/体素）、脑体积估算（~10 万体素）、计算时间（1.2 秒/体素）、输出张量维度 |

---

## 参数汇总

| 参数 | 符号 | 范围 | 默认值 | 生物学意义 |
|------|------|------|--------|-----------|
| IC 体积分数 | `f_ic` | [0, 1] | 0.30 | 小胶质细胞突起密度 |
| Watson 集中参数 | `k` | [0, 30] | 5.0 | 突起取向一致性（0=各向同性，∞=完美平行） |
| 小球体积分数 | `f_ss` | [0, 1] | 0.10 | 小胶质细胞胞体密度 |
| 小球半径 | `R_ss` | [2, 6] μm | 4.0 μm | 小胶质细胞胞体大小 |
| 大球体积分数 | `f_ls` | [0, 1] | 0.10 | 星形胶质细胞胞体密度 |
| 大球半径 | `R_ls` | [6, 12] μm | 8.0 μm | 星形胶质细胞胞体大小 |
| EC 体积分数 | `f_ec` | [0, 1] | 0.20 | 细胞外空间占比 |
| EC 平行扩散率 | `d_parallel_ec` | [0.5, 2.0]×10⁻³ mm²/s | 1.2×10⁻³ | 沿主方向扩散 |
| EC 垂直扩散率 | `d_perp_ec` | [0.1, 1.0]×10⁻³ mm²/s | 0.6×10⁻³ | 跨纤维方向扩散 |
| 组织水占比 | `f_T` | [0, 1] | 0.80 | 总结合水比例 |

---

## 性能与规模

| 规模 | 体素数 | 单核 | 16 核并行 |
|------|--------|------|----------|
| ROI 调试 | 1,000 | 20 分钟 | 1.5 分钟 |
| 半脑 | 10,000 | 3.3 小时 | 12 分钟 |
| 全脑 | 100,000 | 33 小时 | 2 小时 |

> 单个体素拟合时间：**~1.2 秒**（优化后的 Python）。体素间完全可并行。

详见 [`docs/fitting_scale_and_data.md`](docs/fitting_scale_and_data.md)。

---

## 引用

如使用本代码，请引用：

> Canals, S., et al. (2022). *Mapping microglia and astrocyte activation in vivo using diffusion MRI*. Nature Communications.

球体受限扩散实现基于以下框架：

> Grebenkov, D.S. (2007). *NMR survey of reflected Brownian motion*. Reviews of Modern Physics, 79(3), 1077.

---

## 许可证

MIT 许可证 — 详见 `LICENSE` 文件。
