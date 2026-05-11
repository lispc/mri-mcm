# MCM 拟合规模与数据规模说明

## 1. 输入数据规模

### 1.1 单个体素的采集数据

按照 Canals et al. (2022) 的大鼠 7T 方案：

| 参数 | 值 | 说明 |
|------|-----|------|
| 梯度方向数 | 30 | 均匀球面分布（Fibonacci 采样） |
| b 值 | 2 个 | 2000, 4000 s/mm² |
| 扩散时间 Δ | 4 个 | 15, 25, 40, 60 ms |
| 脉冲宽度 δ | 1 个 | 5 ms（假设） |
| **单个体素数据点** | **240** | = 30 × 2 × 4 |
| B0 图像 | 每 Δ 4 幅 | 用于归一化和噪声估计 |

数据维度：
```
S_data : (240,)  float64
    每个元素是一个 DWI 信号值（幅度图像，已归一化到 B0）
```

### 1.2 3D 矩阵规模

典型的大鼠脑 DWI 数据：

| 维度 | 典型值 | 说明 |
|------|--------|------|
| 矩阵大小 | 128 × 128 | 平面分辨率 |
| 层数 | 30–60 | 覆盖全脑 |
| 体素大小 | 0.2 × 0.2 × 0.2 mm³ | 各向同性 |
| **总体素数** | ~50万–100万 | = 128² × 30–60 |
| 掩膜后体素数 | ~10万–30万 | 排除脑外空气/头骨 |

原始数据文件大小：
```
240 方向 × 128 × 128 × 40 层 × 4 bytes (float32) ≈ 630 MB / 时间点
```

### 1.3 数据预处理流程

```
原始 DWI 数据 (Nifti 4D)
    ↓
[1] 运动校正 (MCFLIRT/FSL)
    ↓
[2] 涡流校正 (eddy/FSL)
    ↓
[3] 脑提取 (BET/FSL) → 生成二值掩膜
    ↓
[4] Rician 噪声估计
    ↓
[5] B0 归一化：S_data = S_b>0 / S_b=0
    ↓
掩膜内体素列表 + 归一化信号矩阵 (N_voxels × 240)
```

---

## 2. 拟合输出规模

### 2.1 每个体素拟合 12 个参数

```python
MCMParameters(
    f_ic=0.3,           # 1. 细胞内体积分数
    k=5.0,              # 2. Watson 分散参数
    mu=(x, y, z),       # 3–5. 主方向（3D 单位向量）
    d_parallel_ic=1e-3, # 6. 细胞内平行扩散率（固定）
    f_ss=0.1,           # 7. 小球体积分数
    R_ss=4.0,           # 8. 小球半径 (μm)
    f_ls=0.1,           # 9. 大球体积分数
    R_ls=8.0,           # 10. 大球半径 (μm)
    f_ec=0.2,           # 11. 细胞外体积分数
    d_parallel_ec=1.2e-3,  # 12. EC 平行扩散率 (mm²/s)
    d_perp_ec=0.6e-3,      # 13. EC 垂直扩散率 (mm²/s)
    f_T=0.8,            # 14. 组织水占比
    d_iso=3.0e-3,       # 15. 自由水扩散率（固定）
)
```

**实际自由参数**：12 个（d_parallel_ic, d_iso 固定；mu 用 θ, φ 表示减少 1 个）

### 2.2 输出参数图（3D 矩阵）

对每个参数生成一幅 3D 图像：

| 输出图像 | 维度 | 数据类型 | 说明 |
|---------|------|---------|------|
| f_ic_map | 128×128×40 | float32 | 小胶质细胞突起密度 |
| k_map | 128×128×40 | float32 | 突起取向一致性 |
| mu_x, mu_y, mu_z | 3×128×128×40 | float32 | 主方向（RGB 可染色）|
| f_ss_map | 128×128×40 | float32 | 小胶质细胞胞体密度 |
| R_ss_map | 128×128×40 | float32 | 小胶质细胞胞体大小 (μm) |
| f_ls_map | 128×128×40 | float32 | 星形胶质细胞胞体密度 |
| R_ls_map | 128×128×40 | float32 | 星形胶质细胞胞体大小 (μm) |
| f_ec_map | 128×128×40 | float32 | 细胞外空间占比 |
| f_fw_map | 128×128×40 | float32 | 自由水体积分数 (=1-f_T) |
| R2_map | 128×128×40 | float32 | 拟合优度 |
| cost_map | 128×128×40 | float32 | 残差平方和 |
| **总计** | **~15 幅 3D 图像** | | |

输出文件大小：
```
15 幅 × 128 × 128 × 40 × 4 bytes ≈ 40 MB
```

---

## 3. 计算规模

### 3.1 单个体素拟合时间

| 配置 | 时间 | 说明 |
|------|------|------|
| 基础版（Python 循环） | ~10 s/体素 | 未优化前 |
| 优化版（缓存 + 批量） | ~1.2 s/体素 | 当前实现 |
| 预期版（Numba/Cython） | ~0.1 s/体素 | 理论极限 |

### 3.2 全脑拟合时间估算

| 体素数 | 单核时间 | 16 核并行 | 64 核集群 |
|--------|---------|----------|----------|
| 1,000（ROI） | 20 min | 1.5 min | — |
| 10,000（半脑） | 3.3 h | 12 min | 3 min |
| 100,000（全脑） | 33 h | 2 h | 30 min |
| 500,000（高分辨） | 7 d | 10 h | 2.5 h |

**实际建议**：
- 开发/调试：ROI 1000 体素
- 论文结果：全脑 10 万体素，16 核服务器过夜
- 临床规模：需要 GPU 或 C++ 实现

### 3.3 内存需求

| 阶段 | 内存 | 说明 |
|------|------|------|
| 加载数据 | 1–2 GB | 240 方向 × 128³ 原始数据 |
| 预处理 | 3–5 GB | 运动校正、涡流校正的临时数据 |
| 拟合（单核） | <100 MB | 逐体素处理，不需要全图驻留 |
| 拟合（16 核） | ~1 GB | 每个核独立处理，略有开销 |
| 输出 | 40 MB | 15 幅参数图 |

---

## 4. 与标准 DWI 分析对比

| 方法 | 拟合参数数 | 单个体素时间 | 输出维度 |
|------|-----------|-------------|---------|
| **DTI** | 6 (扩散张量) | <1 ms | 6 幅图 + RGB |
| **NODDI** | 3 (f_ic, f_iso, ODI) | ~10 ms | 3 幅图 |
| **MCM (本项目)** | 12 | ~1.2 s | 12+ 幅图 |
| **SMT / AxCaliber** | 2–4 | ~100 ms | 2–4 幅图 |

MCM 的代价是计算时间高 2–3 个数量级，但换来的是：
- 5 个独立隔室的生物学解释
- 同时区分小胶质细胞和星形胶质细胞
- 多 b 值 + 多 Δ 的信息利用

---

## 5. 实际使用建议

### 5.1 ROI 分析（推荐用于开发）

```python
from mcm_forward import make_acquisition_scheme, simulate_mcm_data
from mcm_fit import fit_mcm_multi_start

# 加载 ROI 体素（例如小胶质细胞富集区）
roi_signals = load_roi_signals(...)  # (N_voxels, 240)
scheme = make_acquisition_scheme(n_dirs=30, b_values=(2000, 4000))

results = []
for i, voxel_signal in enumerate(roi_signals):
    result = fit_mcm_multi_start(
        **scheme, observed=voxel_signal,
        n_starts=3, max_nfev=2000, seed=42,
    )
    results.append(result["params"])
```

### 5.2 并行化策略

使用 Python 的 `multiprocessing` 或 `joblib`：

```python
from joblib import Parallel, delayed

def fit_one_voxel(voxel_signal, scheme):
    result = fit_mcm_multi_start(**scheme, observed=voxel_signal, n_starts=2)
    return result["params"]

params_list = Parallel(n_jobs=16)(
    delayed(fit_one_voxel)(sig, scheme) for sig in roi_signals
)
```

### 5.3 拟合稳定性建议

| 问题 | 解决方案 |
|------|---------|
| 局部最小值 | 多初始值（n_starts=3–5） |
| 参数边界 | 适当放宽边界（如 f_T 下限=0.5） |
| 噪声敏感 | 先用中值滤波预处理信号 |
| 计算太慢 | ROI 分析 → 全脑；或降采样体素 |
| R_ss/R_ls 不收敛 | 固定为文献值，只拟合 f |
