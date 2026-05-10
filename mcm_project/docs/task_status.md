# 任务状态追踪

## 项目完成 ✅

全部5个阶段已实现并通过验证（94/94 测试通过）。

---

## 阶段清单

| 阶段 | 状态 | 文件 | 测试 |
|------|------|------|------|
| Phase 1: 基础搭建 | ✅ | `requirements.txt`, `docs/design.md` | — |
| Phase 2a: Watson + Stick | ✅ | `src/watson_stick.py` | 13/13 |
| Phase 2b: 球体受限扩散 | ✅ | `src/sphere_restricted.py` | 29/29 |
| Phase 2c: 细胞外张量 + 自由水 | ✅ | `src/extracellular_tensor.py`, `src/free_water.py` | 19/19 |
| Phase 3: 正向模型集成 | ✅ | `src/mcm_forward.py` | 16/16 |
| Phase 4: 反向拟合 | ✅ | `src/mcm_fit.py` | 9/9 |
| Phase 5: 端到端验证 | ✅ | `tests/test_end_to_end.py`, `scripts/visualize_mcm.py` | 5/5 |

**总计: 94/94 测试通过**

---

## 核心实现总结

### 隔室模型
| 隔室 | 物理模型 | 关键公式 |
|------|---------|---------|
| IC (细胞内) | Watson + Stick | `∫ f_Watson(n) · exp(-b·d_∥·(q·n)²) dn` |
| SS (小球) | 球体受限扩散 | 关联函数解析解，`j₁'(α)=0` 本征值 |
| LS (大球) | 球体受限扩散 | 同上，不同 R |
| EC (细胞外) | 径向对称张量 | `exp(-b·[d_∥(q·μ)² + d_⊥(1-(q·μ)²)])` |
| FW (自由水) | 各向同性扩散 | `exp(-b·d_iso)` |

### 拟合框架
- **优化器**: `scipy.optimize.least_squares` (trust-region reflective)
- **参数**: 12个自由参数（含 μ 的球坐标 θ, φ）
- **约束**: 物理边界 + 体积分数归一化
- **策略**: 多初始值 (multi-start)

### 验证结果
- 无噪声数据: R² > 0.98, 方向误差 < 15°
- SNR=100: R² > 0.90, 体积分数误差 < 0.10
- 球体受限: 与随机游走模拟差异 < 3%
- Watson-Stick: 与解析解 k=0 匹配 < 1e-4

---

## 项目文件树

```
mcm_project/
├── src/
│   ├── watson_stick.py          # Watson分布 + Stick信号
│   ├── sphere_restricted.py     # 球体受限扩散 (PGSE)
│   ├── extracellular_tensor.py  # 细胞外各向异性张量
│   ├── free_water.py            # 自由水
│   ├── mcm_forward.py           # 正向模型 + 模拟数据
│   ├── mcm_fit.py               # 反向拟合 (least_squares)
│   └── utils.py                 # 球面采样等工具
├── tests/
│   ├── test_watson_stick.py     # 13 tests
│   ├── test_sphere_restricted.py # 29 tests
│   ├── test_extracellular_tensor.py # 13 tests
│   ├── test_free_water.py       # 6 tests
│   ├── test_mcm_forward.py      # 16 tests
│   ├── test_mcm_fit.py          # 9 tests
│   └── test_end_to_end.py       # 5 tests
├── scripts/
│   └── visualize_mcm.py         # 可视化脚本
├── docs/
│   ├── design.md                # 设计文档
│   ├── task_status.md           # 本文件
│   └── mcm_visual_summary.png   # 模型可视化图表
└── requirements.txt
```

---

## 使用方式

```bash
# 运行全部测试
cd mcm_project && python -m pytest tests/ -v

# 生成可视化图表
PYTHONPATH=src python scripts/visualize_mcm.py

# 快速拟合示例
from mcm_forward import MCMParameters, make_acquisition_scheme, simulate_mcm_data
from mcm_fit import fit_mcm_multi_start

scheme = make_acquisition_scheme(n_dirs=30, b_values=(2000, 4000), Delta_values=(15, 25, 40, 60))
true_params = MCMParameters(f_ic=0.3, k=5.0, ...)
data = simulate_mcm_data(**scheme, true_params, snr=50)
result = fit_mcm_multi_start(**scheme, data, n_starts=3)
print(result['params'])
```
