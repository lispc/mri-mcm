# 工作日志：Phase 1 — 基础搭建

**日期**：2026-05-10  
**阶段**：Phase 1: 基础搭建与文档初始化  
**目标**：创建项目结构、初始化文档、安装依赖

---

## 已完成工作

1. **创建项目目录结构**
   - `mcm_project/{docs/logs, src, tests, notebooks, scripts}/`
   - `src/__init__.py`
   - `requirements.txt`

2. **编写设计文档** (`docs/design.md`)
   - 完整数学公式定义
   - 参数表和范围
   - 采集方案
   - 拟合策略
   - 关键实现决策

3. **编写任务状态文档** (`docs/task_status.md`)
   - 阶段清单
   - 风险追踪

---

## 关键决策记录

### 决策：扩散系数取值

论文原文："Water diffusivity inside restriction is assumed to be 1 × 10⁻⁹ mm²/s"

分析：
- 如果按字面理解 $10^{-9}$ mm²/s = $10^{-15}$ m²/s，比正常组织水扩散率（$10^{-9}$ m²/s）小了 $10^6$ 倍，物理上不合理。
- 合理理解应为 $10^{-9}$ m²/s = $10^{-3}$ mm²/s = 1.0 μm²/ms。
- **结论**：代码中采用 $1.0 \times 10^{-3}$ mm²/s，并在文档中标注。

### 决策：梯度脉冲宽度 δ

论文未明确给出 δ 值。对于 EPI-DWI 序列：
- TE = 25 ms（大鼠），b 值最高 4000 s/mm²
- 反推 δ ≈ 5 ms 是合理假设
- **结论**：代码中默认 δ = 5 ms，作为可配置参数。

---

## 验证结果

- [x] 依赖安装成功（numpy 2.4.3, scipy 1.15.3, matplotlib 3.10.0, pytest 8.3.4）
- [x] Python import 无报错
- [x] 项目结构完整（src, tests, notebooks, scripts, docs/logs 全部存在）
- [x] 关键文件存在（design.md, task_status.md, requirements.txt）

**结论**：Phase 1 完成，环境就绪。

## 下一步

进入 Phase 2a（Watson + Stick 信号实现）。

---

## 下一步

安装依赖并验证环境，然后进入 Phase 2a（Watson + Stick 信号实现）。
