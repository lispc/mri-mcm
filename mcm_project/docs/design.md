# MCM 多室微结构模型 — 设计文档

> 本文档随代码迭代维护，记录最新数学定义和实现细节。

---

## 1. 总信号模型

$$S(\mathbf{q}, b, \Delta) = f_{IC} \cdot S_{IC}(\mathbf{q}, b, k) + f_{SS} \cdot S_{SS}(b, \Delta, R_{SS}) + f_{LS} \cdot S_{LS}(b, \Delta, R_{LS}) + f_{EC} \cdot S_{EC}(\mathbf{q}, b) + (1-f_T) \cdot S_{FW}(b)$$

**体积约束**：$f_{IC} + f_{SS} + f_{LS} + f_{EC} + (1-f_T) = 1$，所有分数 $\in [0,1]$。

---

## 2. 隔室信号详细公式

### 2.1 Stick + Watson分布 — 小胶质细胞突起 ($S_{IC}$)

$$S_{IC}(\mathbf{q}, b, k) = \int_{S^2} f_{Watson}(\mathbf{n}; \boldsymbol{\mu}, k) \cdot \exp\left(-b \cdot d_{\parallel} \cdot (\mathbf{q} \cdot \mathbf{n})^2\right) d\mathbf{n}$$

**Watson分布**：

$$f_{Watson}(\mathbf{n}) = M\left(\frac{1}{2}, \frac{3}{2}, k\right)^{-1} \cdot \exp\left(k \cdot (\boldsymbol{\mu} \cdot \mathbf{n})^2\right)$$

其中 $M(a,b,z)$ 为合流超几何函数（Kummer function），由 `scipy.special.hyp1f1` 计算。

**参数**：
- $d_{\parallel} = 1.0 \times 10^{-3}$ mm²/s（受限水扩散率）
- $k \in [0, 30]$：Watson 集中参数（$k \to \infty$ 完全平行，$k=0$ 完全各向同性）
- $\boldsymbol{\mu}$：平均取向单位向量

**实现**：球面数值积分。对 Watson 分布方向进行均匀采样，加权平均 stick 信号。

**边界验证**：
- $k \to \infty$：退化为单方向 stick，$S_{IC} \to \exp(-b d_{\parallel} (\mathbf{q} \cdot \boldsymbol{\mu})^2)$
- $k \to 0$：退化为各向同性，$S_{IC} \to \frac{\sqrt{\pi}}{2\sqrt{b d_{\parallel}}} \text{erf}(\sqrt{b d_{\parallel}})$（需数值验证）

---

### 2.2 球体受限扩散 — 小胶质/星形胶质细胞胞体 ($S_{SS}, S_{LS}$)

**PGSE 序列下的 Murday-Cotts 公式**：

$$\ln\frac{S_{sphere}}{S_0} = -2\gamma^2 G^2 \sum_{m=1}^{\infty} \frac{1}{\alpha_m^6 (\alpha_m^2 R^2 - 2) D} \cdot F(\alpha_m^2 D, \delta, \Delta)$$

其中：

$$F(x, \delta, \Delta) = 2\delta - \frac{3 - 4e^{-x\delta} + e^{-2x\delta} + 2e^{-x\Delta} - 2e^{-x(\Delta+\delta)} - 2e^{-x(\Delta-\delta)} + e^{-x(2\Delta-\delta)} + e^{-2x\Delta}}{x}$$

**本征值** $\alpha_m$：满足反射边界条件的球贝塞尔函数根

$$\tan(\alpha R) = \frac{3\alpha R}{3 - (\alpha R)^2}$$

**物理常数**：
- $\gamma = 2.675 \times 10^8$ rad/(s·T)（质子旋磁比）
- $D = 1.0 \times 10^{-3}$ mm²/s（球体内水扩散率）
- $\delta = 5$ ms（梯度脉冲宽度，假设值）
- $G = \sqrt{b / (\gamma^2 \delta^2 (\Delta - \delta/3))}$（从 b 值反推梯度强度）

**实现**：
1. 数值求解前 15 个本征值 $\alpha_m R$
2. 计算级数求和（截断至 $m=15$，误差 $<10^{-6}$）
3. 从 b 值和 Δ 计算 G

**边界验证**：
- $R \to \infty$：退化为自由扩散，$\ln(S/S_0) \to -bD$
- 短时间极限（$\Delta \ll R^2/D$）：$\ln(S/S_0) \propto -bD$（自由扩散行为）
- 长时间极限（$\Delta \gg R^2/D$）：$\ln(S/S_0) \propto -\gamma^2 G^2 R^4 \Delta / D$（受限平台）

---

### 2.3 细胞外空间 — 径向对称张量 ($S_{EC}$)

$$S_{EC}(\mathbf{q}, b) = \exp\left(-b \cdot [d_{\parallel}^{EC} (\mathbf{q} \cdot \boldsymbol{\mu})^2 + d_{\perp}^{EC} (1 - (\mathbf{q} \cdot \boldsymbol{\mu})^2)]\right)$$

**注意**：无 tortuosity 假设，$d_{\parallel}^{EC}$ 和 $d_{\perp}^{EC}$ 独立。

**参数范围**：
- $d_{\parallel}^{EC} \in [0.5, 2.0] \times 10^{-3}$ mm²/s
- $d_{\perp}^{EC} \in [0.1, 1.0] \times 10^{-3}$ mm²/s

---

### 2.4 自由水 ($S_{FW}$)

$$S_{FW}(b) = \exp(-b \cdot d_{iso}), \quad d_{iso} = 3.0 \times 10^{-3} \text{ mm}^2/\text{s}$$

---

## 3. 采集方案

### 3.1 大鼠 7T 参数（模拟采用）

| 参数 | 值 |
|------|-----|
| 梯度方向 | 30（均匀球面分布） |
| b 值 | 2000, 4000 s/mm² |
| 扩散时间 Δ | 15, 25, 40, 60 ms |
| B0 | 每 Δ 4 幅 |
| 脉冲宽度 δ | 5 ms（假设） |

### 3.2 信号维度

每个体素总信号数 = 30 方向 × 2 b 值 × 4 Δ = 240 个（不含 B0）

---

## 4. 拟合参数汇总

| 参数 | 符号 | 范围 | 初始值 | 生物学意义 |
|------|------|------|--------|-----------|
| Stick 体积分数 | $f_{IC}$ | [0, 1] | 0.3 | 小胶质细胞突起密度 |
| Watson 分散参数 | $k$ | [0, 30] | 5.0 | 突起取向一致性 |
| 小球体积分数 | $f_{SS}$ | [0, 1] | 0.1 | 小胶质细胞胞体密度 |
| 小球半径 | $R_{SS}$ | [2, 6] μm | 4.0 μm | 小胶质细胞胞体大小 |
| 大球体积分数 | $f_{LS}$ | [0, 1] | 0.1 | 星形胶质细胞胞体密度 |
| 大球半径 | $R_{LS}$ | [6, 12] μm | 8.0 μm | 星形胶质细胞胞体大小 |
| 细胞外体积分数 | $f_{EC}$ | [0, 1] | 0.2 | 细胞外空间占比 |
| 细胞外平行扩散率 | $d_{\parallel}^{EC}$ | [0.5, 2.0]×10⁻³ | 1.2×10⁻³ | 细胞外沿主方向扩散 |
| 细胞外垂直扩散率 | $d_{\perp}^{EC}$ | [0.1, 1.0]×10⁻³ | 0.6×10⁻³ | 细胞外垂直方向扩散 |
| 组织水分数 | $f_T$ | [0, 1] | 0.8 | 总组织水占比 |

---

## 5. 拟合策略

### 5.1 目标函数

$$\chi^2(\boldsymbol{\theta}) = \sum_i \left(\frac{S_i^{model}(\boldsymbol{\theta}) - S_i^{data}}{\sigma_i}\right)^2$$

其中 $\sigma_i$ 为噪声标准差（Rician 噪声）。

### 5.2 约束处理

- **边界约束**：所有参数在预定义范围内
- **体积守恒**：$f_{IC} + f_{SS} + f_{LS} + f_{EC} + (1-f_T) = 1$
  - 实现：将 $f_{FW} = 1-f_T$ 作为自由参数，其他分数归一化
  - 或：使用 `scipy.optimize.minimize` 的约束优化

### 5.3 优化器

- 主优化器：`scipy.optimize.least_squares`（trust-region-reflective 或 lm）
- 多初始值策略：从 3-5 组不同初始值出发，选择最优结果
- 后备：如局部最优严重，使用 `scipy.optimize.differential_evolution`

---

## 6. 关键实现决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 球面积分方法 | 均匀方向采样（60点） | 计算快、精度足够 |
| 球体受限扩散 | Murday-Cotts PGSE（15项截断） | 物理精确、级数收敛快 |
| 扩散系数 | $1.0 \times 10^{-3}$ mm²/s | 论文 $10^{-9}$ mm²/s 疑似排版错误 |
| 噪声模型 | Rician 噪声 | MRI 幅度重建的真实噪声分布 |
| 拟合框架 | scipy.optimize.least_squares | 稳健、支持边界约束 |

---

## 7. 变更记录

| 日期 | 变更 | 作者 |
|------|------|------|
| 2026-05-10 | 初始版本 | Kimi |
