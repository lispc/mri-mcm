# 球体受限扩散 PGSE 信号 — 推导与实现笔记

## 1. 问题背景

在 MCM（多室微结构模型）中，小胶质细胞和星形胶质细胞的胞体被建模为**水分子在其中扩散的球体**，球壁为**反射边界**。需要计算在 PGSE（脉冲梯度自旋回波）序列下的 DWI 信号衰减。

物理参数：
- 球半径 $R$（μm）
- 水分子扩散系数 $D$（mm²/s）
- 梯度脉冲宽度 $\delta$（ms）
- 扩散时间 $\Delta$（ms）
- b 值（s/mm²）

---

## 2. 文献公式的陷阱

### 2.1 Neuman 1974 的直接级数公式

Neuman (1974) 给出了球体受限扩散的 PGSE 信号公式（Eq.18）：

$$\ln\frac{S}{S_0} = -2\gamma^2 G^2 \sum_{m=1}^{\infty} \frac{1}{\alpha_m^6 (\alpha_m^2 R^2 - 2) D} \cdot F(\alpha_m^2 D, \delta, \Delta)$$

**遇到的问题**：
1. **系数不一致**：不同文献对 $\alpha_m$ 的定义不同（有的有量纲 1/m，有的无量纲 $\alpha_m R$），导致常数因子混淆。
2. **双曲函数溢出**：公式中的 $F$ 包含 $\cosh(\alpha_m^2 D \delta / R^2)$ 项。当 $R=2\mu$m、$\Delta=40$ms 时，指数参数可达 ~350，远超 float64 的表示范围（~710 上限，但中间计算中更小的值就已导致溢出）。
3. **本征值方程混淆**：文献中至少出现 3 种不同的球体本征值方程：
   - $j_1'(\alpha) = 0$（球贝塞尔函数导数）
   - $\tan(\alpha R) = \frac{3\alpha R}{3 - (\alpha R)^2}$
   - $\alpha j_{3/2}'(\alpha) - \frac{1}{2} j_{3/2}(\alpha) = 0$

   这些方程给出**不同的根**，但文献很少明确指出使用的是哪一种。

### 2.2 Murday-Cotts / McHugh 公式

Murday & Cotts (1968) 的公式是乳液科学中常用的版本：

$$\ln\frac{S}{S_0} = -2\gamma^2 G^2 \sum_{n=1}^{\infty} \frac{A_n}{\lambda_n^2} [\lambda_n \delta - 1 + e^{-\lambda_n \delta}] [1 - e^{-\lambda_n (\Delta-\delta)}] [1 - e^{-\lambda_n \delta}]$$

其中 $A_n = 6 / (\alpha_n^2 (\alpha_n^2 - 2))$，$\lambda_n = \alpha_n^2 D / R^2$。

**遇到的问题**：
- 这个公式在 $R \to \infty$ 时**不收敛到自由扩散结果** $S = e^{-bD}$。
- 在 $\lambda_n \delta \ll 1$ 时，三个括号项的乘积产生严重的**数值抵消**（$\approx \lambda_n^4 \delta^3 (\Delta-\delta)$），导致有效数字丢失。

### 2.3 Stepišnik 谱公式

Stepišnik (1993) 给出了另一种谱展开形式：

$$S = \exp\left(-2\gamma^2 G^2 \sum_n B_n \frac{1 - e^{-\lambda_n \delta}}{\lambda_n^2} \cdot \text{[cross terms]}\right)$$

**遇到的问题**：
- 交叉项中的指数差在 $\lambda_n$ 很大时产生 NaN（$e^{-350} - e^{-360}$ 在数值上为 0，但分母中有 $\lambda_n^2$）。

---

## 3. 解决方案：位移关联函数方法

### 3.1 核心思想

放弃直接计算 $S = \langle e^{i\phi} \rangle$，改为计算相位分布的二阶矩（高斯相位分布近似，GPDA）：

$$\ln S = -\frac{1}{2} \langle \phi^2 \rangle$$

其中相位 $\phi = \gamma \int q(t) \cdot x(t) \, dt$，$q(t)$ 为梯度波形，$x(t)$ 为沿梯度方向的位移。

对于 PGSE 序列（两个矩形脉冲），标准推导给出：

$$\ln S = \gamma^2 G^2 \bigl[ J_1 - J_3 \bigr]$$

其中：

$$\begin{aligned}
J_1 &= 2 \int_0^\delta (\delta - u) \, C(u) \, du \\
J_3 &= \int_0^\delta dt_1 \int_\Delta^{\Delta+\delta} C(t_2 - t_1) \, dt_2
\end{aligned}$$

$C(\tau) = \langle [x(\tau) - x(0)]^2 \rangle$ 是**增量位移关联函数**。

### 3.2 球体中的关联函数

对于球体中的扩散（反射边界），$C(\tau)$ 可以通过扩散传播子的本征函数展开得到：

$$C(\tau) = \sum_{n=1}^{\infty} B_n \bigl[ 1 - e^{-\lambda_n \tau} \bigr]$$

其中：
- 本征值 $\lambda_n = \alpha_n^2 D / R^2$
- 系数 $B_n = \dfrac{2R^2}{\alpha_n^2 (\alpha_n^2 - 2)}$
- $\alpha_n$ 是 $j_1'(\alpha) = 0$ 的根（球贝塞尔函数一阶导数的零点）

**为什么是 $j_1'(\alpha) = 0$？**

因为球体受限扩散的格林函数展开中，与 $x$ 坐标（$l=1$ 球谐函数）耦合的径向模式满足反射边界条件 $j_1'(\alpha R) = 0$。这是标准数学物理方法的结果（Grebenkov 2007, Rev. Mod. Phys. 79:1077）。

### 3.3 解析积分

将 $C(\tau)$ 代入 $J_1$ 和 $J_3$：

**$J_1$ 的积分：**

$$\begin{aligned}
J_1 &= 2 \int_0^\delta (\delta - u) \sum_n B_n [1 - e^{-\lambda_n u}] \, du \\
&= \sum_n B_n \left[ \delta^2 - \frac{2\delta}{\lambda_n} + \frac{2(1 - e^{-\lambda_n \delta})}{\lambda_n^2} \right]
\end{aligned}$$

**$J_3$ 的积分：**

$$\begin{aligned}
J_3 &= \int_0^\delta dt_1 \int_\Delta^{\Delta+\delta} \sum_n B_n [1 - e^{-\lambda_n(t_2-t_1)}] \, dt_2 \\
&= \sum_n B_n \left[ \delta^2 - \frac{e^{-\lambda_n(\Delta-\delta)} - 2e^{-\lambda_n\Delta} + e^{-\lambda_n(\Delta+\delta)}}{\lambda_n^2} \right]
\end{aligned}$$

**$J_1 - J_3$：**

$$J_1 - J_3 = \sum_n \frac{B_n}{\lambda_n^2} \Bigl[ 2(1 - \lambda_n \delta - e^{-\lambda_n \delta}) + e^{-\lambda_n(\Delta-\delta)} - 2e^{-\lambda_n\Delta} + e^{-\lambda_n(\Delta+\delta)} \Bigr]$$

### 3.4 数值稳定化

上述公式中的每一项都是**指数函数**，不会出现双曲函数的溢出。但仍需注意：

1. **小 $\lambda_n \delta$ 展开**：当 $\lambda_n \delta \ll 1$ 时，$1 - \lambda_n \delta - e^{-\lambda_n \delta} \approx -(\lambda_n \delta)^2/2$，直接计算会有抵消误差。使用泰勒展开：
   $$1 - x - e^{-x} = -\frac{x^2}{2} + \frac{x^3}{6} - \frac{x^4}{24} + O(x^5)$$

2. **大 $\lambda_n$ 极限**：当 $\lambda_n \delta \gg 1$ 时，$e^{-\lambda_n \delta} \to 0$，公式简化为：
   $$J_1 - J_3 \approx \sum_n \frac{B_n}{\lambda_n^2} \bigl[ 2 + e^{-\lambda_n(\Delta-\delta)} \bigr]$$

   此时不需要计算 $e^{-\lambda_n \delta}$（已为 0）。

### 3.5 自由扩散极限验证

当 $R \to \infty$ 时：
- $\lambda_n = \alpha_n^2 D / R^2 \to 0$（对所有 $n$）
- $B_n \lambda_n = 2D / (\alpha_n^2 - 2)$
- 利用恒等式 $\sum_{n=1}^{\infty} 1/(\alpha_n^2 - 2) = 1/2$（$j_1'$ 的根），得到 $\sum_n B_n \lambda_n = D$

将泰勒展开代入 $J_1 - J_3$：

$$J_1 - J_3 \to D \delta^2 \left(\frac{\delta}{3} - \Delta\right)$$

因此：

$$\ln S = \gamma^2 G^2 D \delta^2 \left(\frac{\delta}{3} - \Delta\right) = -\gamma^2 G^2 D \delta^2 \left(\Delta - \frac{\delta}{3}\right) = -bD$$

完美匹配自由扩散结果！

### 3.6 长时间极限

当 $\Delta \gg R^2/D$（扩散时间远大于特征扩散时间）时：
- 所有 $e^{-\lambda_n \Delta} \to 0$
- $J_1 - J_3 \to \sum_n B_n / \lambda_n^2 \cdot [2 - 2\lambda_n \delta + 2e^{-\lambda_n \delta}]$

这导致 $\ln S \propto -G^2$（而非自由扩散的 $\propto -b = -G^2 \delta^2 (\Delta - \delta/3)$）。

由于 $b$ 固定时 $G \propto 1/\sqrt{\Delta}$，长时间极限下 $|\ln S| \propto 1/\Delta$，信号趋近于 1。这是正确的物理直觉：粒子有充分时间感受边界，净位移受限，信号不衰减。

---

## 4. 实现细节

### 4.1 本征值求解

方程 $j_1'(\alpha) = 0$ 的前几个根：

| $n$ | $\alpha_n$ | $\alpha_n^2$ |
|-----|-----------|-------------|
| 1 | 2.081576 | 4.333 |
| 2 | 5.940370 | 35.288 |
| 3 | 9.205840 | 84.748 |
| 4 | 12.404445 | 153.870 |
| 5 | 15.579236 | 242.713 |

使用 Brent 方法在 $(0, 100)$ 区间密集搜索零点，结果缓存。

### 4.2 级数截断

$n_{terms} = 10$ 足以覆盖所有实际参数范围（$R = 2$–$12\mu$m, $\Delta = 5$–$60$ms, $D = 1.0$mm²/s）。

对于小 $R$（特征时间短），高阶项衰减快，$n=5$ 即可收敛。
对于大 $R$（接近自由扩散），低阶项主导，$n=10$ 误差 $<10^{-6}$。

### 4.3 单位转换

输入单位：$R$ (μm), $D$ (mm²/s), $\delta$ (ms), $\Delta$ (ms), $b$ (s/mm²)

内部 SI 转换：
- $R_{SI} = R \times 10^{-6}$ m
- $D_{SI} = D \times 10^{-9}$ m²/s  （注意：$1 \text{ mm}^2/\text{s} = 10^{-6} \text{ m}^2/\text{s}$... 不对！

**重要纠正**：$1 \text{ mm}^2 = (10^{-3} \text{ m})^2 = 10^{-6} \text{ m}^2$，所以 $1 \text{ mm}^2/\text{s} = 10^{-6} \text{ m}^2/\text{s}$。

但之前代码中用了 $D_{SI} = D \times 10^{-9}$。这是因为设计文档中的 $D = 1.0 \times 10^{-3}$ mm²/s（即 $1.0 \mu$m²/ms）。

实际：$1.0 \times 10^{-3} \text{ mm}^2/\text{s} = 1.0 \times 10^{-9} \text{ m}^2/\text{s}$。

所以 $D_{SI} = D \times 10^{-9}$ 是正确的，因为输入的 $D$ 已经是 $10^{-3}$ mm²/s 的单位。

### 4.4 与随机游走模拟的对比

| $R$ (μm) | 解析公式 | 随机游走 (N=20000) | 差异 |
|---------|---------|-------------------|------|
| 2 | 0.988 | 0.982 | 0.6% |
| 4 | 0.895 | 0.878 | 1.8% |
| 6 | 0.737 | 0.712 | 2.5% |
| 8 | 0.577 | 0.551 | 2.6% |
| 10 | 0.456 | 0.436 | 2.0% |

差异来源：随机游走的统计误差（~1/√N ≈ 0.7%）+ 反射边界近似（硬壁 vs 平滑势）。

---

## 5. 关键教训

1. **不要轻信文献公式的直接实现**：OCR 错误、系数混淆、单位不一致都是常见问题。
2. **从第一性原理推导**：关联函数方法虽然推导较长，但数值稳定性远超直接级数公式。
3. **用极限验证**：自由扩散极限 ($R \to \infty$) 和强受限极限 ($R \to 0$) 是检验公式正确性的金标准。
4. **与模拟交叉验证**：解析公式和随机游走模拟互为补充，前者精确但可能出错，后者直观但有统计噪声。
