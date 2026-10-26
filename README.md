# 人工智能

## 机器学习相关作业：SVD、贝叶斯过程与交叉熵

## 1. 线性代数中的奇异值分解（SVD）及例子

### 1.1 SVD 的定义

对任意实矩阵 \(A\in\mathbb{R}^{m\times n}\)，存在正交矩阵 \(U\in\mathbb{R}^{m\times m}\)、\(V\in\mathbb{R}^{n\times n}\) 和对角矩阵 \(\Sigma\in\mathbb{R}^{m\times n}\)，使得：

$$
A=U\Sigma V^{T}
$$

其中 \(\Sigma\) 的对角元素 \(\sigma_1\ge \sigma_2\ge\cdots\ge\sigma_r>0\) 称为奇异值，\(r=\mathrm{rank}(A)\)。\(U\) 的列称为左奇异向量，\(V\) 的列称为右奇异向量。

### 1.2 SVD 的构造过程

1. 计算 \(A^{T}A\)。由于 \(A^{T}A\) 是对称半正定矩阵，可正交对角化：
   $$
   A^{T}A=V\Lambda V^{T}
   $$
   其中 \(\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)\)，且 \(\lambda_i\ge 0\)。

2. 取奇异值：
   $$
   \sigma_i=\sqrt{\lambda_i},\quad i=1,\ldots,r
   $$

3. 对非零奇异值对应的右奇异向量 \(v_i\)，定义左奇异向量：
   $$
   u_i=\frac{1}{\sigma_i}Av_i
   $$
   可以验证 \(u_i\) 两两正交且单位化。

4. 将 \(U,V\) 补全为正交矩阵，得到：
   $$
   A=U\Sigma V^{T}
   $$

若只保留前 \(k\) 个奇异值，则得到截断 SVD：
$$
A_k=\sum_{i=1}^{k}\sigma_i u_i v_i^{T}
$$
它是在 Frobenius 范数意义下对 \(A\) 的最优低秩近似。

### 1.3 自己设计的例子

取：
$$
A=\begin{bmatrix}
3 & 0\\
4 & 5
\end{bmatrix}
$$

计算：
$$
A^{T}A=
\begin{bmatrix}
3 & 4\\
0 & 5
\end{bmatrix}
\begin{bmatrix}
3 & 0\\
4 & 5
\end{bmatrix}
=
\begin{bmatrix}
25 & 20\\
20 & 25
\end{bmatrix}
$$

求特征值：
$$
\det(A^{T}A-\lambda I)
=(25-\lambda)^2-400=0
$$
得到：
$$
\lambda_1=45,\quad \lambda_2=5
$$
因此奇异值为：
$$
\sigma_1=3\sqrt{5},\quad \sigma_2=\sqrt{5}
$$

对应单位特征向量可取：
$$
v_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},
\quad
v_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}
$$

计算左奇异向量：
$$
Av_1=
\begin{bmatrix}
3 & 0\\
4 & 5
\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\1\end{bmatrix}
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}3\\9\end{bmatrix}
$$
单位化得：
$$
u_1=\frac{Av_1}{\sigma_1}
=
\begin{bmatrix}
\frac{1}{\sqrt{10}}\\
\frac{3}{\sqrt{10}}
\end{bmatrix}
$$

同理：
$$
Av_2=
\frac{1}{\sqrt{2}}
\begin{bmatrix}3\\-1\end{bmatrix},
\quad
u_2=\frac{Av_2}{\sigma_2}
=
\begin{bmatrix}
\frac{3}{\sqrt{10}}\\
-\frac{1}{\sqrt{10}}
\end{bmatrix}
$$

于是：
$$
U=
\begin{bmatrix}
\frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}}\\
\frac{3}{\sqrt{10}} & -\frac{1}{\sqrt{10}}
\end{bmatrix},
\quad
\Sigma=
\begin{bmatrix}
3\sqrt{5} & 0\\
0 & \sqrt{5}
\end{bmatrix},
\quad
V=
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

因此：
$$
A=U\Sigma V^{T}
$$
代回可验证：
$$
U\Sigma V^{T}
=
\begin{bmatrix}
3 & 0\\
4 & 5
\end{bmatrix}
$$

---

## 2. 贝叶斯过程理解及项目应用案例

> 说明：以下以“工业设备剩余寿命预测”为示例项目。提交时请将项目名称、传感器特征和标签替换为你自己的真实项目内容。

### 2.1 对贝叶斯过程的理解

贝叶斯过程的核心是：用先验分布表达对未知参数或未知函数的初始认识；用观测数据构造似然；通过贝叶斯公式得到后验分布；再用后验分布做预测和决策。其基本形式为：

$$
p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}
\propto p(D\mid\theta)p(\theta)
$$
其中 \(D\) 是观测数据，\(\theta\) 是未知参数或未知函数。

贝叶斯方法不仅给出点预测，还给出不确定性。若数据不断到来，可以用后验作为新的先验进行在线更新：
$$
p(\theta\mid D_1,D_2)\propto p(D_2\mid\theta)p(\theta\mid D_1)
$$
在高斯过程等函数型贝叶斯模型中，可以把“未知函数”本身作为随机对象建模，从而得到预测均值和预测方差。

### 2.2 项目问题建模

**项目问题**：根据设备振动、温度、电流等传感器特征，预测设备剩余使用寿命 RUL，并给出预测不确定性，用于维护决策。

设第 \(i\) 个样本的特征向量为 \(\phi(x_i)\in\mathbb{R}^{d}\)，标签为剩余寿命 \(y_i\)。建立贝叶斯线性回归模型：
$$
y_i=\phi(x_i)^{T}w+\varepsilon_i,
\quad
\varepsilon_i\sim\mathcal{N}(0,\sigma^2)
$$

设参数先验为：
$$
w\sim\mathcal{N}(m_0,S_0)
$$

记：
$$
\Phi=
\begin{bmatrix}
\phi(x_1)^{T}\\
\vdots\\
\phi(x_n)^{T}
\end{bmatrix},
\quad
y=
\begin{bmatrix}
y_1\\
\vdots\\
y_n
\end{bmatrix}
$$
则似然为：
$$
y\mid w\sim\mathcal{N}(\Phi w,\sigma^2 I)
$$

由贝叶斯公式：
$$
p(w\mid D)\propto p(y\mid w)p(w)
$$
由于高斯先验与高斯似然共轭，后验仍为高斯分布：
$$
w\mid D\sim\mathcal{N}(m_n,S_n)
$$
其中：
$$
S_n=\left(S_0^{-1}+\frac{1}{\sigma^2}\Phi^{T}\Phi\right)^{-1}
$$
$$
m_n=S_n\left(S_0^{-1}m_0+\frac{1}{\sigma^2}\Phi^{T}y\right)
$$

对于新样本 \(x_*\)，令 \(\phi_*=\phi(x_*)\)，后验预测分布为：
$$
y_*\mid x_*,D
\sim
\mathcal{N}\left(
\phi_*^{T}m_n,
\sigma^2+\phi_*^{T}S_n\phi_*
\right)
$$
其中 \(\phi_*^{T}m_n\) 是预测均值，\(\sigma^2+\phi_*^{T}S_n\phi_*\) 反映预测不确定性。

### 2.3 项目落地流程

1. 采集设备振动、温度、电流等传感器数据；
2. 做滑窗特征提取，如均值、方差、频谱能量、峰值因子等；
3. 构造训练集 \(D=\{(x_i,y_i)\}\)，其中 \(y_i\) 为剩余寿命或健康指标；
4. 设定先验 \(m_0,S_0\) 和噪声方差 \(\sigma^2\)；
5. 计算后验 \(m_n,S_n\)；
6. 对新样本输出预测均值和置信区间；
7. 当预测均值低于维护阈值或预测方差过大时，触发检修或增加采样频率；
8. 新数据到来后，用后验作为新先验进行在线更新。

若线性模型不足，可扩展为高斯过程回归：
$$
f(x)\sim\mathcal{GP}(m(x),k(x,x'))
$$
则预测分布为：
$$
f_*\mid D\sim\mathcal{N}(\mu_*,\sigma_*^2)
$$
其中：
$$
\mu_*=m(x_*)+k_*^{T}(K+\sigma^2 I)^{-1}(y-m)
$$
$$
\sigma_*^2=k(x_*,x_*)-k_*^{T}(K+\sigma^2 I)^{-1}k_*
$$
这体现了贝叶斯过程“先验函数—观测似然—后验函数—预测不确定性”的完整建模思想。

---

## 3. 交叉熵的定义、原理、应用及推导例子

### 3.1 定义

设真实分布为 \(p(x)\)，模型预测分布为 \(q(x)\)，则离散情形下交叉熵定义为：
$$
H(p,q)=-\sum_x p(x)\log q(x)
$$
连续情形为：
$$
H(p,q)=-\int p(x)\log q(x)\,dx
$$

### 3.2 原理

信息量可定义为 \(I(x)=-\log q(x)\)。交叉熵是真实分布 \(p\) 下信息量的期望：
$$
H(p,q)=\mathbb{E}_{x\sim p}[-\log q(x)]
$$
交叉熵可分解为：
$$
H(p,q)=H(p)+\mathrm{KL}(p\Vert q)
$$
其中：
$$
H(p)=-\sum_x p(x)\log p(x)
$$
$$
\mathrm{KL}(p\Vert q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
$$
由于 \(\mathrm{KL}(p\Vert q)\ge 0\)，所以 \(H(p,q)\ge H(p)\)。当且仅当 \(p=q\) 时取等号。因此最小化交叉熵等价于让预测分布 \(q\) 尽量接近真实分布 \(p\)。在分类问题中，若真实标签为 one-hot 分布，则最小化交叉熵等价于极大似然估计。

### 3.3 例子推导

设三分类问题，真实类别为第 1 类，因此真实分布为 \(p=(1,0,0)\)。某模型预测分布为 \(q=(0.7,0.2,0.1)\)。

则交叉熵为：
$$
H(p,q)
=-\left[1\cdot\log 0.7+0\cdot\log 0.2+0\cdot\log 0.1\right]
=-\log 0.7
\approx 0.3567
$$

若另一个模型预测为 \(q'=(0.9,0.05,0.05)\)，则：
$$
H(p,q')=-\log 0.9\approx 0.1053
$$
因为 \(0.1053<0.3567\)，所以第二个模型交叉熵更小，预测更接近真实分布。

对于 softmax 分类器，设：
$$
q_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$
损失函数为：
$$
L=-\sum_i p_i\log q_i
$$
可推导得到：
$$
\frac{\partial L}{\partial z_i}=q_i-p_i
$$
这说明当预测概率 \(q_i\) 大于真实概率 \(p_i\) 时，梯度为正，参数更新会降低该类别得分；反之则提高该类别得分。

二分类情形下，若真实标签 \(y\in\{0,1\}\)，预测正类概率为 \(\hat y\)，则：
$$
L=-\left[y\log\hat y+(1-y)\log(1-\hat y)\right]
$$

### 3.4 应用

交叉熵常用于图像分类、文本分类、语言模型、推荐系统点击率预估等任务。它既可以作为分类损失函数，也可以从极大似然角度解释为负对数似然。

---

## 4. Git 提交说明

将本文件保存为 `README.md`，在本地仓库执行：

```bash
git init
git add README.md
git commit -m "提交SVD、贝叶斯过程与交叉熵作业"
git branch -M main
git remote add origin <你的远程仓库地址>
git push -u origin main
