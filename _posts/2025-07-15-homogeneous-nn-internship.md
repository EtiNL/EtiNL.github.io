---
layout: post
title: "Evaluating the Performance of homogeneous Neural Networks"
date: 2025-07-15
tags: [research, deep-learning, Rademacher complexity, PyTorch, hNN]
math: true
toc: true
---
This post is a summary of my work during the M1 research project and internship at **CRIStAL (CNRS/Inria/Université de Lille)** under the supervision of **Andrey Polyakov** and **Mihaly Petreczky** on **homogeneous Neural Networks (hNNs)**.

[Download the full report (PDF)]({{ "/assets/pdf/Lanzeray_Etienne_Internship_report.pdf" | relative_url }}){:target="_blank"}

**Code:** [github.com/EtiNL/HNN](https://github.com/EtiNL/HNN)

## TLDR
hNNs encode dilation symmetry to learn homogeneous targets with fewer parameters and tighter Out of Distribution behavior. I prove a Rademacher-based generalization bound, derive a shift-aware risk law under $d(s)=e^{sG_d}$, ship a stable PyTorch stack, and show that hNNs converge faster and keep predictable error growth under scale shifts compared to MLPs.

---

## Context and motivation
Deep neural networks (DNNs) are central to modern ML but state-of-the-art models often use many parameters, which raises compute, memory, and generalization costs.

A way to mitigate this is to exploit the **geometry** of target functions, such as **homogeneity**. Homogeneity, studied since Euler, describes invariance under specific scaling transformations which appears naturally in control, physics, and modeling. 

This idea led to **homogeneous Neural Networks (hNNs)** (Polyakov, 2023), which can approximate any continuous homogeneous function on $\mathbb{R}^n$. Unlike using homogeneous activations in generic ANNs, hNNs enforce a **dilation symmetry** in the architecture itself.

Classical ANNs satisfy universal approximation theorems (Cybenko 1989; Hornik 1990), but those results are on **compact** domains, so extrapolation far from the training distribution is weak. By preserving a dilation symmetry, **hNNs** can extrapolate global behavior from local data.

---

## 1. Dilations, homogeneity, and hNN

### Euler homogeneity 
$f:\mathbb{R}^n\to\mathbb{R}$, 

$$f(\lambda x)=\lambda^{\nu}f(x) \text{ for } \lambda>0$$

The dilation is the isotropic scaling $x\mapsto \lambda x$, and $\nu\in\mathbb{R}$ is the homogeneity degree.  
Euler’s definition assumes isotropic scaling, which is a special case.

**From isotropic to anisotropic scaling.**  
In many practical settings (e.g., control, physics), scaling is anisotropic, meaning different components of $x$ may scale at different rates.  
This motivates generalized homogeneity, expressed via linear dilations.

### Dilations
A dilation family $\left(d(s)\right)_{s\in\mathbb{R}}$ is a continuous one-parameter group of linear operators:

$$
d(s)=e^{sG_d},\; G_d\in\mathcal{M}_{n}(\mathbb{R})\ \text{with}\ \operatorname{Re}(\operatorname{Spec}(G_d))>0
$$

This yields anisotropic scaling. ($\operatorname{Spec}$ = set of eigenvalues)

### Generalized homogeneity
A function $f:\mathbb{R}^n\to\mathbb{R}$ is homogeneous of degree $\nu\in\mathbb{R}$ with respect to $d$ iff

$$
f(d(s)x)=e^{\nu s}f(x),\; \forall x\in\mathbb{R}^n,\ \forall s\in\mathbb{R}
$$

**Remark.** With $G_d=I$ we recover isotropic dilation and Euler’s definition.

**Canonical $d$-norm.**  
Under suitable conditions on $d$ and a norm $$\|\cdot\|$$ on $\mathbb{R}^n$ (e.g., Euclidean), we can define the canonical homogeneous "norm"[^d-norm]:

$$
\|x\|_d:=e^{s_x}\quad\text{where}\quad \|d(-s_x)x\|=1
$$

And the projection onto the $d$-unit sphere:

$$
\pi_d(x):=d(-\ln\|x\|_d)\,x
$$
 

Then $f$ is homogeneous of degree $\nu$ w.r.t. $d$ iff

$$
f(x)=\|x\|_d^{\nu}\,f(\pi_d(x)),\; \forall x\in\mathbb{R}^n
$$

### hNN architecture (Polyakov, 2023)
Let $A\in\mathcal{M}_{m,n}(\mathbb{R})$, $\, b\in\mathbb{R}^m$, $\, C\in\mathbb{R}^m$, and $\sigma$ an activation (ReLU, tanh...).  

One-layer Neural Network:

$$
h(x)=C\,\sigma(Ax+b)
$$

One-layer homogeneous Neural Network:

$$
h_{\text{hom}}(x)=\|x\|_{d}^{\nu}\;C\,\sigma\big(A\,\pi_d(x)+b\big)
$$

This architecture enforces homogeneity: the network learns on the $d$-sphere and rescales radially.

[^d-norm]: Not a true norm on $\mathbb{R}^n$, but a norm in a space homeomorphic to $\mathbb{R}^n$ where the dilation $d$ becomes isotropic.



---

## 2. Generalization gap and Rademacher complexity

### Setup
- $\mathcal{X}\subset\mathbb{R}^n$ the **feature space** and $\mathcal{Y}\subset\mathbb{R}$ the **label space**.
- $\mathcal{D}$ a distribution on $\mathcal{X}\times\mathcal{Y}$.
- $\mathcal{H}$ the **hypothesis class** of functions $h:\mathcal{X}\to\mathcal{Y}$.
- $\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_+$ a **loss function**. 
- For $h\in\mathcal{H}$, the **True risk** is: 

$$L_{\mathcal{D}}(h):=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(h(x),y)\right]$$

- For samples $S=(x_i,y_i)_{1 \le i\le N}\sim\mathcal{D}^N$ and $h\in\mathcal{H}$, the **Empirical risk** is:

$$
L_S(h):=\frac{1}{N}\sum_{i=1}^N \ell\big(h(x_i),y_i\big).
$$

### Generalization gap

The Generalization gap is:

$$
\mathcal{E}(h)\ :=\ L_{\mathcal{D}}(h)-L_S(h)
$$

It measures how well a model generalizes outside of its training data $S$.

In machine learning, the goal is to learn

$$h^\star=\arg\min_{h\in\mathcal{H}} L_{\mathcal{D}}(h)$$


But we don't have access to $\mathcal{D}$, only $S\sim\mathcal{D}^N$.

So we minimize $L_S(h)$ (Empirical Risk Minimization) and obtain

$$
\hat h\ =\ \mathrm{ERM}_{\mathcal{H}}(S)\ :=\ \arg\min_{h\in\mathcal{H}} L_S(h)
$$


Then even if $L_S(\hat h)$ is small, we must ensure $\mathcal{E}(\hat h)$ is small with high probability, because $\mathcal{E}(\hat h)$ large $\Rightarrow$ **overfitting**.


### Rademacher complexity

The empirical Rademacher complexity measures the richness of a hypothesis class $\mathcal{H}$ for a given sample $S$, denoted 

$$\mathfrak{R}_S(\ell \circ \mathcal{H})\in \mathbb{R}_+$$



There are multiple complexity measures for hypothesis classes besides Rademacher complexity, such as the growth function and the VC dimension. Informally, Rademacher complexity quantifies a class’s ability to fit random noise.

And we can derive a Generalization bound from it:

Assuming  $\ell\le c$ on $$\operatorname{supp}(\mathcal{D}_\mathcal{Y} \times \mathcal{D}_\mathcal{Y})$$.
With probability $1-\delta$ over the draw of $S\sim\mathcal{D}^N$:

$$
\mathcal{E}(h) \le 2\,\mathfrak{R}_S(\ell\circ\mathcal{H}) + 4c\sqrt{\frac{\log(4/\delta)}{N}}
$$

Interpretation:
- A lower Rademacher complexity indicates a simpler hypothesis class, leading to better generalization and reduced overfitting:
![Overfitting with a too "complex" class of models](/assets/img/hnn_internship/overfitting.png)
- Increasing the number of training samples $N$ also decreases the risk of overfitting.

So we want to find the tightest bound possible for a class of models.

**One-layer neural networks:**

$$
\mathcal{H}_{\mathrm{NN}} =
\lbrace \,
h(x)=C\,\mathrm{ReLU}(Ax)
\ \big|\ 
\|C\|_2\le B_C,\;
\|A_{[i,:]}\|_2\le B_A,\;
\forall\,1\le i\le m
\, \rbrace
$$

$$
\mathfrak{R}_S(\mathcal{H}_{\mathrm{NN}})
\le
\frac{2 B_C B_A \sqrt{m}}{N}
\sqrt{ \sum_{i=1}^{N} \|x_i\|_2^2 }
$$

**One layer homogeneous neural network:**

$$
\mathcal{H}_{\mathrm{hNN}} =
\lbrace \,
h(x)=||x||_d^\nu C\,\mathrm{ReLU}(A\pi_d(x))
\ \big|\ 
\|C\|_2\le B_C,\;
\|A_{[i,:]}\|_2\le B_A,\;
\forall\,1\le i\le m
\, \rbrace
$$

$$
\mathfrak{R}_S(\mathcal{H}_{\mathrm{hNN}})
\le
\frac{2 B_C B_A \sqrt{m}}{N}
\sqrt{ \sum_{i=1}^{N} \|x_i\|_d^{2\nu} }
$$

---

## 3. True error under a dilation-induced distribution shift

**Motivation.**

If the data distribution is shifted along dilation orbits (e.g., due to a change of scale along $\mathbf d$), how does the true risk of a $\mathbf d$-homogeneous estimator transform?

Let $h$ be our model, $f$ the target homogeneous function and $\ell$ the squared loss.

Let $\mathcal D$ have a density $p$ [^D], and define the shifted distribution $\mathcal D_s$ by pushing forward $\mathcal D$ through $\mathbf d(s)$, i.e.,

$$
X\sim\mathcal D,\quad X_s=\mathbf d(s)X\sim\mathcal D_s
$$

If we apply the dilation to the input space, the new distribution $\mathcal{D}_s$ has density:

$$
p_s(x) = \exp(-s \operatorname{Tr}(G_d)) \, p\left( \exp(-s G_d) x \right)
$$

Then, the true error of $h$ under the shifted distribution $\mathcal{D}_s$ is:

$$
\begin{aligned}
\mathbb{E}_{x \sim \mathcal{D}_s} \left[ (h(x) - f(x))^2 \right] &= \int_{\mathbb{R}^n} (h(x) - f(x))^2 \, p_s(x) \, dx.\\
&= e^{2\nu s}\mathbb{E}_{x \sim \mathcal{D}} \left[ (h(x) - f(x))^2 \right]\\
&= e^{2\nu s}L_\mathcal{D}(h)
\end{aligned}
$$

This is particularly interesting because it lets us **infer the true risk outside the convex hull of the training data** (along dilation orbits) from the
true risk on the training distribution, which itself can be upper-bounded by
empirical risk plus a Rademacher term.

[^D]: Here $\mathcal{D}$ is defined only over $\mathcal{X}$ since the target space is determined by the function $f$.

---

## 4. Empirical comparison between hNN and MLP

### Comparison on synthetic data

The goal is to empirically evaluate the performance of a homogeneous neural network (hNN) against a standard MLP in approximating an **anisotropic homogeneous function**:

$$
f_{\text{anisotropic}}(x) = \sum_k c_k |x_1|^{\alpha_{1,k}} |x_2|^{\alpha_{2,k}}
$$

The exponents $\alpha_{1,k}$ and $\alpha_{2,k}$ satisfy the **homogeneity condition**:

$$
\alpha_{1,k} r_1 + \alpha_{2,k} r_2 = \nu, \, \forall k
$$

ensuring that the function is homogeneous under the anisotropic dilation defined by the diagonal generator  
$G_d = \mathrm{diag}(r_1, r_2)$.

### Data generation
Training data are generated synthetically as follows:

- **Inputs** $X$ are sampled from a uniform distribution:
  $$
  X \sim \mathcal{U}([-5,5]^2)
  $$

- **Outputs** $y$ are computed from the anisotropic homogeneous function:
  $$
  y_i = f_{\text{anisotropic}}(X_i), \quad 1\le i \le N
  $$


### Protocol
- **Loss function:** Mean Squared Error (MSE).  
- **Optimizer:** Adam.
- **Architectures:** depths $$L \in \lbrace 1, 2, 4\rbrace$, widths $m \in\lbrace 16, 32, 64\rbrace$$ 
- **Data split:** $N_{\text{train}} = 5000$, $N_{\text{test}} = 5000$ (same split for both models)  
- **Initialization:** identical weights per $(L, m)$ (same seed)  
- **Training:** same batches, schedule, and hyperparameters
 

---

### Results

**Test MSE at epochs 100 and 400** (best in each pair is **bold**):

| $L$ | $m$ | HNN@100 | MLP@100 | HNN@400 | MLP@400 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 16 | 6.6685 | **5.7901** | **1.3919** | 5.1490 |
| 1 | 32 | 7.2210 | **5.7467** | **1.4064** | 5.0438 |
| 1 | 64 | **1.6703** | 5.2474 | **1.4055** | 4.9431 |
| 2 | 16 | **1.7027** | 5.3313 | **1.3034** | 4.7761 |
| 2 | 32 | **1.4072** | 5.1450 | **1.3523** | 4.1455 |
| 2 | 64 | **1.3324** | 4.9475 | **1.1242** | 1.4535 |
| 4 | 16 | **1.3324** | 4.9475 | **1.1242** | 1.4535 |
| 4 | 32 | **1.3654** | 4.8335 | 1.0101 | **0.8878** |
| 4 | 64 | **1.2856** | 4.6582 | 0.7517 | **0.7069** |

### Observed trends

- **Low depth ($L=1$):**  
  The MLP fails to learn the target (test $\approx 5$ after 400 epochs), whereas the hNN converges to $\approx 1.4$ across widths.  

  ![hNN vs MLP at depth L=1 — hNN converges while MLP stagnates]( /assets/img/hnn_internship/mlp_vs_hnn_L1.png )

- **Moderate depth ($L=2$):**  
  The hNN keeps a strong advantage at all widths; even at $m=64$, it finishes better (1.124 vs 1.454) and reaches low error much earlier.  

  ![hNN vs MLP at depth L=2 — hNN converges faster and stays ahead]( /assets/img/hnn_internship/mlp_vs_hnn_L2.png )

- **Higher depth ($L=4$):**  
  The hNN converges fastest and reaches low error early; with enough width, the MLP eventually catches up and slightly surpasses final test MSE (0.888 at $m=32$, 0.707 at $m=64$), but only after long training.  

  ![hNN vs MLP at depth L=4 — hNN rapid convergence; MLP catches up later]( /assets/img/hnn_internship/mlp_vs_hnn_L4.png )

### Takeaway

For this anisotropic homogeneous target, the hNN’s inductive bias[^inductive-bias] enables faster and more reliable convergence at small and moderate capacities, while the MLP requires deeper/wider architectures and longer training to achieve comparable results.

[^inductive-bias]: The set of assumptions a learning algorithm makes about the target function before seeing any data.

---
### Empirical error under test-time dilation shift

From the analysis of the true error under dilation shift, we established that dilating the input distribution by  
$d(s) = \exp(s G_d)$ scales the true error as:

$$
\mathbb{E}_{x \sim \mathcal{D}_s} \big[ (h(x) - f(x))^2 \big]
= e^{2 \nu s} L_\mathcal{D}(h).
$$

Now let's directly compare the observed error growth with the theoretical scaling law.

![Empirical test error growth under anisotropic dilation for hNN, MLP, and the theoretical scaling law.](/assets/img/hnn_internship/mlp_vs_hnn_dilation_shift.png)

We observe that the hNN’s error grows almost exactly according to the theoretical law, confirming that it generalizes in a predictable and controlled manner under distributional shift.  
In contrast, the MLP’s error increases more rapidly, showing that the hNN generalizes significantly better **outside the convex hull of its training distribution** compared to the MLP.

---

## 5. Limitations & Perspectives

The generalization bound can likely be sharpened using more recent Rademacher-based results.  
On the empirical side, testing on other tasks with known scaling laws is a natural next step, for instance, in **homogeneous control**, or by adapting the framework to **Homogeneous Galerkin projection to solve PDEs** [^futur] (as done in Galerkin Neural Networks).


[^futur]: This will be the focus of my Master 2 research project, which explores applying homogeneous neural networks to fluid mechanics models through **homogeneous Galerkin projection**.

## References

- **Polyakov, Andrey** (Nov. 2023). *Homogeneous Artificial Neural Network.*  
  *arXiv:* [2311.17973](https://arxiv.org/abs/2311.17973) [cs].

- **Shalev-Shwartz, Shai** and **Ben-David, Shai** (2014). *Understanding Machine Learning — From Theory to Algorithms.*  
  *Cambridge University Press*, pp. I–XVI, 1–397.
