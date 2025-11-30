---
layout: post
title: "The Large Sieve: additive analytic form and number-theoretic applications"
description: "Summary of my 2017–2018 M1 project (Université Paris Saclay) supervised by Étienne Fouvry."
tags: [number theory, large sieve, exponential sums]
date: 2024-09-10
math: true
toc: true
---

Summary of my 2017–2018 pure mathematics M1 research project (Université Paris Saclay).

<p align="center">
  <img src="/assets/img/Large_sieve/Complex_zeta.jpg" alt="Zeta function" width="50%">
</p>

[Download the full report (PDF) (in french)]({{ "/assets/pdf/Le_grand_crible__sa_forme_analytique.pdf" | relative_url }}){:target="_blank"}

## Context

- Program: **M1 Jacques Hadamard**, Université Paris Saclay, 2017–2018.  
- Supervisor: **Étienne Fouvry**, Institut de Mathématiques d'Orsay.  
- Core sources: Montgomery (1978) *The analytic principle of the large sieve*; Bombieri (Cours au Collège de France, 1973).

## Goal

Study the **additive analytic large sieve** and aapply it to problems of equidistribution in number theory. Compare classical bounds (Gallagher) to the sharp form (Selberg). Derive arithmetic corollaries.

## TLDR

The **analytic additive form of the large sieve** is a single inequality about exponential sums. If you sample a trigonometric sum at many **well-spaced points**, the total energy you see cannot exceed about

$$
N \;+\; \delta^{-1}
$$

times the energy of its coefficients.  
Here $N$ is the length of the sum and $\delta$ is the minimal spacing between sample points.  
From that one bound you can prove useful facts about equidistribution, least quadratic non-residues, and the density of primes in arithmetic progressions.

---

## What is the question?

You have complex numbers $a_{M+1},\dots,a_{M+N}$ and look at the exponential sum

$$
S(\alpha)=\sum_{n=M+1}^{M+N} a_n\,e^{2\pi i n\alpha}.
$$

Pick points $\alpha_1,\dots,\alpha_R$ on the circle that are **$\delta$-separated** (no two closer than $\delta$ modulo 1).

How large can

$$
\sum_{r=1}^{R} |S(\alpha_r)|^2
$$

be, compared to $\sum_{n=M+1}^{M+N} \|a_n\|^2$ ?


**Large sieve inequality (additive form).**  
For all choices above,

$$
\sum_{r=1}^R |S(\alpha_r)|^2 \;\le\; \Delta(N,\delta)\,\sum_{n=M+1}^{M+N} \|a_n\|^2,
$$

with an optimal scale

$$
\boxed{\;\Delta(N,\delta)\ \asymp\ N+\delta^{-1}\;}
$$

and in fact

$$
\Delta(N,\delta) \le N+\delta^{-1}
$$

by a theorem attributed to **Selberg**.

**Why those two terms?**  
- You can always concentrate everything at a single frequency, which forces $\Delta\ge N$.  
- If you take **many** well-spaced points, you can average out the oscillations and force $\Delta\ge \delta^{-1}$.  
The right-hand side $N+\delta^{-1}$ matches both pressures.


---

## Intuition

$S(\alpha)$ can be seen as a **radio signal** built from $N$ consecutive frequencies.  
Sampling at many **well-separated** dials $\alpha_r$ cannot reveal more total power than what is actually present in the coefficients.The term $N$ is the number of frequencies, the term $\delta^{-1}$ is “how densely you probe the dial”.

---

## Core analytic inequality

If you sample at rationals $\alpha=\tfrac{a}{q}$ with $1\le q\le Q$, these points are $Q^{-2}$-separated. Plugging $\delta=Q^{-2}$ and Selberg’s bound gives

$$
\sum_{q\le Q}\ \sum_{\substack{a=1\\(a,q)=1}}^{q}\bigl|S(\tfrac{a}{q})\bigr|^2
\ \le\ (N+Q^2)\sum_{n}|a_n|^2.
\tag{★}
$$

This single inequality powers most applications below.

---

## Main applications

### 1) Equidistribution over residue classes
Let $A\subset\lbrace M+1,\dots,M+N\rbrace$ with $|A|=Z$. Write

$$
Z(q,h)=|\{n\in A: n\equiv h \pmod q\}|.
$$

Then (★) implies the **dispersion bound**

$$
\sum_{\substack{p\le Q\\ p\ \text{prime}}}
p\sum_{h=1}^{p}\left(Z(p,h)-\frac{Z}{p}\right)^2
\ \le\ (N+Q^2)\,Z.
$$

Interpretation: unless $Z$ is tiny, the set $A$ must look nearly uniform modulo many small primes.

**Sifted set bound.** If $\omega(p)$ residue classes modulo $p$ are forbidden, then

$$
Z\ \le\ \frac{N+Q^2}{\displaystyle \sum_{p\le Q}\frac{\omega(p)}{p}}.
$$

### 2) Least quadratic non-residue is usually small
For a prime $p$, the least $r$ that is **not** a square modulo $p$ is called the **least quadratic non-residue**.  
Using the large sieve with a smooth-numbers construction, one shows: for any $\varepsilon>0$, the primes $p\le x$ with least non-residue $>p^\varepsilon$ are rare, in fact $O_\varepsilon(\log\log x)$. This supports the heuristic that typical least non-residues are very small.

### 3) Primes in arithmetic progressions on short intervals (Brun–Titchmarsh)
Apply (★) to the progression $an+b$ and optimize $Q$. You recover the classical **Brun–Titchmarsh inequality**:

$$
\pi(x+y;a,b)-\pi(x;a,b)
\ \le\ (2+o(1))\ \frac{y}{\varphi(a)\,\log(y/a)}
$$

uniformly when $a$ is not too large compared to $y$. This gives a robust upper bound without unproved hypotheses.

---

## How the proof works at a glance

- **Duality trick.** Rewrite the inequality so the roles of $(a_n)$ and the sample points $(\alpha_r)$ swap.  
- **A good kernel.** Evaluate sums like

$$
\sum_{n} e^{2\pi i n(\alpha_r-\alpha_s)}=\frac{\sin(\pi N(\alpha_r-\alpha_s))}{\sin(\pi(\alpha_r-\alpha_s))}\times e^{i\cdots}.
$$

  The numerator kills far-apart points; the denominator penalizes close ones.  
- **Montgomery–Vaughan estimate.** Control bilinear forms with well-spaced $\alpha_r$, which delivers the $ \delta^{-1}$ part.  
- **Parseval.** Converts integrals of $\|S\|^2$ to $\sum \|a_n\|^2$, giving the $N$ part.

Gallagher first obtained $\Delta\le \delta^{-1}+\pi N$. Selberg sharpened it to $\Delta\le N+\delta^{-1}$, which is the right scale in general.

---

## References

- H. L. Montgomery, *The analytic principle of the large sieve* (1978).  
- E. Bombieri, *Le grand crible dans la théorie analytique des nombres* (Collège de France, 1973).

