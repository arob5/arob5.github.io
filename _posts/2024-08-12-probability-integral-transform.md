---
title: The Probability Integral Transform
subtitle: Derivation of the PIT, and various generalizations.
layout: default
date: 2024-08-12
keywords: probability, statistics
published: false
---

# Setup and Background
{% katexmm %}

### Random Variable
Throughout this post we consider a probability space
$(\Omega, \mathcal{A}, \mathbb{B})$ and random variable
$Y: \Omega \to \mathcal{Y}$, where $(\mathcal{Y}, \mathcal{B})$ is a
measurable space. We let $\mu$ denote the distribution (law) of $Y$, which
is a measure on $(\mathcal{Y}, \mathcal{B})$ defined by
$\mu(B) := \mathbb{P}[Y \in B] := \mathbb{P}[Y^{-1}(B)]$.
We will primarily focus on the setting where
$\mathcal{Y} = \mathbb{R}$ and $\mathcal{B} = \mathcal{B}(\mathbb{R})$, the
Borel sets on the real line.

### Distribution Function: Univariate
In the real-valued setting the cumulative distribution
function (CDF) of $Y$ is defined as the function $F: \mathcal{B} \to [0,1]$
satisfying $F(y) := \mathbb{P}[Y \leq y] = \mu((-\infty, y])$, which satisfies:
1. Right continuity: $\lim_{t \to y+} F(t) = F(y)$.
2. Monotone nondecreasing: $F(y_1) \leq F(y_2)$ when $y_1 < y_2$.
3. Limits at infinity: $\lim_{y \to -\infty} F(y) = 0$ and
$\lim_{y \to \infty} F(y) = 1$.

### Distribution Function: Multivariate
The distribution function can also be defined when
$\mathcal{Y} = \mathbb{R}^d$. In this case, $F: \mathbb{R}^d \to [0,1]$ is
defined by
$$
F(y) :=
\mathbb{P}[Y_1 \leq y_1, \dots, Y_d \leq y_d] :=
\mu((-\infty,y_1] \times \dots \times (-\infty,y_d]).
$$
In this case, $F$ is right-continuous and non-decreasing in each component.
The third property generalizes as
$$
\lim_{y_1, \dots, y_d \to \infty} F(y) = 1
$$
and
$$
\lim_{y_i \to -\infty} F(y) = 0, \qquad \forall i = 1, \dots, d.
$$
{% endkatexmm %}

# Invertible Setting
We start by considering the setting where $F: \mathbb{R} \to \mathbb{R}$ is
invertible. Note that this implies that $F$ is strictly increasing and
continuous, meaning that $Y$ is a continuous random variable (it contains
no atoms).

<blockquote>
  <p><strong>Probability Integral Transform.</strong> <br>
  Assume that the real-valued random variable $Y$ has an invertible distribution
  function $F: \mathbb{R} \to [0,1]$. Then
  $$
  F(Y) \sim \mathcal{U}(0, 1). \tag{1}
  $$
  Equivalently, in the language of measures,
  $\mu \circ T^{-1}$ is uniform on $[0,1]$.
  </p>
</blockquote>

**Proof.**


<blockquote>
  <p><strong>Inverse Probability Integral Transform.</strong> <br>
  Assume that the real-valued random variable $Y$ has an invertible distribution
  function $F: \mathbb{R} \to [0,1]$. Then
  $$
  F^{-1}(U) \sim \mu, \tag{2}
  $$
  where $U \sim \mathcal{U}(0,1)$. Equivalently, in the language of measures,
  $\mu = \mathcal{U} \circ F$.
  </p>
</blockquote>

**Proof.**


{% katexmm %}
{% endkatexmm %}



# Invertible Setting
- Quantile function
- Transport map
- Intuition with graphic of Gaussian CDF
- Does this work in invertible transformation from Rn to Rn?
- Emphasize that F(Y) ~ Unif case requires Y to be continuous.
- Is it true that F_inv(U) ~ F case requires Y to be continuous?
(don't think so).

# Beyond Invertibility
- Common example: scalar valued maps (e.g., test statistics)
- Generalized inverse
- Which continuity assumptions are required
- Extension to non-continuous random variables?

# Knothe-Rosenblatt Rearrangement
- See Ghattas/Sanz-Alonso Monte Carlo book

# Randomized PIT
- See references for forecast theory post

## References
1. https://math.stackexchange.com/questions/375700/probability-integral-transform-is-it-integral-transform-can-it-be-for-discrete
2. https://stats.stackexchange.com/questions/209998/proving-the-probability-integral-transform-without-assuming-that-the-cdf-is-stri
3. Ryan Tibshirani forecast theory notes.
4. Ghattas and Sanz-Alonso Monte Carlo book
5. A note on generalized inverses
6. Please, not another note about generalized inverses
