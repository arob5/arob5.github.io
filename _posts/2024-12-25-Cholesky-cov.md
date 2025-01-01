---
title: Cholesky Decomposition of a Covariance Matrix
subtitle: Regression interpretation of the Cholesky decomposition, with applications to covariance estimation.
layout: default
date: 2024-12-25
keywords:
published: true
---

In this post we consider the Cholesky decomposition of the covariance
matrix of a Gaussian distribution. The eigendecomposition of covariance
matrices gives rise to the well-known method of
[principal components analysis](https://arob5.github.io/blog/2023/12/15/PCA/).
The Cholesky decomposition is not as widely discussed in this context, but
also has a variety of useful statistical applications.

# Setup and Background
{% katexmm %}

## The Cholesky Decomposition
The Cholesky decomposition of a positive definite matrix $C$ is the
unique factorization of the form
$$
C = LL^\top, \tag{1}
$$
where $L$ is a lower-triangular matrix with positive diagonal
elements (note that constraining the diagonal to be positive is required
for uniqueness).
A positive definite matrix can also be uniquely decomposed as
$$
C = LDL^\top, \tag{2}
$$
where $L$ is lower-triangular with ones on the diagonal, and $D$ is a diagonal
matrix with positive entries on the diagonal. We will refer to this as the
*modified Cholesky decomposition*, but it is also often called the
*LDL decomposition*. Given the modified Cholesky decomposition
$C = \tilde{L}D\tilde{L}^\top$, we can form (1) by setting
$L := \tilde{L}D^{1/2}$. We refer to both $\tilde{L}$ and $L$ as the
*lower Cholesky factor* of $C$; which we are referring to will be clear from
context. $L$ is guaranteed to be invertible, and $L^{-1}$ is itself a
lower-triangular matrix. Finally, note that we could also consider
decompositions of the form
$$
C = UDU^\top, \tag{3}
$$
where $U$ is upper triangular. [This](https://math.stackexchange.com/questions/2039477/cholesky-decompostion-upper-triangular-or-lower-triangular)
"reversed" Cholesky decomposition is not as common, but will show up at one
point in this post.

## Statistical Setup
Throughout this post we consider a random vector
$$
x := \left(x^{(1)}, \dots, x^{(p)} \right)^\top \in \mathbb{R}^p, \tag{4}
$$
with positive definite covariance $C := \text{Cov}[x]$. We will often assume
that $x$ is Gaussian, but this assumption is not required for some of the
results discussed below. We focus on the (modified) Cholesky decomposition
$C = LDL^\top$, letting $L = \{\ell_{ij}\}$ denote the entries of the
lower Cholesky factor. For the modified decomposition, and
$D := \text{diag}(d_1, \dots, d_p)$, where $d_i > 0$.

Let $C = LDL^\top$ and define the random variable
$$
\epsilon := L^{-1}x, \tag{5}
$$
which satisfies
$$
\text{Cov}[\epsilon] = L^{-1}CL^{-\top} = L^{-1}LDL^\top L^{-\top} = D. \tag{6}
$$
Thus, the map $x \mapsto L^{-1}x$ outputs a "decorrelated" random vector. The
inverse map $\epsilon \mapsto L\epsilon$ "re-correlates" $\epsilon$, producing
a random vector with covariance $C$. If we add on the assumption that
$x$ is Gaussian, then $\epsilon$ is a Gaussian vector with independent
entries. The transformation $L\epsilon$ is the typical method used in simulating
draws from a correlated Gaussian vector. Note that if we instead considered the
standard Cholesky factorization $C = LL^\top$ with $\epsilon$ still defined as
in (5), then $\text{Cov}[\epsilon] = I$.
{% endkatexmm %}

# Conditional Variances
{% katexmm %}
The following result provides an interpretation of the diagonal entries of $D$
in the Gaussian setting.

<blockquote>
  <p><strong>Proposition (conditional variances).</strong> <br>
  Let $x \sim \mathcal{N}(m,C)$, with $C = \text{Cov}[x]$ positive definite. Set
  $\epsilon := L^{-1}x$, where $C = LDL^\top$. Then
  $$
  \text{Var}[x^{(j)}|x^{(1)}, \dots, x^{(j-1)}] = d_j, \qquad j = 1, \dots, p \tag{7}
  $$
  where the $j=1$ case is interpreted as the unconditional variance
  $\text{Var}[x^{(1)}]$. If we instead define $L$ by $C = LL^\top$, then
  $$
  \text{Var}[x^{(j)}|x^{(1)}, \dots, x^{(j-1)}] = \ell^2_{jj}, \qquad j = 1, \dots, p. \tag{8}
  $$
  </p>
</blockquote>

**Proof.**
From the definition $x = L\epsilon$ and the fact that $L$ is lower triangular,
we have
$$
x^{(j)} = \sum_{k=1}^{j} \ell_{jk} \epsilon^{(k)}. \tag{9}
$$
Thus,

\begin{align}
\text{Var}[x^{(j)}|x^{(1)}, \dots, x^{(j-1)}]
= \text{Var}\left[\sum_{k=1}^{j} \ell_{jk} \epsilon^{(k)} \bigg|x^{(1)}, \dots, x^{(j-1)}\right]
&= \text{Var}\left[\sum_{k=1}^{j} \ell_{jk} \epsilon^{(k)} \bigg|\epsilon^{(1)}, \dots, \epsilon^{(j-1)}\right] \newline
&= \text{Var}\left[\ell_{jj} \epsilon^{(j)}|\epsilon^{(1)}, \dots, \epsilon^{(j-1)}\right] \newline
&= \text{Var}\left[\ell_{jj} \epsilon^{(j)}\right] \newline
&= \ell^2_{jj} \text{Var}[\epsilon^{(j)}]. \tag{10}
\end{align}

The first equality follows from the fact that $x$ is an invertible
transformation of $\epsilon$, while the fourth uses the fact that the
$\epsilon^{(j)}$ are independent (owing to the Gaussian assumption).
In the modified Cholesky case, (10) simplifies to
$\ell^2_{jj} \text{Var}[\epsilon^{(j)}] = 1 \times d_j = d_j$. For standard
Cholesky, it becomes
$\ell^2_{jj} \text{Var}[\epsilon^{(j)}] = \ell^2_{jj} \cdot 1 = \ell^2_{jj}$.
$\qquad \blacksquare$

Thus, the diagonal entries of $D$ give the variances of the $x^{(j)}$,
conditional on all preceding entries in the vector. Clearly, the interpretation
depends on the ordering of the entries, a fact that will be true for many
results that rely on the Cholesky decomposition.
{% endkatexmm %}

# A Regression Interpretation
{% katexmm %}
In this section, we summarize a least squares regression interpretation of the
modified Cholesky decomposition $C = LDL^\top$. The result is similar in
spirit to (7), as we will consider a sequence of regressions that condition
on previous entries of $x$. The ideas discussed here come primarily from
from {% cite CholeskyCovReg %}.

## Sequential Least Squares
We start by recursively defining a sequence of least squares problems, which
we then link to the factorization $C = LDL^\top$.

<blockquote>
  <p><strong>Sequential Least Squares.</strong> <br>
  Let $x \sim \mathcal{N}(m,C)$, with $C = \text{Cov}[x]$ positive definite.
  We recursively define the entries of
  $\epsilon := (\epsilon^{(1)}, \dots, \epsilon^{(p)})$ as follows:
  - Set $\epsilon^{(1)} := x^{(1)}$.
  - For $j = 2, \dots, p$ define
  </p>
</blockquote>


{% endkatexmm %}

# Another Regression Interpretation
{% katexmm %}
In this section, we provide an alternative regression interpretation.
We consider a slightly different sequence of least squares problems that
connects to the modified Cholesky decomposition of the *precision* $C^{-1}$,
rather than the covariance.

{% endkatexmm %}



{% endkatexmm %}
