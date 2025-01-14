---
title: Cholesky Decomposition of a Covariance Matrix
subtitle: Interpreting the Cholesky factor of a Gaussian covariance matrix.
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
lower Cholesky factor. For the modified decomposition, we write
$D := \text{diag}(d_1, \dots, d_p)$, where each $d_j > 0$.

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

# Conditional Variances and Covariances
{% katexmm %}
We start by demonstrating how the (modified) Cholesky decomposition
encodes information
related to conditional variances and covariances between the $x^{(j)}$.
The below result considers conditional variances, and provides an interpretation
of the diagonal entries of $D$ in the Gaussian setting.

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

We can generalize the above result to consider conditional covariances
instead of variances, which yields an interpretation of the off-diagonal
elements of $L$.

<blockquote>
  <p><strong>Proposition (conditional covariances).</strong> <br>
  Let $x \sim \mathcal{N}(m,C)$, with $C = \text{Cov}[x]$ positive definite. Set
  $\epsilon := L^{-1}x$, where $C = LDL^\top$. Then for $i > j$,
  $$
  \text{Cov}[x^{(i)}, x^{(j)}|x^{(1)}, \dots, x^{(j-1)}] = \ell_{ij}d_j, \qquad j = 1, \dots, p-1 \tag{11}
  $$
  where the $j=1$ case is interpreted as the unconditional covariance
  $\text{Cov}[x^{(i)},x^{(1)}]$. If we instead define $L$ by $C = LL^\top$,
  then
  $$
  \text{Cov}[x^{(i)}, x^{(j)}|x^{(1)}, \dots, x^{(j-1)}] = \ell_{ij}\ell_{jj}, \qquad j = 1, \dots, p-1. \tag{12}
  $$
  In particular, in either case it holds that
  $$
  \ell_{ij}=0 \iff
  \text{Cov}[x^{(i)}, x^{(j)}|x^{(1)}, \dots, x^{(j-1)}] = 0 \iff
  x^{(i)} \perp x^{(j)} | x^{(1)}, \dots, x^{(j-1)}. \tag{13}
  $$
  </p>
</blockquote>

**Proof.** The proof proceeds similarly to the conditional variance case.
We have
\begin{align}
\text{Cov}[x^{(i)}, x^{(j)}|x^{(1)}, \dots, x^{(j-1)}]
&= \text{Cov}\left[\sum_{k=1}^{i} \ell_{ik} \epsilon^{(k)},
\sum_{k=1}^{j} \ell_{jk} \epsilon^{(k)}
\bigg|x^{(1)}, \dots, x^{(j-1)}\right] \newline
&= \text{Cov}\left[\sum_{k=1}^{i} \ell_{ik} \epsilon^{(k)},
\sum_{k=1}^{j} \ell_{jk} \epsilon^{(k)}
\bigg|\epsilon^{(1)}, \dots, \epsilon^{(j-1)}\right] \newline
&= \text{Cov}\left[\sum_{k=j}^{i} \ell_{ik} \epsilon^{(k)},
\ell_{jj} \epsilon^{(j)}
\bigg|\epsilon^{(1)}, \dots, \epsilon^{(j-1)}\right] \newline
&= \sum_{k=j}^{i} \ell_{ik}\ell_{jj} \text{Cov}\left[\epsilon^{(k)}, \epsilon^{(j)}|\epsilon^{(1)}, \dots, \epsilon^{(j-1)}\right] \newline
&= \sum_{k=j}^{i} \ell_{ik}\ell_{jj} \text{Cov}\left[\epsilon^{(k)}, \epsilon^{(j)}\right] \newline
&= \ell_{ij}\ell_{jj} \text{Var}\left[\epsilon^{(j)}\right]
\end{align}
The penultimate step uses the fact that the $\epsilon^{(j)}$ are
conditionally uncorrelated, owing to the fact that the $\epsilon^{(j)}$
are jointly Gaussian and independent. The final step also uses the
fact that the $\epsilon^{(j)}$ are uncorrelated, and hence all terms where
$k \neq j$ vanish. For $C = LDL^\top$ the final expression simplifies to
$\ell_{ij}\ell_{jj} \text{Var}\left[\epsilon^{(j)}\right] = \ell_{ij} \cdot 1 \cdot d_j = \ell_{ij}d_j$. For $C = LL^\top$ it becomes
$\ell_{ij}\ell_{jj} \cdot 1 = \ell_{ij}\ell_{jj}$. The first implication
in (13) follows immediately from (11) and (12). The second implication
follows from the fact that $x$ is Gaussian, and hence the conditional
uncorrelatedness implies conditional independence. $\blacksquare$

We thus find that the Cholesky decomposition of a Gaussian covariance
is closely linked to the *ordered* conditional dependence structure of
$x$. The factorization encodes conditional covariances, where the
conditioning is with respect to all preceding variables; reordering
the entries of $x$ may yield drastically different insights.
The connection between sparsity in the Cholesky factor and
conditional independence can be leveraged in the design of statistical
models and algorithms. For an example, see the
paper {% cite SparseCholeskyVecchia %}.
{% endkatexmm %}

# A Regression Interpretation
{% katexmm %}
In this section, we summarize a least squares regression interpretation of the
modified Cholesky decomposition $C = LDL^\top$. The result is similar in
spirit to (7), as we will consider a sequence of regressions that condition
on previous entries of $x$. The ideas discussed here come primarily from
from {% cite CholeskyCovReg %}.

## Sequence of Least Squares Problems
We start by recursively defining a sequence of least squares problems, which
we then link to the factorization $C = LDL^\top$.

<blockquote>
  <p><strong>Sequential Least Squares.</strong> <br>
  Let $x \sim \mathcal{N}(0,C)$, with $C = \text{Cov}[x]$ positive definite.
  We recursively define the entries of
  $\epsilon := (\epsilon^{(1)}, \dots, \epsilon^{(p)})$ as follows: <br><br>
  1. Set $\epsilon^{(1)} := x^{(1)}$. <br>
  2. For $j = 2, \dots, p$ define the regression coefficient
  $\beta^{(j)} \in \R^{j-1}$ by
  $$
  \beta^{(j)}
  := \text{argmin}_{\beta} \mathbb{E}\left\lvert x^{(j)} - \sum_{k=1}^{j-1} \beta_k \epsilon^{(k)} \right\rvert^2 \tag{14}
  $$
  and set
  $$
  \epsilon^{(j)} := x^{(j)} - \sum_{k=1}^{j-1} \beta_k^{(j)} \epsilon^{(k)}. \tag{15}
  $$
  </p>
</blockquote>

In words, $\epsilon^{(j)}$ is the residual of the least squares regression of
the response $x^{(j)}$ on the explanatory variables
$\epsilon^{(1)}, \dots, \epsilon^{(j-1)}$, and $\beta^{(j)}$ is the coefficient
vector. Take note that we are regressing on the *residuals* from the previous
regressions, rather than the $x^{(j)}$ themselves. We assume for simplicity that
$x$ is mean zero to avoid having to deal with an intercept term; for non mean
zero variables, we can start by subtracting off their mean and then apply
the same procedure. Note also that the zero mean assumption implies that
$\mathbb{E}[\epsilon] = 0$; this follows from $\epsilon^{(1)} = x^{(1)}$ along
with the recursion (15).

Our goal is now to connect this algorithm to the modified Cholesky decomposition
of $C$. In particular, we will show that the $\epsilon$ defined by the
regression residuals is precisely the $\epsilon$ defined in (5), which arises
from the modified Cholesky decomposition. To start, note that if we rearrange
(15) as
$$
x^{(j)} := \epsilon^{(j)} + \sum_{k=1}^{j-1} \beta^{(k)} \epsilon^{(k)}, \tag{16}
$$
then we see the vectors $\epsilon$ and $x$ are related as
$$
x = L\epsilon, \tag{17}
$$
where
\begin{align}
L &:=
\begin{bmatrix} 1 & 0 & 0 & \cdots & 0 \newline
                \beta^{(2)}_1 & 1 & 0 & \cdots & 0 \newline
                \vdots & \vdots & \vdots & \cdots & 0 \newline
                \beta^{(p)}_1 & \beta^{(p)}_2 & \cdots & \cdots & 1\end{bmatrix}. \tag{18}
\end{align}
That is, we have defined $L$ to be the lower triangular matrix with
$j^{\text{th}}$ row set to $(\beta^{(j)}, 1)$, the $j^{\text{th}}$ coefficient
vector with a $1$ appended to the end. We immediately have that $L$ is
invertible, as it is a triangular matrix with non-zero entries on the diagonal.
We also have
$$
C = \text{Cov}[x] = \text{Cov}[L\epsilon] = L \text{Cov}[\epsilon] L^{\top}. \tag{19}
$$
In order to show that (19) actually yields the modified Cholesky factorization,
we must establish that $\text{Cov}[\epsilon]$, the residual covariance matrix,
is diagonal with positive diagonal entries.

<blockquote>
  <p><strong>Proposition.</strong> <br>
  The random vector $\epsilon$ defined by (12) satisfies
  $$
  \epsilon \sim \mathcal{N}(0, D), \tag{20}
  $$
  where $D$ is a diagonal matrix with positive entries on the diagonal.
  </p>
</blockquote>

**Proof.** The result follows immediately upon viewing (14) as a projection
in a suitable inner product space, and then applying the
[Hilbert projection theorem](https://en.wikipedia.org/wiki/Hilbert_projection_theorem).
In particular, note that all of the $x^{(j)}$ and $\epsilon^{(j)}$ are zero
mean, square integrable random variables. We can thus consider the Hilbert
space of all such random variables with inner product defined by
$\langle \psi, \eta \rangle := \mathbb{E}[\psi \eta]$. Under this interpretation,
we see that (15) can be rewritten as
$$
\sum_{k=1}^{j-1} \beta_k^{(j)} \epsilon^{(k)} =
\text{argmin}_{x^\prime \in \mathcal{E}^{(j)}} \lVert x - x^\prime \rVert, \tag{21}
$$
where $\mathcal{E}^{(j)}$ is the subspace spanned by
$\epsilon^{(1)}, \dots, \epsilon^{(j)}$
and $\lVert \cdot \rVert$ is the norm induced by $\langle \cdot, \cdot \rangle$.
Since $\epsilon^{(j)}$ is the residual associated with the projection (21),
the Hilbert projection theorem gives the optimality condition
$\epsilon^{(j)} \perp \mathcal{E}^{(j)}$; that is,
$$
\langle \epsilon^{(j)}, \epsilon^{(k)} \rangle
= \mathbb{E}[\epsilon^{(j)} \epsilon^{(k)}]
= \text{Cov}[\epsilon^{(j)}, \epsilon^{(k)}] = 0, \qquad k = 1, \dots, j-1. \tag{22}
$$
This implies that all of the residuals are pairwise uncorrelated, and hence
$D = \text{Cov}[\epsilon]$ is diagonal. We know from (17) that
$x = L\epsilon$; since $C = \text{Cov}[x]$ is positive definite, then
$\text{Cov}[\epsilon]$ must also be positive definite. Thus, the diagonal
entries of $D$ must be strictly positive. $\qquad \blacksquare$

Using the recursive regression procedure in (14) and (15), we have constructed
$\epsilon$ and $L$ satisfying $C = LDL^\top$, where $D = \text{Cov}[\epsilon]$
is a diagonal matrix with positive diagonal entries, and $L$ is lower triangular.
By the uniqueness of the modified Cholesky decomposition (noted in the
introduction) it follows that we have precisely formed the unique matrices
$L$ and $D$ defining the modified Cholesky decomposition of $C$.

TODO: connect the conditional covariance and regression interpretations by
using the known form of the regression coefficient. The conditional
covariance forms a portion of this coefficient expression.

## Connection to the Conditional Covariance Perspective
At this point we have two different interpretations of the (modified) Cholesky
decomposition of $C$: (i.) the conditional covariance perspective provided
in (12); and (ii.) the regression formulation given in (14), (15), and (18).
In particular, these results yield interpretations of the entries of $L$.
Assuming we use the factorization $C = LDL^\top$, and letting $i > j$, the
above results give
$$
\ell_{ij} =
\frac{\text{Cov}[x^{(i)}, x^{(j)}| x^{(1)}, \dots, x^{(j-1)} ]}{\text{Var}[x^{(j)}|x^{(1)}, \dots, x^{(j-1)}]} (23)
$$
and
$$
\ell_{ij} =
\beta^{(i)}_j = \text{Cov}[\epsilon^{(1:j-1)}]^{-1} \text{Cov}[\epsilon^{(1:j-1)}, x_i]. \tag{24}
$$
In (24) we are using the notation
$\epsilon^{(1:j-1)} := (\epsilon^{(1)}, \dots, \epsilon^{(j-1)})$, and inserting the
closed-form solution of the optimization problem (14). As a side note, by combining
(23) and (24), we see that
$$
\beta^{(i)}_j = \frac{\text{Cov}[x^{(i)}, x^{(j)}| x^{(1)}, \dots, x^{(j-1)} ]}{\text{Var}[x^{(j)}|x^{(1)}, \dots, x^{(j-1)}]}, \tag{25}
$$
which shows that the regression coefficient (24), which is a function of
variances and covariances of the residuals $\epsilon$, can alternatively be
written using conditional variances and covariances of the original $x$
variables.

{% endkatexmm %}
