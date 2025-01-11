---
title: Precision Matrices, Partial Correlations, and Conditional Independence
subtitle:
layout: default
date: 2025-01-11
keywords:
published: true
---

# Setup and Notation
{% katexmm %}
Throughout this post we will consider the set of random variables
$x := \{x_1, x_2, \dots, x_n\}$, each taking values in $\R$.
We will generically write $p(x_1, x_2, \dots, x_n)$
to denote the joint distribution of these variables, and abuse notation by
using the same symbol $p(\cdot)$ to indicate marginal and conditional
distributions. For example $p(x_i, x_j)$ is the marginal distribution for
$(x_i, x_j)$ and $p(x_i, x_j|x_k)$ is the distribution of $(x_i, x_j)$,
conditional on $x_k$. We will find it convenient to introduce some shorthand
to avoid listing these random variables all the time. For and index set
$A \subseteq \{1, 2, \dots, n\}$ we define $x_A := \{x_i\}_{i \in A}$.
We also write
$\tilde{x}_A := x \setminus x_A := \{x_i\}_{i \notin A}$ to indicate the subset
of all variables in $x$ excluding $x_A$. Finally, we introduce the shorthand
$x_{ij} := \{x_i, x_j\}$ and $x_{i:j} := \{x_i, \dots, x_j\}$ (for $i \leq j$).
Finally, while we have introduced the above notation using sets, we will
utilize the same notation for vectors when relevant. For example, when relevant,
$x_A$ will denote the column vector $[x_i]_{i \in A}$ with entries ordered
according to the specified order of the index set $A$.
{% endkatexmm %}

# The Precision Matrix of a Gaussian
{% katexmm %}
In this section we will explore how the precision matrix of a Gaussian
distribution is closely related to the conditional dependence structure of the
random variables $x_1, \dots, x_n$. Throughout this section, we assume
$$
x = (x_1, \dots, x_n)^\top \sim \mathcal{N}(m, C), \tag{3}
$$
where $C$ is positive definite. Hence, it is invertible, and we denote its
inverse by
$$
P := C^{-1}, \tag{4}
$$
which we refer to as the *precision matrix*. The precision inherits positive
definiteness from $C$. In some contexts
(e.g., {% cite LauritzenGraphicalModels %}), $P$ is also called the
*concentration matrix*.

Throughout this section, our focus will be on the dependence between two
variables $x_i$ and $x_j$, conditional on all others. Thus, let's define the
index sets $A := \{i,j\}$ and $B := \{1,\dots,n\} \setminus \{i,j\}$. We partition
the joint Gaussian (3) (after possibly reordering the variables) as
\begin{align}
\begin{bmatrix} x_A \newline x_B \end{bmatrix}
&\sim \mathcal{N}\left(
\begin{bmatrix} m_A \newline m_B \end{bmatrix},
\begin{bmatrix} C_A & C_{AB} \newline C_{BA} & C_B \end{bmatrix}
\right). \tag{5}
\end{align}

Our focus is
thus on the conditional distribution of $x_A|x_B$. We recall that conditionals
of Gaussians are themselves Gaussian, and that the conditional covariance
takes the form
$$
C_{A|B} := \text{Cov}[x_A|x_B] = C_A - C_{AB}C_B^{-1}C_{BA}. \tag{6}
$$
I go through these derivations in depth in [this](https://arob5.github.io/blog/2024/05/19/Gaussian-Measures-multivariate/) post. For our present purposes, it is important to appreciate the
connection between the conditional covariance (6) and the joint precision $P$.
To this end, let's consider partitioning the precision in the same manner
as the covariance:
\begin{align}
P &= C^{-1} = \begin{bmatrix} P_A & P_{AB} \newline P_{BA} & P_B \end{bmatrix}. \tag{7}
\end{align}
The above blocks of $P$ can be obtained via a direct application of partitioned
matrix inverse identities from linear algebra (see, e.g., James E. Pustejovsky's
[post](https://jepusto.com/posts/inverting-partitioned-matrices/) for some nice background).
Applying the partitioned matrix identity to (7) yields
\begin{align}
P_A &= (C_A - C_{AB}C_B^{-1}C_{BA})^{-1} \tag{8} \newline
P_{AB} &= -C_B^{-1} C_{BA}P_A. \tag{9}
\end{align}
Notice in (8) that $P_A$, the upper-left block of the joint precision, is
precisely equal to the inverse of the conditional covariance
$C_{A|B}$ given in (6). We denote this conditional precision by
$$
P_{A|B} := C_{A|B}^{-1}. \tag{10}
$$
We summarize this important connection below.
<blockquote>
  <p><strong>Joint and Conditional Precision.</strong> <br>
  The precision matrix of the conditional distribution $x_A|x_B$ is given by
  $$
  P_{A|B} = (C_{A|B})^{-1} = P_A. \tag{11}
  $$
  In words, the conditional precision is obtained by deleting the rows and
  columns of the joint precision $P$ that involve the conditioning variables
  $x_B$.
  </p>
</blockquote>

This connection also leads us to our main result, which states that conditional
independence can be inferred from zero entries of the precision matrix. This
result follows immediately by rearranging (11) to
$$
C_{A|B} = (P_A)^{-1} \tag{12}
$$
and noting that $(P_A)^{-1}$ is simply a two-by-two matrix that we can consider
inverting by hand.
<blockquote>
  <p><strong>Zeros in Precision imply Conditional Independence.</strong> <br>
  An entry $P_{ij}$, $i \neq j$ of the joint precision matrix is zero if and only
  if $x_i$ and $x_j$ are conditionally independent, given all other variables.
  That is,
  $$
  P_{ij} = 0 \iff \text{Cov}[x_i,x_j|x_B] = 0 \iff x_i \perp x_j | x_B, \tag{13}
  $$
  where $B := \{1, \dots, n\} \setminus \{i,j\}$.
  </p>
</blockquote>

**Proof.** Setting $A := \{i,j\}$, the above derivation showed
$C_{A|B} = (P_A)^{-1}$, where $P_A$ is the two-by-two block of the joint
precision $P$ corresponding to the variables $x_i$ and $x_j$. Thus,
$$
\text{Cov}[x_i,x_j|x_B] = [C_{A|B}]_{12} = 0 \iff [(P_A)^{-1}]_{12} = 0.
$$
We use the well-known formula
for the inverse of a two-by-two matrix to obtain
\begin{align}
(P_A)^{-1} &= \begin{bmatrix} P_{ii} & P_{ij} \newline P_{ji} & P_{jj} \end{bmatrix}
= \frac{1}{P_{ii}P_{jj}- P_{ij}^2} \begin{bmatrix} P_{jj} & -P_{ji} \newline -P_{ij} & P_{ii}\end{bmatrix}.
\end{align}
Notice that the off-diagonal entries of $P_A$ and $(P_A)^{-1}$ are the same,
up to a minus sign. This means
$$
\text{Cov}[x_i,x_j|x_B] = [C_{A|B}]_{12} = 0 \iff P_{ij} = 0,
$$
so the result is proved. The fact that conditional uncorrelatedness implies conditional
independence follows from the fact that $x_A|x_B$ is Gaussian. $\qquad \blacksquare$

The above result interprets the zero values of off-diagonal elements of the
precision matrix. Later in this post we will revisit the precision, and see
that non-zero values can be interpreted through the lens of partial correlation.
An interpretation of the magnitude of the diagonal elements is more
straightforward, and is given in the below result.

<blockquote>
  <p><strong>Diagonal Entries of Precision.</strong> <br>
  The diagonal entry $P_{ii}$ of the precision gives the reciprocal of the
  variance of $x_i$, conditional on all other variables; that is,
  $$
  P_{ii} = \text{Var}[x_i|x_{\tilde i}]^{-1} \tag{14}
  $$
  </p>
</blockquote>

**Proof.** We know from (12) that $C_{A|B} = (P_A)^{-1}$. Let $A := \{i\}$ and
$B := \{1, \dots, n\} \setminus \{i,j\}$. It follows that
$$
\text{Var}[x_i|x_{\tilde i}] = C_{A|B} = (P_A)^{-1} = P_{ii}^{-1}. \qquad \qquad \blacksquare
$$

{% endkatexmm %}

# Partial correlations
{% katexmm %}
We now turn our focus to the definition of a quantity analogous to a correlation
coefficient, but which measures the linear dependence between two random
variables after the effect of a third confounding variable has been removed.
This notion is made precise below.
<blockquote>
  <p><strong>Definition: partial correlation.</strong> <br>
  For a set of random variables $x_1, \dots, x_n$, let
  $A,B,C \subset \{1, \dots, n\}$ be disjoint index sets. Define the linear
  regression coefficients
  \begin{align}
  \alpha_A^\star, \beta_A^\star &:= \text{argmin}_{\alpha,\beta} \mathbb{E}\lVert x_A - (\alpha + \beta_A^\top x_C)\rVert^2 \newline
  \alpha_B^\star, \beta_B^\star &:= \text{argmin}_{\alpha,\beta} \mathbb{E} \lVert x_B - (\alpha + \beta_B^\top x_C)\rVert^2,
  \end{align}
  and associated residuals
  \begin{align}
  e_{A} &:= x_A - (\alpha_A^\star + (\beta_A^\star)^\top x_C) \newline
  e_{B} &:= x_B - (\alpha_B^\star + (\beta_B^\star)^\top x_C).
  \end{align}
  The partial correlation coefficient between $x_A$ and $x_B$ given $x_C$ is
  defined as
  $$
  \rho_{AB \cdot C} := \text{Cor}[e_A, e_B]. \tag{15}
  $$
  </p>
</blockquote>

The intuition here is that the residuals from the regressions contain variation
that is unexplained by $x_C$, so that $\rho_{AB \cdot C}$ quantifies the linear
dependence between $x_A$ and $x_B$ after removing the effect of $x_C$. We
emphasize the importance of the word "linear" here, as linearity plays a role
in two different ways in the above definition. Recall that the typical
correlation coefficient measures linear dependence, so the statement in (15)
is a measure of linear dependence between the residuals. Moreover, the
residuals themselves are defined via linear regressions, and hence the sense
in which the effect of $x_C$ is "removed" also relies on linearity
assumptions. Note also that we are allowing $x_A$ and $x_B$ to be sets of
variables, so the linear regressions considered above are multi-output
regressions, which can essentially be thought of as a set of independent
univariate regressions, since the loss function simply sums the error across
the outputs. Thus, in general, the $\alpha$s and $\beta$s are vectors and
matrices, respectively.

We should also note briefly that partial correlation is in general distinct
from the concept of *conditional correlation*, though the two coincide when
the variables in question are jointly Gaussian. For details see
{% cite BabaPartialCondCor %} and {% cite LawrancePartialCondCor %}.
{% endkatexmm %}

# Precision Matrix and Partial Correlation
{% katexmm %}
{% endkatexmm %}


1. Estimation and Model Identification for Continuous Spatial Processes (Vecchia)
2. Graphical Models in Applied Mathematical Multivariate Statistics
3. Graphical Models (Lauritzen)
4. https://stats.stackexchange.com/questions/10795/how-to-interpret-an-inverse-covariance-or-precision-matrix
5. http://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
6. https://stats.stackexchange.com/questions/140080/why-does-inversion-of-a-covariance-matrix-yield-partial-correlations-between-ran
7. Dichotomization, Partial Correlation, and Conditional Independence
8. Kernel Partial Correlation Coefficient — a Measure of Conditional Dependence
9. On Conditional and Partial Correlation (Lawrance)
10. A note on the partial correlation coefficient (Fleiss and Tanur)
11. Partial correlation and conditional correlation as measures of conditional independence (Baba et al)
