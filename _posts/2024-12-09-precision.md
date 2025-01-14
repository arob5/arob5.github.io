---
title: Precision Matrices, Partial Correlations, and Conditional Independence
subtitle: Interpreting the inverse covariance matrix for Gaussian distributions.
layout: default
date: 2025-01-11
keywords:
published: true
---

In this post, we explore how the precision (inverse covariance) matrix for a
Gaussian distribution encodes conditional dependence relations between the
underlying variables. We also introduce the notions of conditional and
partial correlation, and prove their equivalence in the Gaussian setting.
Ideas along these lines are widely used in areas such as probabilistic graphical
models (e.g., Gaussian Markov random fields and scalable Gaussian processes)
and high-dimensional covariance estimation. A natural complement to this
topic is my [post](https://arob5.github.io/blog/2024/12/25/Cholesky-cov/) on the
Cholesky decomposition of a covariance matrix, which provides an alternate route
to analyzing conditional dependence structure. The main source for this post
is {% cite LauritzenGraphicalModels %}, which is freely available online and
provides an excellent rigorous introduction to graphical models in statistics.

# Setup and Notation
{% katexmm %}
Throughout this post we consider a set of random variables $x_1, \dots, x_n$,
each taking values in $\mathbb{R}$. For an (ordered) index set
$A \subseteq \{1,\dots,n\}$, we write $x_A$ to denote the column vector
$[x_i]_{i \in A}$ that retains the ordering of $A$. We will use the convention
that $B := \{1,\dots,n\} \setminus A$ is the complement of $A$, and use the
shorthand $x_{\sim i}$ for the vector of all variables excluding $x_i$.
We also write
$x_i \perp x_j|x_B$ to mean that $x_i$ and $x_j$ are conditionally independent
given $x_B$.
{% endkatexmm %}

# The Precision Matrix of a Gaussian
{% katexmm %}
In this section we will explore how the precision matrix of a Gaussian
distribution is closely related to the conditional dependence structure of the
random variables $x_1, \dots, x_n$. Throughout this section, we assume
$$
x = (x_1, \dots, x_n)^\top \sim \mathcal{N}(m, C), \tag{1}
$$
where $C$ is positive definite. Hence, it is invertible, and we denote its
inverse by
$$
P := C^{-1}, \tag{2}
$$
which we refer to as the *precision matrix*. The precision inherits positive
definiteness from $C$. In some contexts
(e.g., {% cite LauritzenGraphicalModels %}), $P$ is also called the
*concentration matrix*.

Throughout this section, our focus will be on the dependence between two
variables $x_i$ and $x_j$, conditional on all others. Thus, let's define the
index sets $A := \{i,j\}$ and $B := \{1,\dots,n\} \setminus \{i,j\}$. We
partition the joint Gaussian (1) (after possibly reordering the variables) as
\begin{align}
\begin{bmatrix} x_A \newline x_B \end{bmatrix}
&\sim \mathcal{N}\left(
\begin{bmatrix} m_A \newline m_B \end{bmatrix},
\begin{bmatrix} C_A & C_{AB} \newline C_{BA} & C_B \end{bmatrix}
\right). \tag{3}
\end{align}

Our focus is
thus on the conditional distribution of $x_A|x_B$. We recall that conditionals
of Gaussians are themselves Gaussian, and that the conditional covariance
takes the form
$$
C_{A|B} := \text{Cov}[x_A|x_B] = C_A - C_{AB}C_B^{-1}C_{BA}. \tag{4}
$$
I go through these derivations in depth in [this](https://arob5.github.io/blog/2024/05/19/Gaussian-Measures-multivariate/) post. For our present purposes, it is important to appreciate the
connection between the conditional covariance (4) and the joint precision $P$.
To this end, let's consider partitioning the precision in the same manner
as the covariance:
\begin{align}
P &= C^{-1} = \begin{bmatrix} P_A & P_{AB} \newline P_{BA} & P_B \end{bmatrix}. \tag{5}
\end{align}
The above blocks of $P$ can be obtained via a direct application of partitioned
matrix inverse identities from linear algebra (see, e.g., James E. Pustejovsky's
[post](https://jepusto.com/posts/inverting-partitioned-matrices/) for some nice background).
Applying the partitioned matrix identity to (5) yields
\begin{align}
P_A &= [C_A - C_{AB}(C_B)^{-1}C_{BA}]^{-1} \tag{6} \newline
P_{AB} &= -(C_B)^{-1} C_{BA}P_A. \tag{7}
\end{align}
Notice in (6) that $P_A$, the upper-left block of the joint precision, is
precisely equal to the inverse of the conditional covariance
$C_{A|B}$ given in (4). We denote this conditional precision by
$$
P_{A|B} := (C_{A|B})^{-1}. \tag{8}
$$
We summarize this important connection below.
<blockquote>
  <p><strong>Joint and Conditional Precision.</strong> <br>
  The precision matrix of the conditional distribution $x_A|x_B$ is given by
  $$
  P_{A|B} = (C_{A|B})^{-1} = P_A. \tag{9}
  $$
  In words, the conditional precision is obtained by deleting the rows and
  columns of the joint precision $P$ that involve the conditioning variables
  $x_B$.
  </p>
</blockquote>

This connection also leads us to our main result, which states that conditional
independence can be inferred from zero entries of the precision matrix. This
result follows immediately by rearranging (9) to
$$
C_{A|B} = (P_A)^{-1} \tag{10}
$$
and noting that $P_A$ is simply a two-by-two matrix that we can consider
inverting by hand.
<blockquote>
  <p><strong>Zeros in Precision imply Conditional Independence.</strong> <br>
  An entry $P_{ij}$, $i \neq j$ of the joint precision matrix is zero if and only
  if $x_i$ and $x_j$ are conditionally independent, given all other variables.
  That is,
  $$
  P_{ij} = 0 \iff \text{Cov}[x_i,x_j|x_B] = 0 \iff x_i \perp x_j | x_B, \tag{11}
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
  P_{ii} = \text{Var}[x_i|x_{\sim i}]^{-1} \tag{12}
  $$
  </p>
</blockquote>

**Proof.** We know from (9) that $C_{A|B} = (P_A)^{-1}$. Let $A := \{i\}$ and
$B := \{1, \dots, n\} \setminus \{i\}$. It follows that
$$
\text{Var}[x_i|x_{\sim i}] = C_{A|B} = (P_A)^{-1} = P_{ii}^{-1}. \qquad \qquad \blacksquare
$$

{% endkatexmm %}

# Notions of Linear Conditional Dependence
In this section, we introduce two analogs of the correlation coefficient that
quantify the linear dependence between two variables, conditional on another
set of variables. We show that if all random variables are jointly Gaussian,
then the two notions coincide.

{% katexmm %}
## Conditional Correlation
We start by defining *conditional correlation*, which is nothing more than
the typical notion of correlation but defined with respect to a conditional
probability measure. We give the definition with respect to our current setup,
but of course the notion can be generalized.

<blockquote>
  <p><strong>Definition: conditional correlation.</strong> <br>
  For a pair of indices $\{i,j\}$ and its complement
  $B := \{1, \dots, n\} \setminus \{i,j\}$, we define the conditional covariance
  between $x_i$ and $x_j$, given $x_B$, as
  $$
  \text{Cov}[x_i,x_j|x_B] := \mathbb{E}^B\left[(x_i-\mathbb{E}^B[x_i])(x_j-\mathbb{E}^B[x_j]) \right], \tag{13}
  $$
  where we denote $\mathbb{E}^B[\cdot] := \mathbb{E}[\cdot|x_B]$. The conditional
  correlation is then defined as usual by normalizing the covariance (13):
  $$
  \text{Cor}[x_i,x_j|x_B]
  := \frac{\text{Cov}[x_i,x_j|x_B]}{\sqrt{\text{Var}[x_i|x_B]\text{Var}[x_j|x_B]}}. \tag{14}
  $$
  </p>
</blockquote>

The conditional correlation is simply a correlation where all expectations
involved are conditional on $x_B$. It is sometimes also denoted as
$\rho_{ij|B} := \text{Cor}[x_i,x_j|x_B]$.

## Partial correlation
We now consider an alternative notion of conditional linear dependence that is
defined with respect to underlying linear regression models. For generality,
we provide a definition that quantifies dependence between sets of variables
$x_{A_1}$ and $x_{A_2}$, after removing the confounding effect of a third
set $x_{B}$. However, our primary interest will be in the special case
$A_1 = \{i\}$ and $A_2 = \{j\}$, which aligns with the conditional correlation
definition given above.

<blockquote>
  <p><strong>Definition: partial correlation.</strong> <br>
  For a set of random variables $x_1, \dots, x_n$, let
  $A_1,A_2 \subset \{1, \dots, n\}$ be index sets and $B$ a third index set
  disjoint from the other two. Define the linear
  regression coefficients
  \begin{align}
  \alpha_{A_1}^\star, \beta_{A_1}^\star &:= \text{argmin}_{\alpha,\beta} \mathbb{E}\lVert x_{A_1} - \alpha + \beta_{A_1}^\top x_B)\rVert^2 \tag{15} \newline
  \alpha_{A_2}^\star, \beta_{A_2}^\star &:= \text{argmin}_{\alpha,\beta} \mathbb{E} \lVert x_{A_2} - (\alpha + \beta_{A_2}^\top x_B)\rVert^2,
  \end{align}
  and associated residuals
  \begin{align}
  e_{A_1} &:= x_{A_1} - [\alpha_{A_1}^\star + (\beta_{A_1}^\star)^\top x_B] \tag{16} \newline
  e_{A_2} &:= x_{A_2} - [\alpha_{A_2}^\star + (\beta_{A_2}^\star)^\top x_B].
  \end{align}
  The partial correlation coefficient between $x_{A_1}$ and $x_{A_2}$ given
  $x_B$ is defined as
  $$
  \rho_{A_{1}A_{2} \cdot B} := \text{Cor}[e_{A_1}, e_{A_2}]. \tag{17}
  $$
  </p>
</blockquote>

The intuition here is that the residuals from the regressions contain variation
that is unexplained by $x_B$, so that $\rho_{A_{1}A_{2} \cdot B}$ quantifies the
linear dependence between $x_{A_1}$ and $x_{A_2}$ after removing the effect of
$x_B$. We emphasize the importance of the word "linear" here, as linearity plays
a role in two different ways in the above definition. Recall that the typical
correlation coefficient measures linear dependence, so the statement in (17)
is a measure of linear dependence between the residuals. Moreover, the
residuals themselves are defined via linear regressions, and hence the sense
in which the effect of $x_B$ is "removed" also relies on linearity
assumptions. Note also that we are allowing $x_{A_1}$ and $x_{A_2}$ to be sets of
variables, so the linear regressions considered above are multi-output
regressions, which can essentially be thought of as a set of independent
univariate regressions, since the loss function simply sums the error across
the outputs. Thus, in general, the $\alpha$s and $\beta$s are vectors and
matrices, respectively.

## Equivalence for Gaussian Distributions
We now introduce the additional assumption that $x_1, \dots x_n$ are jointly
Gaussian and show that in this setting the definitions of conditional and
partial correlation are equivalent. For more detailed discussion on these
connections, see {% cite BabaPartialCondCor %} and {% cite LawrancePartialCondCor %}.

<blockquote>
  <p><strong>Conditional and Partial Correlation for Gaussians.</strong> <br>
  Suppose that $x_1, \dots, x_n$ are jointly Gaussian, with positive definite
  covariance $C$. For a pair of indices $\{i,j\}$ and its complement
  $B := \{1, \dots, n\} \setminus \{i,j\}$, we have
  $$
  \rho_{ij \cdot B} = \text{Cor}[x_i,x_j|x_B]. \tag{18}
  $$
  That is, under joint Gaussianity the notions of conditional and partial
  correlation coincide.
  </p>
</blockquote>

**Proof.** The result will be proved by establishing that
$$
\text{Cov}[e_i,e_j] = \text{Cov}[x_i,x_j|x_B] \tag{19}
$$
for any $i$ and $j$ (possibly equal), where $e_i$ and $e_j$ are the residual
random variables defined in (16) with $A_1 := \{i\}$ and $A_2 := \{j\}$.
The $i=j$ case establishes the equality of the variances, meaning that the
correlations will also be equal. To this end, we start by noting that the
righthand side in (19) is given by the relevant entry of the matrix
$$
C_{A|B} = C_A - C_{AB}C_B^{-1}C_{BA}.
$$
where $A := \{i,j\}$. Recall that $C_{A|B}$ is defined in (4). By extracting
the relevant entry of $C_{A|B}$ we have
$$
\text{Cov}[x_i,x_j|x_B] = C_{ij} - C_{iB}C_B^{-1}C_{Bj}. \tag{20}
$$
We now show that $\text{Cov}[e_i,e_j]$ reduces to (20). We recall that the
conditional expectation for a square-integrable random variable is given by
the projection
$$
\mathbb{E}[x_i|x_B] = \text{argmin}_{g} \lVert x_i - g(x_B)\rVert^2, \tag{21}
$$
where the minimum is considered over all $x_B$-measurable functions $g$. In our
present Gaussian setting, this is solved by
$$
\mathbb{E}[x_i|x_B] = \mathbb{E}[x_i] + \langle k_i, x_B-\mathbb{E}[x_B]\rangle, \tag{22}
$$
where $k_i = \text{Cov}[x_B]^{-1} \text{Cov}[x_B,x_i] = C_B^{-1}C_{Bi}$. See
[this](https://arob5.github.io/blog/2024/05/19/Gaussian-Measures-multivariate/)
post for the derivation of (22). The important thing to notice is that
(22) is a linear function of $x_B$, which means that it solves the linear
regression problem (15); i.e.,
$\alpha^{\star}_{A_1} + (\beta^{\star}_{A_1})^\top x_B = \mathbb{E}[x_i|x_B]$.
Thus,
$$
e_i = x_i - \mathbb{E}[x_i|x_B] = x_i - \mathbb{E}[x_i] -
\langle k_i, x_B-\mathbb{E}[x_B]\rangle. \tag{23}
$$
and similarly for $e_j$. Notice that the constants $\mathbb{E}[x_i]$ and
$\mathbb{E}[x_B]$ will be dropped when taking covariances, so it suffices to
treat these as zero. We thus have
\begin{align}
\text{Cov}[e_i,e_j]
&= \text{Cov}[x_i-\langle k_i, x_B\rangle,x_j-\langle k_j, x_B\rangle] \newline
&= C_{ij} + k_i^\top C_B k_j - C_{iB}k_j - k_i^\top k_j - k_i^\top C_{Bj} \newline
&= C_{ij} + C_{iB}C_B^{-1}C_B C_B^{-1}C_{Bj} - 2C_{iB}C_B^{-1}C_{Bj} \newline
&= C_{ij} - C_{iB}C_B^{-1}C_{Bj}.
\end{align}
We see that the final expression agrees with (20), so the result is
proved. $\qquad \blacksquare$
{% endkatexmm %}

# Precision Matrix and Partial Correlation

{% katexmm %}
Having defined partial and conditional correlation, we now return to the
question of interpreting the off-diagonal elements of a Gaussian precision
matrix. Throughout this section we assume that $x_1, \dots, x_n$ are jointly
Gaussian with positive definite covariance $C$ and precision matrix $P$.
The following definition normalizes the precision
matrix, analogous to the way a covariance matrix is normalized to produce a
correlation matrix.

<blockquote>
  <p><strong>Normalized Precision.</strong> <br>
  Define the normalized precision $\bar{P}$ as the matrix with elements
  $$
  \bar{P}_{ij} := \frac{P_{ij}}{\sqrt{P_{ii}P_{jj}}}. \tag{24}
  $$
  </p>
</blockquote>

We are now ready to establish the connection between the (normalized) precision
and the notions of conditional and partial correlation. Note that the diagonal
elements of $\bar{P}$ satisfy $\bar{P}_{ii}=1$. The following result interprets
the off-diagonal elements.

<blockquote>
  <p><strong>Off-Diagonal Entries of Precision.</strong> <br>
  Assume that $x_1, \dots, x_n$ are jointly Gaussian with normalized precision
  matrix $\bar{P}$. Let $\{i,j\}$ be a pair of distinct indices and
  $B := \{1,\dots,n\} \setminus \{i,j\}$ its complement. Then
  $$
  \bar{P}_{ij} = -\rho_{ij\cdot B} = -\text{Cor}[x_i,x_j|x_B]. \tag{25}
  $$
  In words, the off-diagonal entry $\bar{P}_{ij}$ is equal to the negated
  partial correlation between $x_i$ and $x_j$ given all other variables.
  It is likewise equal to the negated conditional correlation.
  </p>
</blockquote>

**Proof.**
Let $A := \{i,j\}$. Recall from (9) that
\begin{align}
C_{A|B} &= (P_A)^{-1}
= \begin{bmatrix} P_{ii} & P_{ij} \newline P_{ji} & P_{jj} \end{bmatrix}^{-1}
= \gamma \begin{bmatrix} P_{jj} & -P_{ij} \newline -P_{ji} & P_{ii} \end{bmatrix}, \tag{26}
\end{align}
with $\gamma := (P_{ii}P_{jj}-P^2_{ij})^{-1}$.
We have again used the expression for the inverse of a two-by-two matrix.
The equality in (26) gives
\begin{align}
\text{Var}[x_i|x_B] &= \gamma P_{jj} \newline
\text{Var}[x_j|x_B] &= \gamma P_{ii} \newline
\text{Cov}[x_i,x_j|x_B] &= -\gamma P_{ij}.
\end{align}
We thus have
$$
\bar{P}_{ij}
= \frac{P_{ij}}{\sqrt{P_{ii}P_{jj}}}
= -\frac{\text{Cov}[x_i,x_j|x_B]}{\sqrt{\text{Var}[x_i|x_B]\text{Var}[x_j|x_B]}}
= -\text{Cor}[x_i,x_j|x_B].
$$
This establishes the relationship between the precision elements and
conditional correlation. Owing to the Gaussian assumption, the equivalence with
the partial correlation follows from (18). $\qquad \blacksquare$
{% endkatexmm %}

# Additional Resources
In addition to {% cite LauritzenGraphicalModels %}, {% cite BabaPartialCondCor %},
and {% cite LawrancePartialCondCor %}, here is a list of some other resources
that cover similar topics.

- Graphical Models in Applied Mathematical Multivariate Statistics (Whittaker, 1991)
- Dichotomization, Partial Correlation, and Conditional Independence (Vargha et al, 1996)
- A note on the partial correlation coefficient (Fleiss and Tanur, 1971)
- Kernel Partial Correlation Coefficient — a Measure of Conditional Dependence (Huang et al, 2022)
- Back to the basics: Rethinking partial correlation network methodology (Williams and Rast, 2019)
- Some StackExchange posts on precision matrices and partial correlation:
[here](https://stats.stackexchange.com/questions/10795/how-to-interpret-an-inverse-covariance-or-precision-matrix) and [here](https://stats.stackexchange.com/questions/140080/why-does-inversion-of-a-covariance-matrix-yield-partial-correlations-between-ran)
- Wikipedia [article](http://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion) on partial correlation
