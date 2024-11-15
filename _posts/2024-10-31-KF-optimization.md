---
title: Optimization Viewpoints on Kalman Methodology
subtitle: Viewing the Kalman Filter and its variants from an optimization perspective.
layout: default
date: 2024-10-31
keywords: UQ
published: true
---

{% katexmm %}
Throughout this post, we utilize the following notation for inner products and
norms weighted by a positive definite matrix $C$:
\begin{align}
\langle v, v^\prime \rangle_C &:= \langle C^{-1}v, v^\prime\rangle = v^\top C^{-1}v^\prime \newline
\lVert v \rVert_C^2 &:= \langle v, v\rangle_C.
\end{align}
We will also require consideration of generalizing these expressions when $C$
is strictly positive semidefinite. The notation
$\lVert \cdot \rVert$ and $\langle \cdot, \cdot \rangle$ denotes
the standard Euclidean norm and inner product, respectively. For a matrix $C$,
we will denote by $\mathcal{R}(C)$ and $\mathcal{N}(C)$ its range (column space) and
kernel (null space), respectively.
{% endkatexmm %}

# Setup
{% katexmm %}
We start by considering the following optimization problem
<blockquote>
  <p><strong>Optimization Formulation with Positive Definite $C$.</strong> <br>
  \begin{align}
  v_{\star} &:= \text{argmin}_{v \in \mathbb{R}^d} J(v) \tag{1} \newline
  J(v) &= \frac{1}{2} \lVert y - h(v)\rVert_R^2 + \frac{1}{2} \lVert v - \hat{v}\rVert_C, \tag{2}
  \end{align}
  </p>
</blockquote>
where the objective $J(v)$ is defined with respect to a map
$h: \mathbb{R}^d \to \mathbb{R}^p$, fixed vectors $\hat{v} \in \mathbb{R}^d$
and $y \in \mathbb{R}^p$, and positive definite matrices
$R \in \mathbb{R}^{p \times p}$ and $C \in \mathbb{R}^{d \times d}$. A solution to
(1) can be viewed as a maximum a posteriori (MAP) estimate under the Bayesian model
\begin{align}
y|v &\sim \mathcal{N}(h(v), R) \tag{3} \newline
v &\sim \mathcal{N}(\hat{v}, C). \tag{4}
\end{align}
In the case that $h(v)$ is linear (i.e., $h(v) = Hv$) then (1) assumes the form of a
standard $L^2$ regularized least squares problem, admitting a closed-form solution.
In this special case, (1) also defines the optimization problem that is solved to
derive the analysis update of the standard Kalman filter. Under this interpretation,
$\hat{v}$ is the mean of the current forecast distribution, $H$ is the linear
observation operator, $R$ is the observation covariance, $C$ is the covariance
of the current forecast distribution, and the solution $v_{\star}$ gives the mean
of the Gaussian filtering distribution. Formulation (1) also provides the basis for
deriving variants of the Kalman filter, including the ensemble Kalman filter (EnKF).
For now, we maintain a generic view of (1) in order to investigate its
mathematical properties. Later we will make explicit links with Kalman filter
methods.
{% endkatexmm %}

# Generalizing the Optimization Problem
## Motivation and Setup
{% katexmm %}
While the formulation (1) is perfectly adequate when $C$ is positive definite,
it is not well-defined when $C$ may be only positive semidefinite. In
particular, the inverse of $C$ may not exist and hence the term
$\lVert v - \hat{v}\rVert_C$ is no longer well-defined. Our first goal is to
consider a suitable modification of the optimization problem (1) that is
well-defined in this more general setting. The reason for considering this is
that various filtering algorithms consider setups similar to (1) where the
matrix $C$ is replaced by a low-rank approximation. For example, the EnKF defines
$C$ using a sample covariance estimator, whose rank cannot exceed the number of
samples used in the estimator. Thus, if the state dimension $d$ exceeds the
number of samples, then this matrix will fail to be positive definite. We will
discuss specifics regarding the EnKF later on, but for the time being keep the
discussion generic. Throughout this section, we will consider a few different
optimization formulations that are valid when $C$ is positive semidefinite;
in the following section, we will pursue solutions of the problems formulated
here.

Let $A \in \mathbb{R}^{d \times J}$ be a matrix such that
$$
C = AA^\top. \tag{5}
$$
This implies that
$$
v^\top C v = v^\top AA^\top v = \lVert Av \rVert^2 \geq 0, \tag{6}
$$
so $C$ is indeed positive semidefinite (as we have assumed). However, when
$J < d$ the columns of $A$ must be linearly dependent so there is some
$v$ such that $Av = 0$. This implies that $C$ is not positive definite when
$J < d$. Note that the range $C$ is equal to that of $A$; i.e.,
$$
\mathcal{R}(C) = \mathcal{R}(A). \tag{7}
$$

## Redefining the Objective Function
Our goal is now to extend (1) to the case where $C$ need not be strictly
positive definite. To start, note that when $C$ *is* positive definite, we have
$$
\lVert v - \hat{v}\rVert^2_C
= \langle C^{-1}(v-\hat{v}), v-\hat{v}\rangle
= \langle b, v-\hat{v} \rangle, \tag{8}
$$
where $b \in \mathbb{R}^d$ solves
$$
Cb = v - \hat{v}. \tag{9}
$$
When $C$ is invertible, the unique $b$ solving (9) is simply found by multiplying
both sides by $C^{-1}$. When $C$ is not invertible, then there may be zero or
infinitely many such solutions. As long as there is at least one solution to (9)
we will see that we can give meaning to the expression
$\lVert v - \hat{v}\rVert^2_C$ even when $C$ is only positive semidefinite.

### Constrained optimization over $(v,b)$
Let's consider the case where there is at least
one solution to (9); i.e., $v - \hat{v} \in \mathcal{R}(C)$. We define
$$
\mathcal{B}(v) := \left\{b \in \mathbb{R}^d : Cb = v - \hat{v} \right\}, \tag{10}
$$
the set of all solutions with respect to the input vector $v$. We therefore
might consider generalizing (1) to
$$
\text{argmin}_{v \in \mathbb{R}^d} \text{argmin}_{b \in \mathcal{B}(v)}
\left(\frac{1}{2} \lVert y - h(v)\rVert_R^2 + \frac{1}{2} \langle b, v - \hat{v}\rangle\right), \tag{11}
$$
where we are implicitly restricting the solution space to $v$ such that
$\mathcal{B}(v)$ is not empty. To encode this constraint more explicitly, we
can modify (11) as
$$
\text{argmin}_{(v,b) \in \mathcal{V}} \tilde{J}(v,b), \tag{12}
$$
where
$$
\tilde{J}(v,b)
:= \frac{1}{2} \lVert y - h(v)\rVert_R^2 + \frac{1}{2} \langle b, v-\hat{v}\rangle \tag{13}
$$
and
$$
\mathcal{V} := \left\{(v,b) : Cb = v - \hat{v} \right\}.
$$
Note that if $C$ is positive definite, then (12) reduces to (1). We can
encode this constraint in an unconstrained optimization problem by
leveraging the method of Lagrange multipliers. Introducing the
Lagrange multiplier $\lambda \in \mathbb{R}^d$ yields the following
formulation.
<blockquote>
  <p><strong>Lagrange Multiplier Formulation.</strong> <br>
  \tilde{J}(v,b,\lambda)
  = \frac{1}{2} \lVert y - h(v)\rVert_R^2 +
  \frac{1}{2} \langle b, v-\hat{v}\rangle +
  \langle \lambda, Cb - v + \hat{v} \rangle. \tag{14}
  </p>
</blockquote>

### Removing dependence on $b$
In the previous section, we extended the optimization problem to consider
optimizing over pairs $(v,b)$ in order to deal with the fact that $C$ may not
be invertible. The below result shows that, for a fixed $v$, the objective
$\tilde{J}(v,b)$ in (12) is actually independent of the particular choice of
$b \in \mathcal{B}(v)$. This fact will allow us to remove the need to jointly
optimize over $(v,b)$.
<blockquote>
  <p><strong>Proposition.</strong> <br>
  Let $v \in \mathbb{R}^d$ be a vector such that there is at least one solution
  $b \in \mathbb{R}^d$ to $Cb = v - \hat{v}$. Then $\tilde{J}(v,b)$, defined in
  (12), is constant for any choice of $b$ solving this linear system.
  </p>
</blockquote>

**Proof.** Let $b, b^\prime \in \mathcal{B}(v)$ such that
$Cb = Cb^\prime = v - \hat{v}$. It thus follows that
$$
\langle b, v - \hat{v}\rangle - \langle b^\prime, v - \hat{v}\rangle
= \langle b - b^\prime, v - \hat{v}\rangle
= \langle b - b^\prime, Cb\rangle
= \langle C(b - b^\prime), b\rangle
= \langle 0, b\rangle
= 0,
$$
where we have used the linearity of the inner product and the fact that $C$ is
symmetric. Since the inner product term in (12) is the only portion with
dependence on $b$, it follows that $\tilde{J}(v,b) = \tilde{J}(v,b^\prime)$.
$\qquad \blacksquare$

Thus, for each $v$ with $\mathcal{B}(v)$ non-empty, we can simply pick any element
from $b \in \mathcal{B}(v)$ to insert into $\langle b, v-\hat{v}\rangle$. The objective
will be well-defined since the above result verifies that the specific choice of
$b$ is inconsequential. A natural choice is to choose the $b$ of minimal norm. That is,
for $\mathcal{B}(v)$ non-empty, set
$$
b^{\dagger} := \text{argmin}_{b \in \mathcal{B}(v)} \lVert b \rVert. \tag{15}
$$
This unique minimal norm solution is guaranteed to exist and is conveniently given
by the Moore-Penrose pseudoinverse
$$
b^{\dagger} = C^{\dagger}(v - \hat{v}) = (AA^\top)^{\dagger}(v - \hat{v}). \tag{16}
$$
Note that when $C$ is positive definite, $C^{\dagger} = C^{-1}$. We can now eliminate
the requirement to optimize over $b$ in (12), (13), and (14). The optimization problem
now assumes the following form.
<blockquote>
  <p><strong>Constrained Optimization using Pseudoinverse</strong> <br>
  \begin{align}
  v_{\star}
  &:= \text{argmin}_{v \in \mathcal{V}} \tilde{J}(v) \tag{17} \newline
  \tilde{J}(v)
  &:= \frac{1}{2} \lVert y - h(v)\rVert_R^2 + \frac{1}{2} \langle (AA^\top)^{\dagger}(v - \hat{v}), v-\hat{v}\rangle \tag{18} \newline
  \mathcal{V} &:= \\{v \in \mathbb{R}^d : v-\hat{v} \in \mathcal{R}(A)\\}. \tag{19}
  \end{align}
  </p>
</blockquote>

An important consequence of this formulation is that any solution
$v_{\star}$ must lie in the affine space $\hat{v} + \mathcal{R}(A)$; this is
immediate from the definition of $\mathcal{V}$. In particular, if $C$ is positive
definite then $\mathcal{R}(A) = \mathbb{R}^d$ so the solution space is
unconstrained. Otherwise, the rank of $C$ imposes a constraint on the solution
space.

<blockquote>
  <p><strong>Proposition.</strong> <br>
  A solution to the optimization problem (17) is constrained to the affine
  space $\hat{v} + \mathcal{R}(A) = \hat{v} + \mathcal{R}(C)$, which has rank
  at most $J$.
  </p>
</blockquote>

### Unconstrained Optimization
We go one step farther in the problem formulation before turning to solutions
in the next section. Note that, in contrast to (1), (17) is a *constrained*
optimization problem. We now re-formulate (17) as an unconstrained problem,
which will be more convenient when considering solutions.
As noted above, the solution space of (17) is given by the affine space
$\hat{v} + \mathcal{R}(A)$. Any vector $v$ in this space can thus be written as
$$
v = \hat{v} + \sum_{j=1}^{J} w_j a_j = \hat{v} + Aw, \tag{20}
$$
for some weight vector $w := (w_1, \dots, w_J)^\top \in \mathbb{R}^J$. We
have denoted the $j^{th}$ column of $A$ by $a_j \in \mathbb{R}^d$. If the
columns of $A$ are linearly independent, then $a_1, \dots, a_J$ provide a basis
for $\mathcal{R}(A)$. If not, then there will be some redundancy in the weights
and the representation (20) will not be unique. We can now substitute
$\hat{v} + Aw$ for $v$ in (18) and thus formulate the optimization over the
weights $w$:
$$
\tilde{J}(w)
:= \frac{1}{2} \lVert y - h(\hat{v} + Aw)\rVert_R^2 +
\frac{1}{2} \langle (AA^\top)^{\dagger}Aw, Aw\rangle. \tag{21}
$$

We can simplify this even further using the following properties of the
[pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse):

1. By definition, the pseudoinverse satisfies  $A = A(A^{\dagger}A)$.
2. The matrix $A^{\dagger}A$ is a projection onto the rowspace of $A$; in
particular it is idempotent (i.e., $(A^{\dagger}A)^2 = A^{\dagger}A$) and
symmetric.
3. The following identity holds: $A^{\dagger} = A^\top (AA^\top)^{\dagger}$.

We can thus simplify the second term in (21) as
\begin{align}
\langle (AA^\top)^{\dagger}Aw, Aw\rangle
= \langle A^\top(AA^\top)^{\dagger}Aw, w\rangle
= \langle (A^{\dagger}A)w, w\rangle
&= \langle (A^{\dagger}A)^2w, w\rangle \newline
&= \langle (A^{\dagger}A)w, (A^{\dagger}A)w\rangle \newline
&= \lVert (A^{\dagger}A)w \rVert^2, \tag{22}
\end{align}
where the second equality uses the third pseudinverse property above.
The third and fourth equalities follow from the fact that $A^{\dagger}A$
is idempotent and symmetric. We can thus re-parameterize as
$$
\tilde{w} := (A^{\dagger}A)w, \tag{23}
$$
and note that by the first pseudoinverse property we have
$$
Aw = A(A^{\dagger}A)w = A\tilde{w}, \tag{24}
$$
which allows us to replace $Aw$ by $A\tilde{w}$ in the first term in
(21). We obtain
$$
\tilde{J}(\tilde{w})
:= \frac{1}{2} \lVert y - h(\hat{v} + A\tilde{w})\rVert_R^2 +
\frac{1}{2} \lVert \tilde{w} \rVert^2 . \tag{25}
$$
We see that (25) is the sum of a model-data fit term plus a simple $L^2$ penalty
on the weights. This final formulation is summarized below, where we have
re-labelled $\tilde{w}$ as $w$ to lighten notation.
<blockquote>
  <p><strong>Unconstrained Formulation.</strong> <br>
  The optimization problem in (17) can equivalently be formulated as the
  unconstrained problem
  $$
  v_{\star} := \text{argmin}_{w \in \mathbb{R}^{J}} \tilde{J}(w) \tag{26}
  $$
  with objective function
  $$
  \tilde{J}(w) := \frac{1}{2} \lVert y - h(\hat{v} + Aw)\rVert_R^2 +
  \frac{1}{2} \lVert w \rVert^2. \tag{27}
  $$
  </p>
</blockquote>

One might initially take issue with the claim that (26) is actually unconstrained
given that it relies on the re-parameterization (23), which constrains the
weights to lie in the range of $A^{\dagger}A$. However, recalling that
$A^{\dagger}A$ is an orthogonal projection, we have the following projection
property:
$$
\lVert \tilde{w} \rVert^2
= \lVert (A^{\dagger}A)w \rVert^2 \leq \lVert w \rVert^2. \tag{28}
$$
In other words, allowing the weights to be unconstrained can only increase the
objective function (since switching $w$ and $\tilde{w}$ has no effect on the
first term in (26)), so we are justified in considering the weights to be
unconstrained in (26). In fact, we could have jumped right to this conclusion
from (22) and avoided needing to define $\tilde{w}$ at all.
{% endkatexmm %}

# Solving the Optimization Problem.
{% katexmm %}
With the optimization problem (17) defined, we now consider the characterization
of its solution. In general, this is challenging due to the potential nonlinearity
of $h(\cdot)$. We start by considering the simplified linear case, before returning
to the more general setting.

## Linear $h(\cdot)$
Assume that $h(v) = Hv$ for some matrix $H \in \mathbb{R}^{p \times d}$. Under
a data assimilation interpretation, this corresponds to the common assumption
of a linear observation operator.  
{% endkatexmm %}

# References
1. Data Assimilation Fundamentals (Vossepoel, Evensen, and van Leeuwen; 2022)
2. Inverse Problems and Data Assimilation (Stuart, Taeb, and Sanz-Alonso)
3. Ensemble Kalman Methods with Constraints (Albers et al, 2019)

# TODOs
Introduce the EnKF optimization by extending the state space.
