---
title: Active Subspaces
subtitle: An introduction to dimensionality reduction via active subspaces.
layout: default
date: 2024-05-30
keywords: PCA, Statistics
published: true
---

# Motivation
{% katexmm %}
Consider some (deterministic) function of interest
$$
f: \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}.
$$
While this is a generic
setup, the common motivation is that $f$ conceptualizes a computationally
costly black-box simulation. There are many things one might wish to do
with such a simulation:
- Minimize or maximize $f$ [optimization]
- Integrate $f$ over some domain of interest [numerical integration]
- Find input values $x \in \mathbb{R}^d$ to make
$f(x)$ agree with some observed data [calibration]
- Assess how the the function outputs $f(x)$ vary as the inputs $x$ change
[sensitivity analysis]
- Fit a cheaper-to-evaluate surrogate model that approximates
$f$ [emulation/surrogate modeling]
- Identify subsets of $\mathcal{X}$ that result desirable or undesirable
outputs [contour/excursion set estimation]

All of these tasks are made more difficult when the number of inputs $d$ is large.
However, it has been routinely noted that such high-dimensional settings often
exhibit low-dimensional structure; i.e., low "intrinsic dimensionality." This is to
say that $f$ varies primarily on some low-dimensional manifold embedded in
the higher-dimensional ambient space $\mathcal{X}$. The active subspace method
assumes that the manifold is a linear subspace, and seeks to identify this
subspace using gradient evaluations of $f$.
{% endkatexmm %}

# Ridge Functions
{% katexmm %}
Before introducing the method of active subspaces, we start to build some
intuition regarding the notion of low-dimensional subspace structure in the
map $f$. The extreme case of this idea is a function of the form
\begin{align}
f(x) &= g(Ax), &&A \in \mathbb{R}^{r \times d},
\end{align}
where $r < d$. A function of this type is sometimes called a **ridge function**.
The ridge assumption is a rather stringent one, as it implies that $f$
truly only varies on a subspace of $\mathcal{X}$, and is constant otherwise.
Indeed, let $x, \tilde{x} \in \mathcal{X}$ such that $\tilde{x} \in \text{null}(A)$
(the null space of $A$). Then
\begin{align}
f(\tilde{x}) &= g(A\tilde{x}) = g(0) \newline
f(x + \tilde{x}) &= g(Ax + A\tilde{x}) = g(Ax).
\end{align}
We can consider the input space as partitioned via the direct sum
$\mathcal{X} = \text{null}(A) \oplus \text{row}(A)$ (using the fact that
the orthogonal complement of the null space is the row space). The
ridge function $f$ is responsive only when $x$ changes in $\text{row}(A)$,
and not at all when $x$ only changes in $\text{null}(A)$. Although we have
not defined the concept yet, it seems intuitive that $\text{row}(A)$
would represent the "active subspace" in this case.
{% endkatexmm %}

# Setup and Assumptions
{% katexmm %}
As noted above, the assumption that a function has ridge structure is quite
strong. The notion of active subspaces invokes looser assumptions, assuming that,
on average, the function primarily varies along a subspace and varies significantly
less on its orthogonal complement. To make precise what we mean by "on average",
we introduce a probability measure $\mu$ on $\mathcal{X}$. The choice of $\mu$ will be
dictated by the problem at hand. Given that $f$ may be
highly nonlinear, the question of identifying a subspace that on average captures
the majority of the variation is not a simple one. The active subspace method
addresses this challenge using gradient information, and thus we assume going
forward that $f$ is differentiable. We denote the gradient of $f$ at an input
$x \in \mathcal{X}$ by
$$
\nabla f(x) := \left[D_1 f(x), \dots, D_d f(x) \right]^\top \in \mathbb{R}^d.
$$
On a notational note, for a function $\phi: \mathbb{R}^n \to \mathbb{R}^m$ we
use $D\phi(x)$ to denote the $m \times n$ Jacobian matrix. Therefore, when applied
to the scalar-valued function $f$, we have the relation $\nablaf(x) = Df(x)^\top$.
By observing gradient evaluations $\nabla f(x_i)$ at a set of inputs
$x_1, \dots, x_n$ sampled from $\mu$, we can get a sense of how the function
varies along each coordinate direction, on average. However, it may be the case
that the function varies most significantly on a subspace not aligned with
the standard coordinate directions. Thus, the idea of active subspaces
is to utilize the information in these gradient observations to try to find
such a subspace; that is, to find *linear combinations* of the the coordinate
directions along which $f$ exhibits the largest variation. At this point,
the idea here should be sounding quite reminiscent of principal components
analysis (PCA). Indeed, there are close ties between the two methods. We therefore
briefly review PCA in order to better motivate the active subspace algorithm.
{% endkatexmm %}

# PCA Review
In order to better motivate the active subspace method, we take a brief detour
and briefly review the relevant ideas from PCA. I have a whole
[post](https://arob5.github.io/blog/2023/12/15/PCA/) on this
topic that derives all the facts stated below.

{% katexmm %}
## The Empirical Perspective
Consider a set of points $x_1, \dots, x_n$ in $\mathbb{R}^d$.
Let us assume that these points are centered, in the sense that the
empirical mean has been subtracted from each input.
We can stack the
transpose of these vectors row-wise to obtain a matrix $X \in \mathbb{R}^{n \times d}$.
The data has $d$ features, and the goal of PCA is to construct a new set of
$r < d$ linear combinations that explain the majority of variation in the data.
This is accomplished via an eigendecomposition of the matrix
$$
\hat{C} = X^\top X = \sum_{i=1}^{n} x_i x_i^T,
$$
which is proportional to the empirical covariance of the data.
Considering $\hat{C} = VDV^\top$, then the first $r$ columns of $V$
form a basis for the optimal $r$-dimensional subspace. The low-rank approximations
to the data points are thus given by the projection
$$
\hat{x}_i := \sum_{j=1}^{r} \langle x_i, v_j \rangle v_j
$$
with $v_1, \dots, v_r$ denoting the dominant $r$ eigenvectors of $\hat{C}$.
This approximation is optimal in the sense that it minimizes the average squared
error
$$
\frac{1}{n} \sum_{i=1}^{n} \left\lVert x_n - \sum_{j=1}^{r} w_{ij} b_j \right\rVert^2_2
$$
over all possible orthonormal bases $b_1, \dots, b_r$ of $\mathbb{R}^r$ and weights
$\{w_{ij}\}$, where $i = 1, \dots, n$ and $j = 1, \dots, r$. Notice that the
basis vectors $b_j$ are independent of the data, while the weights $w_{ij}$
are $x$-dependent.

## The Distribution Perspective
The basic PCA algorithm centers on the *empirical* covariance matrix $\hat{C}$.
Instead of working with empirical data, we can alternatively work with the
distribution of $x$ directly. Suppose $x = x(\omega)$ is a random vector defined
with respect to the probability space $(\Omega, \mathcal{A}, \mathbb{P})$.
Essentially all of the above results still
hold, with the covariance $C := \text{Cov}[x]$ replacing the empirical covariance
$\hat{C}$. In this setting, the objective function being minimized is the
expected squared error
$$
\mathbb{E} \left\lVert x(\omega) - \sum_{j=1}^{r} w(\omega) b_j \right\rVert^2_2
$$
over all possible orthonormal bases $b_1, \dots, b_r$ of $\mathbb{R}^r$ and
all random vectors $w \in \mathbb{R}^r$. Just as the $w_{ij}$ encoded information
about the variation in $x$ in the empirical setting, the coefficient
vector $w(\omega)$ is the source of randomness in the low-rank approximation that
seeks to capture the variation in $x(\omega)$.
{% endkatexmm %}

# Motivating Active Subspaces via PCA
## Doing PCA on the Gradient
{% katexmm %}
In seeking active subspaces, we might try to perform PCA directly on $X$ in
order to capture its dominant directions of variation. However, this approach
fails to take into account the function $f$ at all. The input $x$ may
vary significantly (with respect to the measure $\mu$) in a certain direction,
but this variation might induce essentially no change in the function output.
We seek a method that accounts for both the variation in $x$ induced by $\mu$,
as well as the action of the map $f(x)$. In fact, it is common to assume
that $\mu$ has been centered and normalized such that
\begin{align}
&\mathbb{E}[x] = 0, &&\text{Cov}[x] = I.
\end{align}
This means that the dimensions of $x$ are already uncorrelated; there is
nothing to be gained by performing PCA directly on $x$. However, there still
may be low-dimensional structure to exploit in the map $f$.

As noted above, the gradient $\nabla f(x)$ provides information on the local
variation in the function along the standard coordinate axes. Alluding to the
PCA setting, we can think of these directions of variation as $d$ "features",
with the goal of finding linear combinations of the features that explain
the majority of the variation. Let's go with this, taking $\nabla f(x)$ to
be the target for a PCA analysis. Since we're considering starting with
an assumed distribution $x \sim \mu$, it makes sense to apply the
distributional form of PCA, as summarized above. To this end, we start by
considering the matrix
$$
C := \mathbb{E}[(\nabla f(x)) (\nabla f(x))^\top],  \tag{1}
$$
which is an unscaled version of $\text{Cov}[\nabla f(x)]$. Note that the
expectations here are all with respect to $\mu$. The analogous quantities
in the above PCA review are
$$
x \iff \omega, \qquad \nabla f(x) \iff x(\omega).
$$
We can think of $\nabla f$ as a random variable mapping from the
probability space $(\mathcal{X}, \mathcal{B}(\mathcal{X}), \mu)$. Proceeding
as usual, we consider the eigendecomposition $C = VDV^\top$. After choosing
a threshold $r$, the first $r$ columns of $V$ become the basis for our
*active subspace*. The $r$-dimensional approximation of the gradient is then
given by
$$
\hat{\nabla} f(x) := \sum_{j=1}^{r} \langle \nabla f(x), v_j \rangle v_j.
$$

## Defining the Active Subspace
While I find the PCA comparison quite useful for motivating the method of active
subspaces, there is a risk of pushing the analogy too far. In particular, we
emphasize that the goal here is *not* to construct an accurate low-rank
approximation of the gradient, but rather to approximate $f(x)$ by a function
that looks like $g(Ax)$. To provide some initial justification for this latter
goal, we show that, on average, $f$ varies more along the directions defined
by the dominant eigenvectors. We let $D_v f(x) := \nabla f(x)^\top v$ denote
the directional derivative of $f$ in the direction $v$.

<blockquote>
  <p><strong>Proposition.</strong>
  Let $(\lambda_j, v_j)$ denote the $j^{\text{th}}$ eigenvalue-eigenvector pair
  of $C$, as defined above. Then
  \begin{align}
  \mathbb{E}\left[(D_{v_j} f(x))^2 \right] &= \lambda_j. \tag{2}
  \end{align}
  </p>
</blockquote>

**Proof.**
Using the fact that the $v_j$ have unit norm, we obtain  
\begin{align}
\lambda_j
&= \langle C v_j, v_j \rangle \newline
&= v_j^\top \mathbb{E}\left[\nabla f(x) \nabla f(x)^\top\right] v_j \newline
&= \mathbb{E}\left[ v_j^\top \nabla f(x) \nabla f(x)^\top v_j \right] \newline
&= \mathbb{E}\left[ (\nabla f(x)^\top v_j)^2 \right] \newline
&= \mathbb{E}\left[(D_{v_j} f(x))^2 \right] \qquad \blacksquare
\end{align}
In words, this result says that, on average (with respect to $\mu$),
the squared derivative in the direction of $v_j$ is given by the
eigenvalue $\lambda_j$. That is, the eigenvalues provide information about the
smoothness of the function in the directions defined by their respective
eigenvectors. Given this, it seems reasonable to discard directions with
eigenvalues of negligible size.

<blockquote>
  <p><strong>Definition.</strong>
  Consider partitioning the eigenvalues and eigenvectors as
  \begin{align}
  C = V \Lambda V^\top
  = \begin{bmatrix} V_1 & V_2 \end{bmatrix} \begin{bmatrix} \Lambda_1 & 0 \newline 0 & \Lambda_2 \end{bmatrix} \begin{bmatrix} V_1^\top \newline V_2^\top \end{bmatrix}, \tag{3}
  \end{align}
  where $V_1 \in \mathbb{R}^{d \times r}$ and $\Lambda_1 \in \mathbb{R}^{r \times r}$.
  We define the <strong>active subspace</strong> (of dimension $r$)
  to be the span of the columns of $V_1$,
  $$
  \text{span}(v_1, \dots, v_r). \tag{4}
  $$
  Similarly, we refer to the subspace generated by the columns of $V_2$ as
  the <strong>inactive subspace</strong>.
  </p>
</blockquote>

This is the same idea as in PCA: define a cutoff after the first $r$ directions,
after which the variation in the thing you care about becomes negligible. The
subspace generated by the first $r$ directions is the active subspace, and the
columns of $V_1$ form an orthonormal basis for this subspace.
The inactive subspace is the orthogonal complement of the active subspace,
since $V_1^\top V_2 = I$.

# Understanding the Low-Dimensional Subspace
Having defined the active subspace, we now begin to investigate its
basic properties.  

## Active and Inactive Variables
Since the active subspace is $r$-dimensional, we can represent vectors
living in this subspace with $r < d$ coordinates. It is this fact that allows
for useful dimensionality reduction in downstream applications.
Given an input $x \in \mathbb{R}^d$ we want to identify the vector in
the active subspace that provides the best approximation to $x$. The optimal
vector (in a squared error sense) is given by the projection
$$
\text{proj}_{V_1}(x)
:= \sum_{j=1}^{r} \langle x, v_j \rangle v_j \tag{5}
= V_1 V_1^\top x.
$$
Thus, $\langle x, v_1 \rangle, \dots, \langle x, v_r \rangle$ constitute the
$r$ coordinates defining the projection of $x$ onto the active subspace. The
quantity $V_1^\top x \in \mathbb{R}^r$ stacks these coordinates into a vector.
Note that $V_1 V_1^\top x$ describes the same vector (i.e., the projection), but
represents the vector in the original $d$-dimensional coordinate system.
We can similarly project onto the *inactive* subspace via $V_2 V_2^\top x$.

<blockquote>
  <p><strong>Definition.</strong> \tag{6}
  Let $V_1$ and $V_2$ be defined as in (3). We introduce the notation
  $y := V_1^\top x \in \mathbb{R}^r$ and $z := V_2^\top x \in \mathbb{R}^{d-r}$ and
  refer to $y$ and $z$ respectively as the <strong>active variable</strong>
  and <strong>inactive variable</strong>.
  </p>
</blockquote>

Note that $y$ and $z$ are simply the coordinates of the projection of $x$ onto
the active and inactive subspace, respectively. Since $x \sim \mu$
is a random variable, then so are $y$ and $z$. Noting that the active and inactive
subspaces are orthogonal complements, we have that  

$$
x
= \text{proj}_{V_1}(x) + \text{proj}_{V_2}(x)
= V_1 V_1^\top x + V_2 V_2^\top x
= V_1 y + V_2 z. \tag{7}
$$   

Recall that $f$ is a function of $d$ variables that we wish to approximate
with a function of only $r$ variables. Since
$$
f(x) = f(V_1 V_1^\top x + V_2 V_2^\top x) = f(V_1 y + V_2 z),
$$
the key will be to eliminate the dependence on $z$, a question we will return
to shortly. For now, we want to start providing some justification for this
idea by demonstrating that $f$ varies more on average as we vary $y$ as opposed
to when we vary $z$.

<blockquote>
  <p><strong>Proposition.</strong>
  Let $y$ and $z$ be the active and inactive variables defined in (6). Let
  \begin{align}
  &T_{z}(y) := V_1 y + V_2 z, &&T_{y}(z) := V_1 y + V_2 z
  \end{align}
  denote the coordinate transformation from $(y,z)$ to $x$, viewed respectively
  as functions of $y$ or $z$ only. Then,  
  \begin{align}
  \mathbb{E} \lVert D (f \circ T_{z})(y) \rVert_2^2 &= \lambda_1 + \cdots + \lambda_r \newline
  \mathbb{E} \lVert D (f \circ S_{y})(z) \rVert_2^2 &= \lambda_{r+1} + \cdots + \lambda_d.
  \end{align}
  </p>
</blockquote>

Note that for $T_{z}(y)$, the inactive variable is viewed as fixed with respect
to the derivative operation. However, $z$ is still random, and thus the expectation
$\mathbb{E} \lVert D (f \circ T_{z})(y) \rVert_2^2$ averages over the randomness in
both $y$ and $z$.

**Proof.**
We only prove the result for $T_z(y)$, as the proof for $S_y(z)$ is nearly
identical. By the chain rule we have
\begin{align}
D (f \circ T_{z})(y)
= Df(x) DT_{z}(y)
= Df(x)V_1.
\end{align}
For succinctness, we will use the notation
$$
\nabla_{y} f := [D (f \circ T_{z})(y)]^\top = V_1^\top \nabla f(x)
$$
in the below derivations. Thus,
\begin{align}
\mathbb{E} \lVert \nabla_{y} f  \rVert_2^2
&= \mathbb{E}\left[\text{tr}\left([\nabla_y f] [\nabla_y f]^\top \right) \right] \newline
&= \text{tr}\left(\mathbb{E}\left[[\nabla_y f] [\nabla_y f]^\top \right] \right) \newline
&= \text{tr}\left(V_1^\top \mathbb{E}\left[\nabla f(x) \nabla f(x)^\top \right] V_1 \right) \newline
&= \text{tr}\left(V_1^\top C V_1 \right) \newline
&= \text{tr}\left(V_1^\top V \Lambda V^\top V_1 \right) \newline
&= \text{tr}(\Lambda_1) \newline
&= \lambda_1 + \dots + \lambda_r.
\end{align}

## The Distribution of the (In)Active Variables
As mentioned in the previous section, the active and inactive variables
are functions of the random variable $x \sim \mu$ and hence are themselves
random variables. In this section we investigate the joint distribution
of the random vector $u := (y, z)^\top$. We recall from (6) that this
vector is defined as
\begin{align}
u := \begin{bmatrix} y \newline z \end{bmatrix}
= \begin{bmatrix} V_1^\top x \newline V_2^\top x \end{bmatrix}
= V^\top x,
\end{align}
where $V$ is orthonormal. Thus, the transformation $u = T(x) := V^\top x$ can be
thought of as rotating the original coordinate system. In particular, note
that $V$ is invertible with $V^{-1} = V^\top$. Let's suppose that the measure
$\mu$ admits a density function $\rho$. Since the transformation $T$
is invertible and differentiable, then the change-of-variables
formula tells us the density of the random vector $u$. Denoting this density
by $\tilde{\rho}$, we have
$$
\tilde{\rho}(u^\prime)
= \rho(T^{-1}(u^\prime)) \lvert \text{det}(DT^{-1}(u^\prime)) \rvert
= \rho(Vu^\prime)
= \rho(x^\prime),
$$
following from the fact that $\text{det}(DT^{-1}(u^\prime)) = \text{det}(V) = 1$.
In words, to compute the density of $u$ at a point $u^\prime$, we can
simply evaluate the density of $x$ at the point $Vu^\prime$. There is
no distortion of space here since the transformation is a rotation.
We can also find the measure $\tilde{\mu}$ associated with the density
$\tilde{\rho}$ by considering, for some Borel set $A \subset \mathcal{X}$,  
\begin{align}
\tilde{\mu}(A)
&= \int_{\mathbb{R}^d} 1_A[u^\prime] \tilde{\mu}(du^\prime) \newline
&= \int\_{\mathbb{R}^d} 1_A[u^\prime] (\mu \circ T^{-1})(du^\prime) \newline
&= \int\_{\mathbb{R}^d} 1_A[T(x^\prime)] \mu(dx^\prime) \newline
&= \int\_{\mathbb{R}^d} 1[V^\top x^\prime \in A] \mu(dx^\prime) \newline
&=\int\_{\mathbb{R}^d} 1[x^\prime \in VA] \mu(dx^\prime) \newline
&= \mu(VA),
\end{align}
where I'm defining the set
$$
VA := \{V u^\prime: u^\prime \in A \}.
$$
This result is analogous to the density one, and simply says that the distributions
of $x$ and $u$ differ only by a change-of-variables. It is worth emphasizing that
$\tilde{\rho}(u^\prime) = \tilde{\rho}(y^\prime, z^\prime)$ is a *joint* density
over the active and inactive variables. We will shortly be interested in the
marginals and conditionals of this joint distribution.

### The Gaussian Case
As usual, things work out very nicely if we work with Gaussians. Let's consider the
case where $\mu$ is multivariate Gaussian.


{% endkatexmm %}

# References
- Active subspace methods in theory and practice: Applications to kriging surfaces
SIAM J. Sci. Comput., 36 (4) (2014), pp. A1500-A1524, 10.1137/130916138
