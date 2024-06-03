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
with such a simulation: (1) find input values $x \in \mathbb{R}^d$ to make
$f(x)$ agree with some observed data (calibration); (2) assess how the
the function outputs $f(x)$ vary as the inputs $x$ change (sensitivity analysis);
(3) identify subsets of $\mathcal{X}$ that result desirable or undesirable
outputs (contour/excursion set estimation); (4) fit a cheaper-to-evaluate
surrogate model that approximates $f$ (emulation/surrogate modeling).
All of these tasks are made more difficult when the number of inputs $d$ is large.
However, it has been routinely noted that such high-dimensional settings often
exhibit low-dimensional structure, or low "intrinsic dimensionality". This is to
say that $f$ varies primarily on some low-dimensional manifold embedded in
the higher-dimensional ambient space $\mathcal{X}$. Active subspaces assume
that the manifold is a linear subspace, and seeks to find this subspace.
{% endkatexmm %}

# Ridge Functions
{% katexmm %}
Before introducing the method of active subspaces, we start to build some
intuition regarding the notion of low-dimensional subspace structure in the
map $f$. The extreme case of this idea is a function of the form
\begin{align}
f(x) &= g(Ax), &&A \in \mathbb{R}^{r \times d},
\end{align}
where $r < d$. A function of this form is sometimes called a **ridge function**.
The ridge assumption is a rather stringent one, as it implies that the function
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
we introduce a measure $\mu$ on $\mathcal{X}$. The choice of $\mu$ will be
dictated by the problem at hand. In the simplest case, we might consider
setting $\mu$ to the Lebesgue measure. Given that $f$ may be
highly nonlinear, the question of identifying a subspace that on average captures
the majority of the variation is not a simple one. The active subspace method
addresses this challenge using gradient information, and thus we assume going
forward that $f$ is differentiable. We denote the gradient of $f$ at an input
$x \in \mathcal{X}$ by
$$
\nabla f(x) := \left[D_1 f(x), \dots, D_d f(x) \right]^\top \in \mathbb{R}^d.
$$
By observing gradient evaluations $\nabla f(x_i)$ at a set of inputs
$x_1, \dots, x_n$ sampled from $\mu$, we can get a sense of how the function
varies along each coordinate direction, on average. However, it may be the case
that the function varies most significantly on a subspace not aligned with
the standard coordinate directions. Thus, the idea of active subspaces
is to utilize the information in these gradient observations to try to find
such a subspace; that is, to find *linear combinations* of the the coordinate
directions along which $f$ exhibits the largest variation. At this point,
the idea here should be sounding quite reminiscent of principal component
analysis (PCA). Indeed, there are close ties between the two methods. We therefore
briefly review PCA in order to better motivate the active subspace algorithm.
{% endkatexmm %}

# PCA Review
{% katexmm %}
## The Empirical Perspective
Consider a set of points $x_1, \dots, x_n$ in $\mathcal{R}^d$.
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
{% katexmm %}
In seeking active subspaces, we might try to perform PCA directly on $X$ in
order to capture its dominant directions of variation. However, this approach
fails to take into account the function $f$ at all. The input $x$ may
vary significantly (with respect to the measure $\mu$) in a certain direction,
but this variation might induce essentially no change in the function output.
We seek a method that accounts for both the variation in $x$ induced by $\mu$,
as well as the action of the map $f(x)$.

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
C_f := \mathbb{E}[(\nabla f(x)) (\nabla f(x))^\top],  \tag{1}
$$
which is an unscaled version of $\text{Cov}[\nabla f(x)]$. Note that the
expectations here are all with respect to $\mu$. The analogous quantities
in the above PCA review are
$$
x \iff \omega, \qquad \nabla f(x) \iff x(\omega).
$$
We can think of $\nabla f$ as a random variable mapping from the
probability space $(\mathcal{X}, \mathcal{B}(\mathcal{X}), \mu)$. Proceeding
as usual, we consider the eigendecomposition $C_f = VDV^\top$. After choosing
a threshold $r$, the first $r$ columns of $V$ become the basis for our
*active subspace*. The $r$-dimensional approximation of the gradient is then
given by
$$
\hat{\nabla} f(x) := \sum_{j=1}^{r} \langle \nabla f(x), v_j \rangle v_j,
$$
with the guarantee that this minimizes the expected error
$$
\mathbb{E} \left\lVert \nabla f(x) - \sum_{j=1}^{r} w_j(x) b_j \right\rVert_2^2
$$
over all possible orthonormal bases $b_1, \dots, b_r$ of $\mathbb{R}^r$ and
all random vectors $w: \mathcal{X} \to \mathbb{R}^r$. We have been implicitly
assuming that $\mathbb{E}[\nabla f(x)] = 0$, in typical PCA fashion.

While I find the PCA analogy quite useful for motivating the method of active
subspaces, there is a risk of pushing the analogy too far. In particular, we
emphasize that the goal here is *not* to construct an accurate low-rank
approximation of the gradient, but rather to approximate $f(x)$ by a function
that looks like $g(Ax)$. To provide some initial justification for this latter
goal, we show that, on average, $f$ varies more along the directions defined
by the dominant eigenvectors.

<blockquote>
  <p><strong>Proposition.</strong>
  A probability measure $\mu$ defined on the Borel measurable space
  $(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n))$ is called <strong>Gaussian</strong>
  if, for all linear maps $\ell \in (\mathbb{R}^n)^*$, the pushforward measure
  $\mu \circ \ell^{-1}$ is Gaussian on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.
  </p>
</blockquote>



{% endkatexmm %}
