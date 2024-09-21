---
title: Emulating Computer Models with many Outputs - The Basis Function Approach
subtitle: A discussion of the popular output dimensionality strategy for emulating multi-output functions.
layout: default
date: 2024-06-25
keywords: Gaussian-Process, UQ
published: true
---

{% katexmm %}
# Setup
Suppose we are working with a function
\begin{align}
\mathcal{G}: \mathcal{U} \subset \mathbb{R}^d \to \mathbb{R}^p, \tag{1}
\end{align}
where the output dimension $p$ is presumed to be "large"; at the very least
more than a few, but potentially in the hundreds or thousands. On the other
hand, we suppose the input dimension $d$ is of moderate size; say, no more than
10-20. We treat $\mathcal{G}$ generically as a black-box function, but in
practice it is typically some
sort of expensive computer simulation model. We will therefore
refer to $\mathcal{G}$ interchangeably as a *computer model* or *simulator*.
The primary goal of interest
in this post is to construct an *emulator* (i.e., *surrogate model*) that
approximates $\mathcal{G}$. The implicit assumption here is that computing $\mathcal{G}(u)$ is
quite expensive, so the idea is to replace $\mathcal{G}$ with a computationally cheaper
approximation. This can be formulated generically as a surface-fitting problem
(interpolation or regression); suppose
we run the computer model at a carefully chosen set of *design points*
$u_1, \dots, u_n \in \mathcal{U}$. This results in the *design*
$\{u_i, \mathcal{G}(u_i)\}_{i=1}^{n}$, a dataset that can be used to train a regression
model. While any regression model could be applied here, Gaussian processes (GP)
are a particularly popular choice. With a scalar output dimension $p=1$, we can
fit a run-of-the-mill GP to the design. When $\mathcal{G}$ is multi-output but the outputs
are small in number, we might consider fitting a multi-output GP instead, the
simplest choice being the use of $p$ independent GPs. With parallel computing
resources, this independent emulation scheme may even prove feasible for larger values
of $p$. However, larger numbers of outputs are typically come with more
structure; e.g., the outputs may consist of a time series or have some spatial
structure. The independent GP approach completely failures to capture this
structure. The method discussed in this post seeks to take advantage of such
structure by finding a small set of basis vectors that explain the majority of
the variation in the outputs.
{% endkatexmm %}
{% katexmm %}

# Basis Representation

## A Basis Representation of the Output Space
Let's start by considering approximately representing vectors in the range of $G$,
denoted $\mathcal{R}(G)$, with respect to a set of $r \ll p$ orthonormal basis vectors
$\{b_1, \dots, b_r\} \subset \mathbb{R}^p$. Given such a set of vectors, we can
approximate $g \in \mathcal{R}(G)$ by its approximation onto the subspace
$\text{span}(b_1, \dots, b_r)$:
$$
\hat{g} := \sum_{j=1}^{r} \langle g, b_r\rangle b_r.
$$
If we stack the basis vectors as columns in a matrix $B \in \mathbb{R}^{p \times r}$
then we can write this projection compactly as
$$
\hat{g}
= \sum_{j=1}^{r} \langle g, b_r\rangle b_r
= \sum_{j=1}^{r} B^\top g b_r
= B^\top B g,
$$
We see that $B^\top B$ is the projection matrix that projects onto the span of the
basis vectors. With regards to dimensionality reduction, the benefit here is that
the simulator output can now be (approximately) represented using $r \ll p$ numbers
$B^\top g$. We can now ask the question: how do we find the basis vectors $B$?
If we are given a set of vectors $g_1, \dots, g_n \in \mathcal{R}(G)$, we can take
an empirical approach and try to use these examples to determine a $B$ that is
optimal in some well-defined sense.
Assuming that $\mathcal{R}(G) \subseteq \mathbb{R}^p$ is indeed a subspace, the
problem we have laid out here is exactly that of principal components analysis
(PCA), a topic I discuss in depth in [this](https://arob5.github.io/blog/2023/12/15/PCA/)
post. The only difference is that we are applying PCA on the subspace
$\mathbb{R}(G)$. At this point, we should emphasize that in practice
$\mathcal{R}(G)$ will often not be a subspace. Computer simulations may produce
outputs that are subject to certain constraints, and thus $\mathcal{R}(G)$ may
represent a more complicated subset of $\mathbb{R}^p$. In these cases, one
can still typically apply the PCA algorithm to obtain $B$, but the result may
be sub-optimal. Alternate methods of basis construction may be warranted depending
on the problem at hand.

## Linking the Basis Representation with the Input Parameters
In the previous subsection, we considered approximating vectors in the range of
the simulator with respect to a set of basis vectors $B$. However, recall that our
underlying goal here is to approximate the map $u \mapsto G(u)$. We thus need to
consider how to leverage the basis representation of the output space in achieving
this goal. Assuming we have already constructed the basis vectors $B$, the above
map can be approximated as
$$
u \mapsto \sum_{j=1}^{r} \langle G(u), b_r\rangle b_r = B [B^\top G(u)].
$$
In words: feed $u$ through the simulator and project the resulting output onto
the low-dimensional subspace spanned by the basis vectors. Note that
$B^\top G(u)$ stores the $r$ weights defining the projection of $G(u)$ onto the
subspace generated by $B$, thus providing a low dimensional summary of the
simulator output. Let's introduce the notation
\begin{align}
w(u) &:= B^\top G(u) = \left[w_1(u), \dots, w_r(u) \right]^\top \in \mathbb{R}^r,
&& w_r(u) := \langle G(u), b_r \rangle
\end{align}
to denote this weights. The basis function approximation to the simulator can thus
be written as
$$
\hat{G}_r(u) := \sum_{j=1}^{r} w_r(u)b_r = Bw(u) \in \mathbb{R}^p. \tag{1}
$$
At this point, this
isn't helpful since the expensive simulation still needs to be run every time
$\hat{G}_r(u)$ is evaluated. To address this, we now turn back to the idea of
using GP emulators. Recall that such an approach was originally hindered due to the
high-dimensional output space of the simulator. However, under the approximation
$\hat{G}_r(u)$, the dependence on $u$ has been reduced to $w(u)$, which
effectively reduces the output dimension to $r$. The idea is thus to use GPs
to emulate the map $u \mapsto w(u)$. Suppressing all details for now, let's suppose
we have fit such a GP approximation $w^*(u)$. We can now plug $w^*(u)$ in place of
$w(u)$ in (1) to obtain the approximation
$$
\hat{G}^*_r(u) := \sum_{j=1}^{r} w^*_r(u)b_r = Bw^*(u) \in \mathbb{R}^p. \tag{2}
$$
This approximation no longer requires running the full simulator, since evaluating
$\hat{G}^*_r(u)$ just requires (1) computing the GP prediction at $u$; and
(2) applying $B$ to the GP prediction.

# The General Emulation Model  
In this section, we take a step back and define the general emulation model utilizing
both GPs and a basis representation of the model outputs. We then discuss the general
procedure for learning the basis vectors $B$ from training data, and provide some
specific details on fitting the GP emulators. Given a set of vectors
$\{b_1, \dots, b_r\} \subset \mathbb{R}^{p}$, we refer to
\begin{align}
\mathcal{G}(u) &= \sum_{j=1}^{r} w^*_j(u) b_j + \epsilon(u), \tag{3} \newline
w^*_j &\sim \mathcal{GP}(\mu_j, k_j)
\end{align}
as the *basis function GP emulation model*. This decomposes the computer model
output $\mathcal{G}(u)$ into (1) a piece that can be explained by a linear
combination of $r$ basis functions;
and (2) the residual $\epsilon(u)$, representing everything leftover. The
basis functions are independent of the input $u$; the effect of the inputs
is restricted to the coefficients $w_j(u)$, with unaccounted for $u$-dependence
absorbed by the residual term $\epsilon(u)$. In the previous section, we considered
the common setting where the basis decomposition was given by a standard application
of PCA. In this case, the true weights are given by the projection coefficients  
$$
w_j(u) = \langle G(u), b_j \rangle. \tag{4}
$$
The GP emulator $w^*_j(u)$ thus seeks to approximate the
inner product between $G(u)$ and $b_j$. It is worth noting that under this approch,
we have therefore substituted the typical GP emulation strategy of directly
approximating $G(u)$ with that of approximating inner products of $G(u)$ with a
small number of basis vectors.

It is important to emphasize that the model (3) extends beyond this PCA/orthogonal
projection setting. Under different decomposition strategies, the true weights may
not be given by the inner products (4). Nonetheless, we can still consider
GP models to approximate the underlying weight maps $u \mapsto w_j(u)$, regardless of
what form these maps may take.  


The now classic 2008 Higdon et al paper proposes a solution that assumes a model
of the form
\begin{align}
\mathcal{G}(u) = \sum_{j=1}^{r} w_j(u) b_j + \epsilon(u), \tag{2}
\end{align}
where ideally $r \ll p$.  The basis
functions $b_1, \dots, b_r \in \mathbb{R}^d$ are fixed in that they are not a
function of the simulator inputs $u$. The dependence on $u$ is captured entirely
by the basis function weights $w_j(u)$ and of course by the residual $\epsilon(u)$.
The emulation problem has thus been decomposed into $r$ easier univariate
emulation problems; that is, the task is now to emulate the maps
\begin{align}
&w_j: \mathcal{U} \to \mathbb{R}, &&j = 1, \dots, r.
\end{align}
The hope is that $r$ can be chosen sufficiently small enough so that we don't
have to fit too many univariate GPs, while also ensuring that the $r$ basis
functions explain the majority of the variation in the output; i.e., that
$\epsilon(u)$ is not too large. The following sections describe these concepts
in greater detail.
{% endkatexmm %}

{% katexmm %}
# Finding the Basis Functions
The natural question is how to actually identify and compute the $b_j$. The
general idea is to learn the basis vectors from the design
$\{u_i, \mathcal{G}(u_i)\}_{i=1}^{n}$.
Let us suppose that we have collected the outputs these $n$ simulations into a
$n \times p$ matrix $G$. We can now think of the problem generically as one
of linear dimensionality reduction. The rows of $G$ provide a summary of the
variation in the simulator outputs due to variation in the inputs (typically,
the design inputs $u_i$ are chosen to vary in some sense "uniformly" over the
input space $\mathcal{U}$). Each column of $G$ represents a single dimension
in the output space. Our goal is to find a small number of
linear combinations of these dimensions that explain the majority of variation.

There are many different bases one might consider using here. In this section,
we will focus on the most popular: the basis of principal components,
or equivalently the basis constructed via an application of the singular
value decomposition (SVD). I have a whole in-depth
[post](https://arob5.github.io/blog/2023/12/15/PCA/) on principal components
analysis (PCA), so I will assume general background knowledge on this topic.
In typical PCA fashion, we start by centering $G$ so that

{% endkatexmm %}


# References
- Computer Model Calibration using High Dimensional Output (Higdon et al., 2008)
