---
title: Emulating Computer Models with many Outputs: The Basis Function Approach
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
sort of expensive computer simulation model. We will therefore call interchangeably
refer to $\mathcal{G}$ as a *computer model* or *simulator*. The primary goal of interest
in this post is to construct an *emulator* (i.e., *surrogate model*) that
approximates $\mathcal{G}$. The implicit assumption here is that computing $\mathcal{G}(u)$ is
quite expensive, so the idea is to replace $\mathcal{G}$ with a computationally cheaper
approximation. This can be formulated generically as a regression problem; suppose
we run the computer model at a carefully chosen set of *design points*
$u_1, \dots, u_n \in \mathcal{U}$. This results in the *design*
$\{u_i, \mathcal{G}(u_i)\}_{i=1}^{n}$, a dataset that can be used to train a regression
model. While any regression model could be applied here, Gaussian processes (GP)
are a particularly popular choice. When the output dimension $p=1$, we can
fit a run-of-the-mill GP to the design. When $\mathcal{G}$ is multi-output but the outputs
are small in number, we might consider fitting a multi-output GP instead, the
simplest choice being the use of $p$ independent GPs. With parallel computing
resources, this independent emulation scheme may even feasible for larger values
of $p$. However, larger numbers of outputs are typically come with more
structure; e.g., the outputs may consist of a time series or have some spatial
structure. The independent GP approach completely failures to capture this
structure. The method discussed in this post seeks to take advantage of such
structure by finding a small set of basis vectors that explain the majority of
the variation in the outputs.
{% endkatexmm %}

{% katexmm %}
# Basis Representation
The now classic 2008 Higdon et al paper proposes a solution that assumes a model
of the form
\begin{align}
\mathcal{G}(u) = \sum_{j=1}^{r} w_j(u) b_j + \epsilon(u), \tag{2}
\end{align}
where ideally $r \ll p$. This decomposes the computer model output $\mathcal{G}(u)$ into
(1) a piece that can be explained by a linear combination of $r$ basis functions;
and (2) the residual $\epsilon(u)$, representing everything leftover. The basis
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
