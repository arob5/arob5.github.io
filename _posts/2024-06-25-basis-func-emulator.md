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
where the output dimension $p$ is presumed to be "large", potentially in the
hundreds or thousands. We treat $\mathcal{G}$ generically as a black-box function,
but in common applications of interest it takes the form of an expensive computer
simulation model. We will therefore use the terms *black-box function*,
*computer model*, and *simulator* interchangeably throughout this post.
Our primary goal of interest is to construct a model that approximates the
map $u \mapsto G(u)$. We will refer to such a model as an *emulator* or
*surrogate model*. The implicit assumption here is that computing $\mathcal{G}(u)$ is
quite expensive, so the idea is to replace $\mathcal{G}$ with a computationally cheaper
approximation. If you don't care about emulating expensive computer models, you
can also view this generically as a regression (or interpolation) problem with
a very high-dimensional output space. To further elucidate the connection,
suppose that we evaluate the function at a set of points
$u_1, \dots, u_n \in \mathcal{U}$, resulting in the input-output pairs
$\{u_i, \mathcal{G}(u_i)\}_{i=1}^{n}$. We can now treat these pairs as training
examples that can be use to fit a predictive model. From this point of view, the
primary distinction from more traditional regression is that in this case we get
to choose the input points $u_i$ at which we evaluate the function.

The methods we discuss below attempt to address two primary challenges:
1. Function evaluations $G(u)$ are expensive, and thus we seek to minimize the
number of points at which $G$ is evaluated.
2. The output space of $G$ is very high-dimensional.

The second challenge can be problematic for standard regression methods, which
are often tailored to scalar-valued outputs. The obvious solution here might
be to fit a separate model to predict each individual output; i.e., a model
to emulate the map $u \mapsto G_j(u)$ for each $j=1, \dots, p$.  
With parallel computing resources, such an approach might even be feasible for
very large values of $p$. However, larger numbers of outputs typically come with more
structure; e.g., the outputs may consist of a time series or spatial fields.
The independent output-by-output regression approach completely fails to
leverage this structure. The method discussed in this post seeks to take
advantage of such structure by finding a set of basis vectors that explain the
majority of the variation in the outputs. We proceed by first generically
discussing basis representations of the output space, and how such structure
can be leveraged to address the two challenges noted above. With the general
method defined, we conclude by discussing details for a specific choice of
basis approximation (principal components analysis) and a specific choice
of emulator model (Gaussian process regression).    
{% endkatexmm %}

{% katexmm %}
# Basis Representation
From a very generic perspective, the methods we discuss here can be thought of
as a method for dimension reduction of the *output* space of a regression
problem. It is perhaps more typical to see dimensionality reduction applied
to the *input* space in such settings, with principal component regression being
the obvious example. In this section, we start by discussing a low-dimensional
basis representation of the output space, and then explore how such a representation
can be leveraged in solving the regression problem.  

## A Basis Representation of the Output Space
Let's start by considering approximately representing vectors in the range of $G$,
denoted $\mathcal{R}(G)$, with respect to a set of $r \ll p$ orthonormal basis vectors
$\{b_1, \dots, b_r\} \subset \mathbb{R}^p$. Given such a set of vectors, we can
approximate $g \in \mathcal{R}(G)$ by its projection onto the subspace
$\text{span}(b_1, \dots, b_r)$:
$$
\hat{g} := \sum_{j=1}^{r} \langle g, b_r\rangle b_r. \tag{2}
$$
If we stack the basis vectors as columns in a matrix $B \in \mathbb{R}^{p \times r}$
then we can write this projection compactly as
$$
\hat{g}
= \sum_{j=1}^{r} \langle g, b_r\rangle b_r
= = \sum_{j=1}^{r} (b_r b_r^\top)g
= BB^\top g, \tag{3}
$$
We see that $BB^\top$ is the projection matrix that projects onto the span of the
basis vectors. With regards to dimensionality reduction, the benefit here is that
the simulator output can now be (approximately) represented using $r \ll p$ numbers
$B^\top g$. We can now ask the question: how do we find the basis vectors $B$?
If we are given a set of vectors $g_1, \dots, g_n \in \mathcal{R}(G)$, we can take
an empirical approach and try to use these examples to determine a $B$ that is
optimal in some well-defined sense.
Assuming that $\mathcal{R}(G) \subseteq \mathbb{R}^p$ is indeed a subspace and
we define "optimal" in an average squared error sense, the
problem we have laid out here is exactly that of principal components analysis
(PCA), a topic I discuss in depth in [this](https://arob5.github.io/blog/2023/12/15/PCA/)
post. The only difference is that we are applying PCA on the subspace
$\mathcal{R}(G)$. At this point, we should emphasize that in practice
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
u \mapsto \sum_{j=1}^{r} \langle G(u), b_r\rangle b_r = B [B^\top G(u)]. \tag{4}
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
\hat{G}_r(u) := \sum_{j=1}^{r} w_r(u)b_r = Bw(u) \in \mathbb{R}^p. \tag{5}
$$
At this point, this
isn't helpful since the expensive simulation still needs to be run every time
$\hat{G}_r(u)$ is evaluated. To address this, we now turn back to the idea of
using emulators. Recall that such an approach was originally hindered due to the
high-dimensional output space of the simulator. However, under the approximation
$\hat{G}_r(u)$, the dependence on $u$ has been reduced to $w(u)$, which
effectively reduces the output dimension to $r$. The idea is thus to use some sort
of statistical model
to emulate the map $u \mapsto w(u)$. Suppressing all details for now, let's suppose
we have fit such a model $w^*(u)$. We can now plug $w^*(u)$ in place of
$w(u)$ in (5) to obtain the approximation
$$
\hat{G}^*_r(u) := \sum_{j=1}^{r} w^*_r(u)b_r = Bw^*(u) \in \mathbb{R}^p. \tag{6}
$$
This approximation no longer requires running the full simulator, since evaluating
$\hat{G}^*_r(u)$ just requires (1) computing the emulator prediction at $u$; and
(2) applying $B$ to the emulator prediction. It is worth emphasizing how this
approach compares to the direct emulation method. In place of directly trying
to approximate the map from $u$ to the model outputs $G(u)$, we are now considering
approximating the map from $u$ to inner products of $G(u)$ with a
small number of basis vectors. The hope is that these inner products are sufficient
to capture the majority of information in the model response.

## The General Emulation Model
In the preceding subsections, we introduced the idea of representing the output space
(range) of $\mathcal{G}$ with respect to a low-dimensional basis in order to
facilitate emulation of the map $u \mapsto \mathcal{G}(u)$. For concreteness, we
considered an orthogonal basis, whereby approximations of $\mathcal{G}(u)$ take
the form of orthogonal projections onto the basis vectors. In this section,
we take a step back and define the general model, which encompasses basis methods
beyond the orthogonal projection framework.
<blockquote>
  <p><strong>The Basis Function Emulation Model.</strong>
  Given a set of vectors $\{b_1, \dots, b_r\} \subset \mathbb{R}^{p}$,
  we refer to a decomposition of the form
  $$
  G(u) = \sum_{j=1}^{r} w_j(u) b_j + \epsilon(u) \tag{7}
  $$
  as the *basis function GP emulation model*. We write $w^*_j(u)$ to denote
  an emulator model that approximates the map $w_j(u)$.
  </p>
</blockquote>

The basis function emulation model decomposes the computer model
output $\mathcal{G}(u)$ into
1. a piece that can be explained by a linear
combination of $r$ basis functions that are independent of $u$; and
2. the residual $\epsilon(u)$, representing all variation unaccounted for by
the basis functions.

We emphasize that the basis functions are independent of the input $u$; the
effect of the inputs is restricted to the coefficients $w_j(u)$, with unaccounted
for $u$-dependence absorbed by the residual term $\epsilon(u)$. As noted in the
previous section, if we opt for a PCA decomposition of the output space, then
the true weights assume the form
$$
w_j(u) = \langle G(u), b_j \rangle, \tag{8}
$$
but the general model (7) allows for other approaches as well.
Under different decomposition strategies, the true weights may
not be given by the inner products (8). Nonetheless, we can still consider
applying statistical models to approximate the underlying weight maps
$u \mapsto w_j(u)$, regardless of what form these maps may take.
{% endkatexmm %}

# Concrete Details
{% katexmm %}
## Constructing the PCA Basis
The natural question is how to actually identify and compute the $b_j$. In the
absence of prior knowledge, the only recourse is to run the simulation $G(u)$
at a variety of different input points $u$ in order to understand the
patterns of variability in model outputs. We refer to the resulting input-output  
pairs
$$
\{u_i, G(u_i)\}, \qquad i = 1, \dots, n
$$
as the *design*. It practice, the availability of parallel computing resources
typically means that model simulations can be run in parallel, thus
reducing the computational cost of generating the design. The input design points
$u_i$ are often chosen to vary "uniformly" over the input space $\mathcal{U}$
in some sense. This means that the set of corresponding outputs $G(u_i)$ can be
thought of as representative of variation in the simulator output under typical
use cases.  

The
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

## Gaussian Process Emulators


{% endkatexmm %}






# References
- Computer Model Calibration using High Dimensional Output (Higdon et al., 2008)
- JMS&Williamson D. B. (2020). ”Efficient calibration for high-dimensional computer model output using basis methods”. arXiv preprint arXiv:1906.05758
- JMS, Dodwell T.J., et al. (2021) ”A History Matching Approach to Building Full-Field Emulators in Composite Analysis”.
- JMS, Williamson D. B., Scinocca J., and Kharin V. (2019). ”Uncertainty quantification for computer models with spatial output using calibration-optimal bases”. Journal of the American Statistical Association, 114.528, 1800-1814.
- Chang, Won, et al. ”Probabilistic calibration of a Greenland Ice Sheet model using spatially resolved synthetic observations: toward projections of ice mass loss with uncertainties.” Geoscientific Model Development 7.5 (2014): 1933-1943.
