---
title: Transforming Simplex-values Parameters
subtitle: Walking through Stan's parameter transformation for parameters that sum to one.
layout: default
date: 2025-03-07
keywords:
published: true
---

{% katexmm %}
In this post we consider a $d$-dimensional random vector $x$ that is
constrained to lie in the unit simplex,
$$
\Delta_d := \left\{x \in \mathbb{R}^d : x_j \geq 0, \sum_{j=1}^{d} x_j = 1 \right\}. \tag{1}
$$
In words, the $d$ values $x_1, \dots, x_d$ must be nonnegative and sum to one.
Performing statistical inference with respect to a parameter
$x \in \Delta_d$ can be tricky; a common solution is to consider an invertible
transformation
$$
y = \phi(x) \tag{2}
$$
such that $y$ is unconstrained. Inference can then be performed with respect
to $y$ and we can transform back using $\phi^{-1}$ afterwards. This is
the approach used by [Stan](https://mc-stan.org/docs/reference-manual/transforms.html)
for all constrained variables. In this post we walk through the transformation
$\phi$ Stan uses for [simplex](https://mc-stan.org/docs/reference-manual/transforms.html#simplex-transform.section)
constraints. Stan's linked documentation is already quite detailed; I simply
walk through the derivations in a bit more depth for my own benefit.
{% endkatexmm %}

{% katexmm %}
# Overview
The first thing to note here is that a simplex-valued variable is completely
defined by its first $d-1$ entries, since the final value is then immediately
given by
$$
x_d = 1 - \sum_{j=1}^{d-1} x_j. \tag{3}
$$
Geometrically, the set $\Delta_d$ lives within a $(d-1)$-dimensional
subspace embedded in $\mathbb{R}^d$. Practically, this means we can represent
the parameter $x$ using $d-1$ numbers. For convenience, we will thus
overload notation and write
$$
x := (x_1, \dots, x_{d-1}) \in \mathbb{R}^{d-1}. \tag{4}
$$
To re-assemble the full parameter, the final value is computed using (3). We will
thus consider a transformation $\phi: \Delta_d \to \mathbb{R}^{d-1}$.
Note the slight abuse of notation here, as we are thinking of $\phi$ as acting
only on the first $d-1$ entries of the vectors in $\Delta_d$. The
transformation used by Stan takes the form of a composition
$$
y = \phi(x) := (\phi_2 \circ \phi_1)(x), \tag{5}
$$
where $\phi_1: \Delta_d \to [0,1]^d$ and $\phi_2: [0,1]^d \to \mathbb{R}^d$.
The first map accounts for the sum-to-one constraint by mapping to
a set of intermediate variables $z := \phi_1(x) \in [0,1]^d$, and the
second map accounts for the bound constraints on the intermediate variables.
Our goals in this post are to (i) define $\phi$, (ii) derive $\phi^{-1}$,
and (iii) derive the density of the transformed variable $y$. The latter goal
is a key ingredient necessary to leverage the parameter transformation in
a Markov chain Monte Carlo (MCMC) algorithm.

# The Inverse Transformation
We start by defining $\phi^{-1}$. It will be then be straightforward to invert
this map to obtain $\phi$.

## Defining $\phi_1^{-1}$: Stick-breaking procedure
The first part of the transformation $\phi_1^{-1}$ arises naturally when viewing
simplex-valued variables through a stick-breaking procedure:
1. Start with a stick of length one.
2. Break off a portion of the stick, and let $x_1$ denote the length of this portion.
3. From the remaining piece, break off another portion and let $x_2$ denote its length.
4. Repeat the procedure to obain $x_1, \dots, x_{d-1}$.
5. Set $x_d := 1-\sum_{j=1}^{d-1} x_j$, the length of the final remaining piece.

The nice thing about this viewpoint is that it enforces the sum-to-one
constraint by construction. To take advantage of this, we define intermediate
variables $z_1, \dots, z_{d-1}$ where $z_j$ represents the proportion of the
$j^{\text{th}}$ broken piece, relative to the size of the stick from which it
was broken. Note that this is in contrast to $x_j$, which is the proportion
relative to the size of the original unit length stick. Mathematically,
\begin{align}
&x_1 = z_1, &&x_j = \left(1 - \sum_{i=1}^{j-1} x_i \right)z_j, \qquad j=1,\dots,d-1. \tag{6}
\end{align}
As they are proportions, these intermediate variables are constrained to lie
in $[0,1]$ but are not subject to any sum constraints. The equations in
(6) provide the definition for the first part of the inverse map:
$x = \phi_1^{-1}(z)$.

## Defining $\phi_2^{-1}$: Logit transformation
We now define the second portion of the inverse map, $z = \phi_2^{-1}(y)$.
Dealing with the bound constraints $z_j \in [0,1]$ is straightforward; a
standard approach involves using the
[logit](https://mc-stan.org/docs/reference-manual/transforms.html#logit-transform-jacobian.section) map
$\text{logit}: (0,1) \to \mathbb{R}$, defined by
$$
\text{logit}(t) := \log \frac{t}{1-t}, \qquad t \in (0,1). \tag{7}
$$
The inverse $\text{logit}^{-1}: \mathbb{R} \to (0,1)$ is given by the sigmoid
$$
\text{logit}^{-1}(t) = \frac{1}{1+e^{-t}}. \tag{8}
$$
At this point, we could define $z := \text{logit}^{-1}(y)$ (where the inverse
logit map is applied elementwise). This would be fine, but note that the zero
vector $y=0$ would map to $z = (1/2, \dots, 1/2)$. This corresponds to all of
the cut proportions, relative to the piece from which they are cut, being
equal. This implies stick lengths $x_1=\frac{1}{2}$, $x_2=\frac{1}{4}$, etc.
Since $y=0$ corresponds to a sort of "middle" value for $y$, it would be nice
for this to correspond to the balanced case where $x_j=\frac{1}{d}$ for
$j=1, \dots, d$ (i.e., the case where all of the pieces are equal length).
To achieve this, we need only make the slight adjustment
$$
z_j := \text{logit}^{-1}\left(y_j + \log\left(\frac{1}{d-j}\right) \right),
\qquad j=1, \dots, d-1. \tag{9}
$$
Notice that the correction term can also be written as a logit, since
\begin{align}
\text{logit}([d-j+1]^{-1})
= \log\left(\frac{[d-j+1]^{-1}}{1-[d-j+1]^{-1}}\right)
&= \log\left(\frac{1}{(d-j+1)-1}\right) \newline
&= \log\left(\frac{1}{d-j}\right). \tag{10}
\end{align}

This adjustment implies that the zero vector $y=0$ maps to the relative cut
proportions
\begin{align}
z_1 &= \text{logit}^{-1}\left(\text{logit}\left(\frac{1}{d}\right) \right) = \frac{1}{d} \newline
z_2 &= \text{logit}^{-1}\left(\text{logit}\left(\frac{1}{d-1}\right) \right) = \frac{1}{d-1} \tag{11} \newline
&\vdots
\end{align}
Feeding these values back through the map (6), we then see that
$y=0$ maps to $x=(1/d, \dots, 1/d)$, as desired.

## Density of $y$
Let $p_x(x)$ denote a probability density on $x$. The density of $y = \phi(x)$
(where $\phi$ is invertible and differentiable) is then given by the
change-of-variables formula
$$
p_y(y) = p_x(\phi^{-1}(y)) \lvert \text{det} D\phi^{-1}(y) \rvert. \tag{12}
$$
Thus, we must compute the determinant of the Jacobian of the inverse
transformation. In our present setting, notice that the stick-breaking procedure
implies that $x_j$ depends only on $y_1, \dots, y_{j-1}$. This means that
$D\phi^{-1}(y)$ is a $(d-1) \times (d-1)$ lower-triangular matrix. The
determinant term is therefore given by
$$
\text{det} D\phi^{-1}(y)
= \prod_{j=1}^{d} [D\phi^{-1}(y)]_{jj}
= \prod_{j=1}^{d} \frac{\partial x_j}{\partial y_j}
= \prod_{j=1}^{d} \frac{\partial x_j}{\partial z_j} \frac{\partial z_j}{\partial y_j}, \tag{13}
$$
where we have used the fact that the determinant of a triangular matrix is
the product of its diagonal entries. The final step is an application of the
chain rule for derivatives. We therefore need only concern ourselves with
the diagonal entries of the Jacobian. The partial derivatives in (13) are
computed as
\begin{align}
\frac{\partial x_j}{\partial z_j}
&= \frac{\partial}{\partial z_j}\left[\left(1 - \sum_{i=1}^{j-1} x_i \right)z_j \right]
= \left(1 - \sum_{i=1}^{j-1} x_i \right) \tag{14}
\end{align}
and
$$
\frac{\partial z_j}{\partial y_j}
= \frac{\partial}{\partial y_j}\left[\text{logit}^{-1}\left(y_j + \log\left(\frac{1}{d-j}\right) \right) \right]
= z_j(1-z_j). \tag{15}
$$
In (14) we have used the fact that $x_i$ does not depend on $z_j$ for $i < j$.
In the $j=1$ case we treat the summation as equaling zero, so that the derivative
is one (recall that $x_1=z_1$). In (15) we have used the fact that the derivative
of the inverse logit is itself times one minus itself. Putting these two
expressions together, we obtain
$$
\frac{\partial x_j}{\partial y_j}
= \left(1 - \sum_{i=1}^{j-1} x_i \right)z_j(1-z_j). \tag{16}
$$
This expression can then be combined with (12) and (13) to compute the density
$p_y(y)$. Notice that (16) is defined recursively with respect to the
intermediate variables $z_j$. We now provide an algorithm for computing both
$x$ and these partial derivatives. The helper variable $\ell$ tracks the length
remaining from the original stick.

<blockquote>
  <p><strong>Algorithmic implementation of $\phi^{-1}$ and its derivative.</strong> <br><br>

  <strong>Input:</strong> $ y = (y_1, \dots, y_{d-1})$. <br>
  <strong>Returns:</strong> $x=\phi^{-1}(y)$, $g := \text{diag}\{D\phi^{-1}(y)\}$. <br>

  <ol>
    <li>$z := \phi_2^{-1}(y)$, using (9).</li>
    <li>$x_1 := z_1$ and $g_1 := 1$.</li>
    <li>$\ell := 1-x_1$.</li>
    <li>For $j=2, \dots, d-1$:</li>
        <ol type="i">
            <li>$x_j := \ell z_j$.</li>
            <li>$g_j := \ell z_j(1-z_j)$</li>
            <li>$\ell := \ell - x_j$</li>
        </ol>
    <li>Return $x, g$.</li>
  </ol>
</p>
</blockquote>

# The Forward Transformation
We finish up by writing out the forward transformation $\phi$. For $\phi_1$,
we invert (6) to obtain
\begin{align}
&z_1 = x_1, &&z_j = \left(1 - \sum_{i=1}^{j-1} x_i \right)^{-1}x_j, \qquad j=1,\dots,d-1. \tag{17}
\end{align}
For $\phi_2$, we invert (9), which yields
$$
y_j = \text{logit}(z_j) - \log\left(\frac{1}{d-j}\right), \qquad j=1, \dots, d-1. \tag{18}
$$
Given $x$, we can first compute $z$ in an iterative fashion using (17), and
then compute $y$ by applying (18).

{% endkatexmm %}
