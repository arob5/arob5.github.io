---
title: Gaussian Measures in Finite Dimensions
subtitle: A fairly deep dive into the univariate and multivariate Gaussian distributions, in preparation for an extension to infinite dimensions.
layout: default
date: 2024-05-16
keywords: GP, Prob-Theory
published: true
---

# The Univariate Gaussian
We start by recalling that the univariate Gaussian density takes the form
{% katexmm %}
$$
\mathcal{N}(x|m, \sigma^2) := \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left\{-\frac{1}{2\sigma^2}(x - m)^2 \right\}.
$$
{% endkatexmm %}
We're typically used to defining the Gaussian as a random variable with density
equal to $\mathcal{N}(x|m, \sigma^2)$. Since
we're interested in measures here, we can simply define the corresponding
measure by integrating this density.

<blockquote>
  <p><strong>Definition.</strong>
  A probability measure $\mu$ defined on the Borel measurable space
  $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ is called <strong>Gaussian</strong> provided that,
  for any Borel set $B \in \mathcal{B}(\mathbb{R})$, either
  \begin{align}
  \mu(B) = \int_{B} \mathcal{N}(x|m, \sigma^2) dx
  \end{align}
  for some fixed $m \in \mathbb{R}$ and $\sigma^2 > 0$; or
  \begin{align}
  \mu(B) = \delta_m(B).
  \end{align}
  </p>
</blockquote>

{% katexmm %}
Note that a Gaussian measure is a *Borel measure*; that is, we define it on the
Borel sets $\mathcal{B}(\mathbb{R})$. This will remain true as we extend to multiple,
and even infinite, dimensions. The first case in the above definition is the
familiar one, seeing as we're simply integrating over the Gaussian density.
The notation $dx$ in the
integral formally means that the integration is with respect to the Lebesgue
measure $\lambda$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. Another way
we could phrase this is to say that a probability measure $\mu$ is Gaussian
provided that its Radon-Nikodym derivative with respect to $\lambda$ is
$\mathcal{N}(x|m, \sigma^2)$; i.e.,
\begin{align}
\frac{d\mu}{d\lambda}(x) = \mathcal{N}(x|m, \sigma^2).
\end{align}
The density, of course, is only defined if $\sigma^2 > 0$. It turns out to be
nice to also allow for the $\sigma^2 = 0$ case. While $\mathcal{N}(x|m, 0)$
is not defined, we can formalize this notion as a Dirac measure $\delta_m$, which
is defined by
$$
\delta_m(B) := 1[m \in B].
$$
In this case the Gaussian measure is simply a point mass - all of the probability
is concentrated at the mean $m$. We call such a Gaussian measure **degenerate**,
while Gaussian measures that admit densities are labelled **non-degenerate**.
We write $\mu = \mathcal{N}(m, \sigma^2)$ to signify that $\mu$ is a Gaussian
measure with density $\mathcal{N}(x|m, \sigma^2)$ if $\sigma^2 > 0$, or
$\mu = \delta_m$ if $\sigma^2 = 0$.

Up to now, we have been treating $m$ and $\sigma^2$ as generic numbers, but
one can show that they correspond to the mean and variance of $\mu$,
respectively.
{% endkatexmm %}

<blockquote>
  <p><strong>Proposition.</strong>
  Let $\mu = \mathcal{N}(\mu, \sigma^2)$ be a Gaussian measure. Then,
  \begin{align}
  m &= \int x \mu(dx), && \sigma^2 = \int [x - m]^2 \mu(dx).
  \end{align}
  </p>
</blockquote>

{% katexmm %}
The proof in the $\mathcal{N}(\mu, 0)$ case is trivial given the fact that
integrating a measurable function with respect to $\delta_m$ is equivalent
to evaluating that function at $m$. Thus,
\begin{align}
&\int x \delta_m(dx) = m, &&\int [x - m]^2 \delta_m(dx) = [m - m]^2 = 0 = \sigma^2.
\end{align}
The derivations of the non-degenerate case are quite standard results, so we
won't take the time to prove them here.

{% endkatexmm %}
