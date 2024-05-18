---
title: Gaussian Measures, Part 1 - The Univariate Case
subtitle: A brief introduction to Gaussian measures in one dimension, serving to provide the setup for an extension to multiple, and eventually infinite, dimensions.
layout: default
date: 2024-05-16
keywords: GP, Prob-Theory
published: true
---

I intend for this to be part one of a (at least) three part series on Gaussian measures,
with the ultimate goal being to understand Gaussian processes as random elements in
some suitable infinite-dimensional space. Defining a rigorous infinite-dimensional
analog of the familiar Gaussian distribution is no small task, and texts on this
subject can be quite intimidating. I've found that, personally, the key to
make these references more approachable was to first develop a deep understanding
of Gaussian measures in finite dimensions. Indeed, many of the concepts in the
infinite-dimensional case are directly motivated by their finite-dimensional
analogs. In particular, I found the parallels between the transitions from
one-to-multiple and multiple-to-infinite dimensions to be quite enlightening.
Therefore, we start here with the simplest case: Gaussian measures
in one dimension. This basic case is likely worth exploring even for those
well-acquainted with the Gaussian distribution, as it requires a shift in thinking
about densities to thinking more abstractly in terms of measures. While the
former seems perfectly sufficient in one dimension, we will find that the
measure-theoretic approach becomes a necessity in generalizing to infinite
dimensions. This post also serves to establish notation, and introduce
some key concepts that will be used throughout this series, including
Fourier transforms (characteristic functions), Radon-Nikodym derivatives,
and the change-of-variables formula.

## Density Function
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
$\mu = \delta_m$ if $\sigma^2 = 0$. When $m = 0$ we call $\mu$
**centered** or **symmetric**.
Note that in this case
the measure $\mu$ is symmetric in the sense that $\mu(B) = \mu(-B)$
for any Borel set $B$.
If, moreover, $\sigma^2 = 1$ then we call $\mu = \mathcal{N}(0, 1)$
the **standard Gaussian**.

Up to now, we have been treating $m$ and $\sigma^2$ as generic numbers, but
one can show that they correspond to the mean and variance of $\mu$,
respectively.
{% endkatexmm %}

<blockquote>
  <p><strong>Proposition.</strong>
  Let $\mu = \mathcal{N}(m, \sigma^2)$ be a Gaussian measure. Then,
  \begin{align}
  m &= \int x \mu(dx), && \sigma^2 = \int [x - m]^2 \mu(dx).
  \end{align}
  </p>
</blockquote>

{% katexmm %}
The proof in the $\mathcal{N}(m, 0)$ case is trivial given the fact that
integrating a measurable function with respect to $\delta_m$ is equivalent
to evaluating that function at $m$. Thus,
\begin{align}
&\int x \delta_m(dx) = m, &&\int [x - m]^2 \delta_m(dx) = [m - m]^2 = 0 = \sigma^2.
\end{align}
The derivations of the non-degenerate case are quite standard results, so we
won't take the time to prove them here.

## Fourier Transform
A Gaussian measure can alternatively be defined via its Fourier transform
\begin{align}
\hat{\mu}(t) := (\mathcal{F}(\mu))(s) := \int e^{its} \mu(ds).
\end{align}
The notation $\mathcal{F}(\mu)$ makes it clear that the Fourier transform is
an operator that acts on the measure $\mu$, though we will typically stick with
the more succinct notation $\hat{\mu}$. Note that this is a generalization of
the standard Fourier transform, which acts on functions, to an operator which
instead acts on measures. Probability theorists draw a distinction between
the two by referring to $\hat{\mu}$ as the **characteristic function** of
$\mu$. A classical result is that the Fourier transform of a Gaussian density is
itself a Gaussian density (up to scaling). The following result captures
this case, as well as the degenerate one.
{% endkatexmm %}

<blockquote>
  <p><strong>Proposition.</strong>
  Let $\mu = \mathcal{N}(m, \sigma^2)$ be a Gaussian measure. Then its
  Fourier transform is given by
  \begin{align}
  \hat{\mu}(t) &= \exp\left(itm - \frac{1}{2}t^2 \sigma^2 \right). \tag{2}
  \end{align}
  </p>
</blockquote>

{% katexmm %}
The Fourier transform completely characterizes $\mu$ and hence
we could have taken (2) as an alternative definition of a Gaussian measure.
Indeed, it is this definition that ends up proving much more useful, in that
it can be easily generalized to Gaussian measures in multiple, and infinite,
dimensions. We also note that $\hat{\mu}$ conveniently captures both the
degenerate and non-degenerate cases in one expression. In the degenerate case,
we have
$$
\hat{\delta}_m(t) = \int e^{its} \delta_m(ds) = e^{itm},
$$
which indeed agrees with (2) with $\sigma^2 = 0$. The complete result can
be derived in many different ways; a quick Google should satisfy the curious
reader.
{% endkatexmm %}


## Random Variables
{% katexmm %}
We have so far focused our discussion on measures $\mu$ defined on the
measurable space $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. We now extend our
discussion to include Gaussian *random variables*. In short, a random variable
$X$ is Gaussian if its distribution (i.e., law) is Gaussian. Let's be a bit
more precise though.
{% endkatexmm %}

<blockquote>
  <p><strong>Definition.</strong>
  Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, and
  $X: \Omega \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ a random variable.
  The <strong>distribution</strong> (or <strong>law</strong>) of
  $X$ is defined to be the probability
  measure $\mathbb{P} \circ X^{-1}$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.
  We write $\mathcal{L}(X) = \mu$ ($\mathcal{L}$ for "law") or $X \sim \mu$ to
  mean that the random variable $X$ has distribution $\mu$.
  </p>
</blockquote>

<blockquote>
  <p><strong>Definition.</strong>
  We say that $X$ is a <strong>Gaussian random variable</strong> if
  $\mathcal{L}(X) = \mathbb{P} \circ X^{-1}$ is a Gaussian measure.
  </p>
</blockquote>


{% katexmm %}
To be clear on notation, we write $\mathbb{P} \circ X^{-1}$ to denote the
**pushforward** of the measure $\mathbb{P}$ under the map $X$, which is given by
\begin{align}
&(\mathbb{P} \circ X^{-1})(B) := \mathbb{P}(X^{-1}(B)), &&B \in \mathcal{B}(\mathbb{R}).
\end{align}
Here, $X^{-1}(B) := \{\omega \in \Omega : X(\omega) \in B\}$
denotes the **inverse image** (i.e., **pre-image**) of $B$
under $X$.

The introduction of random variables provides a new language to express the concepts
introduced above. For example, suppose that $X \sim \mu$. Then we can write the
expectation of $X$ in a few different ways:
$$
\mathbb{E}_{\mu}[X] := \int_{\mathbb{R}} x \ \mu(dx)
= \int_{\mathbb{R}} x \ (\mathbb{P} \circ X^{-1})(dx)
= \int_{\Omega} X(\omega) \ \mathbb{P}(d\omega).
$$
The final equality is courtesy of the **change-of-variables formula**, a result
that we will be using repeatedly throughout these notes. Following the above
notation, we can also write the Fourier transform $\hat{\mu}$ in terms of the
random variable $X$ as
$$
\hat{\mu}(t) = \mathbb{E}_{\mu}\left[e^{itX} \right].
$$
{% endkatexmm %}


## The Central Limit Theorem
While it is not the focus of these notes, a post on Gaussian measures seems
incomplete without mentioning the central limit theorem (CLT). Proving
this result has the added benefit of reviewing some useful properties
of Fourier transforms.
