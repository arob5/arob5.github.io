---
title: Markov Melding
subtitle: Combining and splitting models in a Bayesian framework.
layout: default
date: 2024-11-03
keywords: UQ
published: true
---

In this post, I summarize a technique called *Markov melding*, originally
proposed in Goudie et al (2023).

# Motivation and Setup
{% katexmm %}
Models of complex systems (climate, disease, etc.) are typically composed
of many sub-models. For example, climate models couple together models
of the atmosphere, land, ocean, and cryosphere. The component models may
be developed largely in isolation, before being combined with the others to
form a complex model for the whole system. In the context of Markov melding, we
are considering that these component models are specified within a Bayesian
framework, so that each component can be identified with a joint
probability distribution of the component model parameters and data.
Letting $M$ denote the number of components, we denote the density for
the joint distribution of the $m^{\text{th}}$ submodel by
\begin{align}
&p_m(\phi, \psi_m, Y_m), &&m = 1, \dots, M \tag{1}
\end{align}
where $Y_m$ denotes the data used in the $m^{\text{th}}$ submodel. The
submodel parameters are
$$
\theta_m := (\phi, \psi_m), \tag{2}
$$
and we make a distinction between component-specific parameters $\psi_m$ and
parameters $\phi$ that are shared across all submodels. Goudie et al (2019)
refer to $\phi$ as the *link parameters*. A typical modeling workflow might
involve obtaining the posterior distributions $p_m(\phi, \psi_m|Y_m)$ for each
submodel independently, and then combining them post-hoc in some fashion to study
the system as a whole. Bayesian melding is an alternative approach that seeks
to construct a joint distribution $p(\phi, \psi_1, \dots, \psi_m, Y_1, \dots, Y_m)$
that combines all components into a unified probabilistic model. Then inference
can proceed by characterizing the posterior distribution
$p(\phi, \psi_1, \dots, \psi_m | Y_1, \dots, Y_m)$. The key distinction here is that
all of the data $Y_1, \dots, Y_m$ is used to inform the parameters across all
components. The isolated component posterior $p_m(\phi, \psi_m|Y_m)$ might look quite
different from $p(\phi, \psi_m|Y_1, \dots, Y_m)$ due to the influence of the other
components.

On a final note before proceeding with the method, we emphasize the assumption that
$\phi$ is the only link between the submodels. From a probabilistic standpoint,
this implies that we will be considering joint distributions with the property
that component-specific variables in any two distinct models are conditionally
independent given the link parameter $\phi$; symbolically,
$$
(\psi_m, Y_m) \perp (\psi_\ell, Y_\ell) | \phi, \qquad m \neq \ell. \tag{3}
$$
{% endkatexmm %}

# Markov Combination
{% katexmm %}
We start with the assumption that the submodels are consistent in the sense
that they all imply the same prior distribution on the link parameter $\phi$; i.e.,
$$
p_1(\phi) = \cdots = p_M(\phi). \tag{4}
$$
The method of constructing a joint distribution under assumption (4) is
referred to as *Markov combination* (Dawid and Lauritzen, 1993). This assumption
makes the definition of a joint distribution relatively straightforward. Denoting
the common prior on the link parameter by $p(\phi) := p_m(\phi)$, the Markov
combination approach defines a joint distribution $p_{\text{comb}}$ by
\begin{align}
p_{\text{comb}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M)
&:= p_{\text{comb}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M|\phi)p(\phi) \newline
&:= p(\phi) \prod_{m=1}^{M} p_m(\psi_m, Y_m|\phi). \tag{5}
\end{align}
We emphasize the both of the above equalities are *definitions*, though both
definitions are quite natural in this setting. The first ensures that the
common prior $p(\phi)$ is preserved by the joint distribution, while the second
defines the conditional density (conditional on $\phi$) using the conditional
independence properties noted in (3). This ensures the submodel distributions
$p_m(\phi, \psi_m, Y_m)$ are also preserved. Specifically, (5) satisfies
$$
p_{\text{comb}}(\phi, \psi_m, Y_m) = p_m(\phi, \psi_m, Y_m), \qquad m = 1, \dots, M.
\tag{6}
$$
It will be useful to rewrite (5) as
\begin{align}
p_{\text{comb}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M)
&= p(\phi) \prod_{m=1}^{M} p_m(\psi_m, Y_m|\phi) \newline
&= p(\phi) \prod_{m=1}^{M} \frac{p_m(\phi, \psi_m, Y_m)}{p(\phi)} \newline
&= \frac{1}{p(\phi)^{M-1}} \prod_{m=1}^{M} p_m(\phi, \psi_m, Y_m), \tag{7}
\end{align}
which shows that $p_{\text{comb}}$ is just the product of the submodel distributions
$p_m$, normalized by $p(\phi)^{M-1}$ to account for the fact that the prior
$p(\phi)$ is multiplied $M$ times in the product. From the expression (7),
we may easily verify (6); letting
$\psi_{-m} := (\psi_1, \dots, \psi_{m-1}, \psi_{m+1}, \dots, \psi_M)$ and
similarly for $Y_{-m}$, we obtain
\begin{align}
p_{\text{comb}}(\phi, \psi_m, Y_m)
&= \int p_{\text{comb}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M) d\psi_{-m}dY_{-m} \newline
&= \frac{1}{p(\phi)^{M-1}}p_m(\phi, \psi_m, Y_m) \prod_{\ell \neq m} \int p_m(\phi, \psi_m, Y_m) d\psi_\ell dY_\ell \newline
&= \frac{1}{p(\phi)^{M-1}}p_m(\phi, \psi_m, Y_m) \prod_{\ell \neq m} p_\ell(\phi) \newline
&= \frac{1}{p(\phi)^{M-1}}p_m(\phi, \psi_m, Y_m) p(\phi)^{M-1} \newline
&= p_m(\phi, \psi_m, Y_m).
\end{align}

While the definition (5) seems somewhat natural, one may wish for a more
quantitative justification. To this end, Massa and Lauritzen (2010) show that
$p_{\text{comb}}$ has maximal entropy among all distributions satisfying the
$M$ constraints (6). Loosely speaking, this tells us that $p_{\text{comb}}$
is the least constrained of all the distributions satisfying (6), a desirable
property in the absence of additional knowledge on the relative merits of
the submodels.
{% endkatexmm %}

# Markov Melding
{% katexmm %}
Goudie et al (2019) consider propose a generalization of Markov combination to
the setting where the marginals $p_m(\phi)$ may not be consistent across
submodels. In this case, we can no longer simply pull out the common prior in
the first line of definition (5). To address this, the authors' propose first
replacing the marginals $p_m(\phi)$ with a common "pooled" prior
$p_{\text{pool}}(\phi)$. After performing this "marginal replacement", Markov
combination can then be applied with respect to the modified submodels.

## Marginal Replacement
We first consider modifying the submodels to ensure they are consistent with
respect to the marginal of the link parameter. The idea is to replace the
original submodels
$$
p_m(\phi, \psi_m, Y_m) = p_m(\phi) p_m(\psi_m, Y_m | \phi)
$$
with modifications that share a common $\phi$-marginal:
\begin{align}
p_{\text{repl},m}(\phi, \psi_m, Y_m)
&:= p_{\text{pool}}(\phi) p_m(\psi_m, Y_m | \phi) \tag{8} \newline
&= \frac{p_{\text{pool}}(\phi)}{p_m(\phi)} p_m(\phi, \psi_m, Y_m). \tag{9}
\end{align}
We see in (8) that the submodel conditional (on $\phi$) is unchanged by the
marginal replacement. The expression (9) gives an alternative viewpoint,
where we view the original joint submodel distribution $p_m$ as re-weighted
by some modification factor $p_{\text{pool}}(\phi)/p_m(\phi)$. This, of course,
begs the question of how to choose $p_{\text{pool}}(\phi)$. Naturally, we would
hope this distribution adequately summarizes the original marginals
$p_1(\phi), \dots, p_M(\phi)$ in some sense. We put aside this question for the
time being, and assume that we have fixed some pooled prior $p_{\text{pool}}(\phi)$.
Given that a common prior $p_{\text{pool}}(\phi)$ has been chosen, Goudie et
al (2019) provides some justification for the definition (9):
$$
p_{\text{repl},m}(\phi, \psi_m, Y_m)
= \text{argmin}_{q \in \mathcal{Q}} \text{D}_{\text{KL}}(q || p_m),
$$
where
$$
\mathcal{Q} = \left\{q(\phi, \psi_m, Y_m) | q(\phi) = p_{\text{pool}}(\phi) \ \forall \phi \right\}.
$$
In words, the marginally replaced distribution minimizes the KL-divergence with
respect to $p_m$ over all distributions with $\phi$-marginals equal to
$p_{\text{pool}}(\phi)$.

## Markov Combination with respect to the modified submodels
We can now apply Markov combination with respect to the modified submodels
$p_{\text{repl},1}, \dots, p_{\text{repl},M}$. Following (5), define
\begin{align}
p_{\text{meld}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M)
&:= p_{\text{pool}}(\phi) \prod_{m=1}^{M} p_{\text{repl},m}(\psi_m, Y_m|\phi) \newline
&= p_{\text{pool}}(\phi) \prod_{m=1}^{M} p_{m}(\psi_m, Y_m|\phi), \tag{10}
\end{align}
where the second equality uses the fact that marginal replacement preserves
the conditionals (see (9)).
We can alternatively expand (10) as
$$
p_{\text{meld}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M)
= p_{\text{pool}}(\phi) \prod_{m=1}^{M} \frac{p_{m}(\phi, \psi_m, Y_m)}{p_m(\phi)}, \tag{11}
$$
which provides a direct comparison to (7). All of the facts noted for Markov
combination still apply here, but now with respect to the marginally replaced
models $p_{\text{repl},m}$, not the original submodels. In particular, we have that
$$
p_{\text{meld}}(\phi, \psi_m, Y_m) = p_{\text{repl},m}(\phi, \psi_m, Y_m), \qquad m = 1, \dots, M,
\tag{12}
$$
but in general
$$
p_{\text{repl},m}(\phi, \psi_m, Y_m) \neq p_{m}(\phi, \psi_m, Y_m), \tag{13}
$$
meaning that $p(\text{meld})$ does not preserve the submodel joint distributions.
However, $p(\text{meld})$ still does enjoy the *conditional* preservation
property:
$$
p_{\text{meld}}(\psi_m, Y_m|\phi) = p_{m}(\psi_m, Y_m|\phi), \qquad m = 1, \dots, M, \tag{14}
$$
which should be somewhat intuitive given that the marginal replacement method
does not modify the conditionals (see (8)). To verify this, apply (11) to see that
\begin{align}
p_{\text{meld}}(\psi_1, \dots, \psi_M, Y_1, \dots, Y_M|\phi)
&= \frac{p_{\text{meld}}(\phi, \psi_1, \dots, \psi_M, Y_1, \dots, Y_M)}{p_{\text{pool}}(\phi)} \newline
&= \frac{1}{p_{\text{pool}}(\phi)} p_{\text{pool}}(\phi) \prod_{m=1}^{M} \frac{p_{m}(\phi, \psi_m, Y_m)}{p_m(\phi)} \newline
&= \prod_{m=1}^{M} p_{m}(\psi_m, Y_m|\phi). \tag{15}
\end{align}
We see from (15) that $p_{\text{meld}}$ still preserved the conditional
independence structure summarized in (3), and that (14) falls out as a result.

## Deterministic Variables
Goudie et al (2019) also consider the scenario where the link variable $\phi$
may appear in a submodel as a deterministic function of the other submodel
variables, rather than being explicitly defined by the joint distribution
$p_m$. In particular, suppose one of the submodels takes the form
\begin{align}
&p_{\theta}(\theta, \psi, Y), &&\phi = f(\theta), \tag{16}
\end{align}
for some deterministic, differentiable map
$f: \mathbb{R}^{\ell} \to \mathbb{R}^k$ with
$k \leq \ell$. If $k=\ell$ and $f$ is invertible, then the density of $\phi$
is given by the change-of-variables formula
$$
p_{\phi}(\phi) = |Df^{-1}(\phi)| p_{\theta}(f^{-1}(\phi)), \tag{17}
$$
where $|Df^{-1}(\phi)|$ denotes the determinant of the Jacobian $Df^{-1}$
evaluated at $\phi$; $Df^{-1}(\phi) \in \mathbb{R}^{\ell \times k}$.
Next, consider the case where $k < \ell$. Here, we assume that $f$ can be
extended to an invertible function $\tilde{f}: \mathbb{R}^{\ell} \to \mathbb{R}^{\ell}$
of the form
\begin{align}
\tilde{f}(\theta) &= \begin{bmatrix} f(\theta) \newline g(\theta) \end{bmatrix},
&&g: \mathbb{R}^{\ell} \to \mathbb{R}^{\ell-k}. \tag{18}
\end{align}
The density of $(\phi, \tilde{\phi})$ induced by the distribution on $\theta$
is then given by
$$
p_{\phi}(\phi,\tilde{\phi}) = |D\tilde{f}^{-1}(\phi, \tilde{\phi})| p_{\theta}(\tilde{f}^{-1}(\phi, \tilde{\phi})). \tag{19}
$$
Similarly, we can consider the change-of-variables for the complete joint
distribution
$$
p_{\phi}(\phi, \tilde{\phi}, \psi, Y)
= |D\tilde{f}^{-1}(\phi, \tilde{\phi})|
p_{\theta}(\tilde{f}^{-1}(\phi, \tilde{\phi}), \psi, Y). \tag{20}
$$
We can then obtain the desired submodel distribution over $(\phi,\psi,Y)$
by marginalizing the nuisance parameter $\tilde{\phi}$:
$$
p(\phi, \psi, Y) := \int p_{\phi}(\phi, \tilde{\phi}, \psi, Y) d\tilde{\phi}. \tag{21}
$$
One concern here is that this procedure will produce difference results for
different choices of the extension function $g(\cdot)$. Goudie et al (2019)
show that this is not the case.
{% endkatexmm %}

# Inference
{% katexmm %}
We now consider algorithmic methods for characterizing the posterior
distribution
$$
p_{\text{meld}}(\phi, \psi_1, \dots, \psi_M|Y_1, \dots, Y_M) \propto
p_{\text{pool}}(\phi) \prod_{m=1}^{M} \frac{p_{m}(\phi, \psi_m, Y_m)}{p_m(\phi)}. \tag{22}
$$
So long as the joint density $p_{\text{meld}}$ can be evaluated pointwise, then
one can consider applying a general-purpose inference algorithm (e.g.,
Metropolis-Hastings) in the typical fashion. Goudie et al (2019) consider some
specific MCMC algorithms that are tailored to the specific structure in the
density (22). Throughout this section we will utilize the shorthand
$\psi_{1:\ell} := (\psi_1, \dots, \psi_\ell)$ and
$y_{1:\ell} := (y_1, \dots, y_\ell)$ for brevity. Here, $y_m$ denotes a particular
fixed realization of the data random variable $Y_m$. In considering
Metropolis-Hastings (MH) algorithms, we will write $\psi_m^*$ and $\psi_m$ to indicate
the proposed and current values, respectively, of the $m^{\text{th}}$ submodel
parameter (and likewise for $\phi^*$).

## Metropolis-within-Gibbs
The structure of (22) suggests updating the component-specific parameters $\psi_m$
one at a time, and hence a Metropolis-within-Gibbs (MwG) scheme.

### $\psi_m$ update
We first consider updating $\psi_m$ conditional on all other parameters. This
implies a MH update with respect to the target density
$p_{\text{meld}}(\psi_m|Y_{1:M}, \phi, \psi_{-m})$. The standard MH update
consists of the following steps:
1. Propose a new value: $\psi_m^* \sim q(\psi_m^*|\psi_m)$.
2. Accept the new value with probability:

$$
\alpha(\psi_m^* |\psi_m) := \min\left(1, r(\psi_m^*|\psi_m) \right), \tag{22}
$$
where
$$
r(\psi_m^*|\psi_m) := \frac{p_{\text{meld}}(\psi_m^*|Y_{1:M}, \phi, \psi_{-m})}{p_{\text{meld}}(\psi_m|Y_{1:M}, \phi, \psi_{-m})} \cdot
\frac{q(\psi_m|\psi_m^*)}{q(\psi_m^*|\psi_m)}. \tag{23}
$$
Here, $q$ is the proposal density, with $q(\psi_m^*|\psi_m)$ denoting the
density of proposing $\psi_m^*$. The acceptance ratio $r(\psi_m^*|\psi_m)$
simplifies considerably in this case, since:

\begin{align}
\frac{p_{\text{meld}}(\psi_m^*|y\_{1:M}, \phi, \psi_{-m})}{p_{\text{meld}}(\psi_m|y\_{1:M}, \phi, \psi_{-m})}
&\propto \frac{p_{\text{meld}}(\psi_m^*, y\_{1:M}, \phi, \psi_{-m})}{p_{\text{meld}}(\psi_m, y\_{1:M}, \phi, \psi_{-m})}
\end{align}

where we have simply plugged in expression (11) and cancelled terms. Notice that
the resulting expression, (24), only contains terms related to the the
$m^{\text{th}}$ submodel. In other words, (24) is the exact density ratio
we would obtain if considering a MwG scheme applied only to the $m^{\text{th}}$
submodel in isolation.

### $\phi$ update
We next consider updating the link parameter, conditional on all
component-specific parameters. We similarly consider some proposal
$\phi^* \sim q(\phi^*|\phi)$, in which case the density ratio becomes
$$
\frac{p_{\text{meld}}(\phi^*|y_{1:M}, \psi_{1:M})}{p_{\text{meld}}(\phi|y_{1:M}, \psi_{1:M})}
= \frac{p_{\text{pool}}(\phi^*)}{p_{\text{pool}}(\phi)}
\prod_{m=1}^{M} \frac{p_m(\phi^*, \psi_m, y_m)}{p_m(\phi, \psi_m, y_m)}
\frac{q(\phi|\phi^*)}{q(\phi^*|\phi)}. \tag{25}
$$



{% endkatexmm %}

# References
1. Joining and Splitting Models with Markov Melding (Goudie et al., 2019)
2. Combining chains of Bayesian models with Markov melding (Manderson and Goudie, 2023)
3. A numerically stable algorithm for integrating Bayesian models using Markov melding (Manderson and Goudie, 2022)
4. Dawid AP, Lauritzen SL. Hyper Markov laws in the statistical analysis of decomposable graphical models. Annals of Statistics. 1993;21(3):1272–1317. 2, 5, 6, 23.
5. Massa MS, Lauritzen SL. Combining statistical models. In: Viana MAG, Wynn HP, editors. Contemporary Mathematics: Algebraic Methods in Statistics and Probability II. 2010. pp. 239–260. 2, 5, 6, 7, 9, 23.
6. Poole D, Raftery AE. Inference for deterministic simulation models: The Bayesian melding approach. Journal of the American Statistical Association. 2000;95(452):1244–1255. 2, 7, 23.
