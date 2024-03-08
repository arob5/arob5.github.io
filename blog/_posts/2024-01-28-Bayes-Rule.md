---
title: The Measure-Theoretic Context of Bayes' Rule
subtitle: I describe Bayes' rule in a measure-theoretic context, explain how it can be viewed as a non-linear operator on probability measures, and detail applications to Bayesian inverse problems.
layout: default
date: 2024-01-28
keywords: Bayes, Inverse-Problem, Prob-Theory
published: true
---

Bayes' rule is typically written as something like
{% katexmm %}
\begin{align}
p(u|y) &= \frac{p(y|u)p(u)}{p(y)} = \frac{p(y|u)p(u)}{\int p(y|u)p(u)du}, \tag{1}
\end{align}
describing the connection between the two conditional probability density
functions $p(u|y)$ and $p(y|u)$. It is not very common to see this result
cast in a more rigorous measure-theoretic setting. This post explores how this
can be accomplished.
{% endkatexmm %}

## Defining the Required Ingredients
{% katexmm %}
Most of the work in rigorously stating Bayes' theorem goes into the setup.
We therefore start by building up the necessary measure-theoretic
foundation. Since Bayes' rule considers probabilistic
dependencies between two random variables $U$ and $Y$, we start by
rigorously defining these random variables.

### Marginal Distributions
Let $(\Omega, \mathcal{A}, \mathbb{P})$
be the underlying probability space which will be used as the foundation for
all of the subsequent development. To define $U$ and $Y$ as random
variables mapping from this space, we introduce the measurable Borel spaces
$(\mathcal{U}, \mathcal{B}(\mathcal{U}))$ and
$(\mathcal{Y}, \mathcal{B}(\mathcal{Y}))$ such that the random variables are
defined as measurable maps $U: \Omega \to \mathcal{U}$ and
$Y: \Omega \to \mathcal{Y}$. The probability distribution (i.e. law) for $U$ is
the function $\mu_U: \mathcal{B}(\mathcal{U}) \to [0,1]$ given by
$\mu_U(A) = \mathbb{P}(U \in A) = \mathbb{P}(U^{-1}(A))$ for all
$A \in \mathcal{B}(\mathcal{U})$. The
distribution $\mu_Y$ of $Y$ is defined analogously.

### Conditional Distributions
The next ingredient we need is a notion of conditional distributions in order
to formally define what we mean by $p(y|u)$ and $p(u|y)$ in (1). To side-step
all of the technical issues associated with rigorously defining a notion of
conditional probability, it is common to simply assume that for each
$u \in \mathcal{U}$, there is a valid probability distribution $P(u, \cdot)$
which represents the conditional distribution $Y|U=u$ (and vice versa for
the conditional $U|Y=y$). This assumption is nicely encoded by assuming the
existence of a *probability kernel* (i.e., *transition kernel*)
$P: \mathcal{U} \times \mathcal{B}(\mathcal{Y}) \to [0,1]$. By definition,
the probability kernel satisfies:
1. For each $u \in \mathcal{U}$, $P(u, \cdot): \mathcal{B}(\mathcal{Y}) \to [0, 1]$
is a probability measure on $(\mathcal{Y}, \mathcal{B}(\mathcal{Y}))$,
which we will denote by $P_u(\cdot) := P(u, \cdot)$.
2. For each $B \in \mathcal{B}(\mathcal{Y})$, $P(\cdot, B): \mathcal{U} \to [0,1]$
is a measurable function.

While this probability kernel is generic, we will think of $P_u$ as representing
the conditional distribution of $Y|U=u$. To define what we mean by the other
conditional $U|Y=y$ we similarly assume there is a probability kernel
$\mu^Y: \mathcal{Y} \times \mathcal{B}(\mathcal{U}) \to [0,1]$, again
satisfying the two required properties for a probability kernel. We will use
the notation $\mu^y(\cdot) := \mu^Y(y, \cdot)$ for the probability measure
that results from fixing the first argument of the kernel at
$y \in \mathcal{Y}$.

Note that since are assuming both of these kernels exist, the statement of
Bayes' theorem presented below will take the form "If both conditional
distributions exist (and are given by the probability kernels described above),
then they must be related in the following way...". This might feel a bit
unsatisfying at first, but it is not difficult to show that in the common
settings of interest, it is straightforward to define these probability kernels.
I discuss this in more detail in the appendix.

### Densities
The last missing ingredient is to consider
probability density functions (or more generally Radon-Nikodym derivatives)
for the probability measures introduced above. While the typical informal
statement of Bayes' theorem implicitly assumes the existence of Lebesgue
densities (or probability mass functions) we can generalize this by assuming
that there is some $\sigma$-finite measure $\nu$ such that $P_u$ is
absolutely continous with respect to $\nu$ for each $u \in \mathcal{U}$; i.e.,
$P_u \ll \nu$ $\forall u \in \mathcal{U}$. We recall that absolute continuity
means that $\nu(B) = 0 \implies P_u(B)=0$ for all $B \in \mathcal{B}(\mathcal{Y})$.
Intuitively,
in order to re-weight $\nu$ to obtain $P_u$, $\nu$ better not be zero where
$P_u$ is positive. In applications, the most common choices
for $\nu$ are the Lebesgue or counting measure, which lead to the standard
presentation of Bayes' rule for continous and discrete random variables,
respectively.

### The Statistical Interpretation
I was careful to keep things quite generic above; at its core Bayes' theorem
concerns the joint and conditional distributions between two generic random
variables $U$ and $Y$. However, the result is most commonly seen applied to
the field of Bayesian statistics, so I take a moment to map the above definitions
onto their common Bayesian interpretations. In a Bayesian context, the random
variable $Y$ is the data, while $U$ is the parameter in the statistical model
being considered. The measure $\mu_U$ thus represents the prior distribution
on the parameter. The probability kernel $P$ formalizes the notion of a parametric
statistical model. Indeed, in a standard parametric statistical setting each
fixed value $u$ for the parameter $U$ yields a different data-generating process;
this data-generating process is encoded by $P_u$. The Radon-Nikodym
derivative of $P_u$ (when it exists) with respect to the Lebesgue density
is typically referred to the *likelihood function* (viewed as a function of
$u$ with data $y \in \mathcal{Y}$ fixed). From this point going
forward I'll fall back on the Bayesian terminology since it is convenient and
often enlightening, but of course Bayes' theorem is agnostic to how the
probability distributions are interpreted.
{% endkatexmm %}


## Finally, Bayes' theorem
With all of this setup out of the way, we can now state the theorem. However,
before doing so, let's provide one last bit of motivation. Assuming the
existence of the two probability kernels $P_u$ and $\mu^Y$ provides two
equivalent ways to write the joint distribution on $(U, Y)$. Letting
$A \in \mathcal{B}(\mathcal{U})$ and $B \in \mathcal{B}(\mathcal{Y})$, we have
\begin{align}
\mathbb{P}(U \in A, Y \in B)
&= \int_A \int_B \mu_U(du)P(u, dy) = \int_B \int_A \mu_Y(dy) \mu^Y(y, du).
\end{align}
Moreover, since $P(u, \cdot) \ll \nu$ by assumption, the middle term can be
re-written as
\begin{align}
\int_A \int_B \mu_U(du)P(u, dy) &= \int_A \int_B \mu_U(du) \frac{dP_u}{d\nu}(y) \nu(dy).
\end{align}
We know that Bayes' theorem give the relationship between the densities
associated with the two conditional distributions. Since the above equations
give
\begin{align}
\int_A \int_B \mu_U(du) \frac{dP_u}{d\nu}(y) \nu(dy)
&= \int_B \int_A \mu_Y(dy) \mu^Y(y, du),
\end{align}
we see that we are almost there, aside from a few missing ingredients. The most
obvious missing piece is the density for the other conditional distribution;
i.e., $\frac{d\mu^Y}{d\nu}$. The first step in writing down Bayes' theorem
should thus be establishing the existence of this density by showing that
$\mu^y \ll \nu$ for all $y \in \mathcal{Y}$. Then, so long as we can reverse
the order of the one of these integrals, we can start seeing how to derive
the relationship between the densities. Indeed, this is exactly how the proof
proceeds. Without further ado, here is the theorem
(the proof is given in the appendix).

**Bayes' Theorem.** Under the assumptions outlined in the preceeding sections,
1. The posterior is absolutely continuous with respect to the prior:
\begin{align}
\mu^y \ll \mu_U \text{ for all } y \in \mathcal{Y}
\end{align}

2. The Radon-Nikodym derivative of the posterior with respect to the prior is
proportional to the likelihood:
\begin{align}
\frac{d\mu^y}{d\mu_U}(u) \propto \frac{dP_u}{d\nu}(y).
\end{align}
More precisely, and including the normalization constant, the posterior measure
is given by  
\begin{align}
\mu^y(A) &= \mathbb{P}(U \in A|Y=y)
= \frac{\int_A \frac{dP_u}{d\nu}(y) \mu_U(du)}{\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du)},
\end{align}
for $A \in \mathcal{B}(\mathcal{U})$, for all $y \in \mathcal{Y}$ such that
the denominator is not $0$ or infinity.

From this perspective, Bayes' theorem describes the change-of-measure that changes
the prior $\mu$ into the posterior $\mu^y$ after conditioning on data $y$. The
theorem shows that this map simply involves re-weighting the prior by the
likelihood, and then normalizing the result.

## The Operator-Theoretic Viewpoint


## Recovering the Informal Statement
{% katexmm %}
One last lingering question might be how to provide a more formal justification
for the commonly seen phrasing of Bayes' rule, as in (1). We have seen that the
rigorous statement of Bayes' rule concerns, $\frac{d\mu^y}{d\mu_U}(u)$,
the Radon-Nikodym derivative of the posterior with respect to the prior,
which is expressed in terms of some base dominating measure $\nu$. In the
standard informal presentation of Bayes' rule for continuous random variables
(e.g., (1)), $\nu$ is taken to be the Lebesgue measure $\lambda$. We can then use the
standard likelihood notation
\begin{align}
p(y|u) := \frac{dP_u}{d\lambda}(y)
\end{align}
to refer to the Radon-Nikodym derivative of $P_u$ with respect to the Lebesgue
measure $\lambda$; i.e. $p(y|u)$ is the *Lebesgue density* of $P_u$.

Next, note that in our rigorous statement of Bayes' theorem, all integrals are
currently with respect to the prior measure $\mu_U$. To recover the typical
informal formula, we require all integrals to be with respect to the Lebesgue
density, which requires the additional assumption $\mu_U \ll \lambda$. Under this
assumption, the Lebesgue density of $\mu_U$ exists and we will write it as
\begin{align}
p(u) := \frac{d\mu_U}{d\lambda}(y).
\end{align}
Note that there are technically two different Lebesgue
measures being considered here: the first one is defined on the measurable space
$(\mathcal{Y}, \mathcal{B}(\mathcal{Y}))$ and corresponds to the particular choice
of $\nu$; we have introduced a second Lebesgue measure on
$(\mathcal{U}, \mathcal{B}(\mathcal{U}))$ in order to obtain Lebesgue densities
for the prior measure.

With all of this established, we now have
\begin{align}
\mu^y(A)
&= \frac{\int_A \frac{dP_u}{d\lambda}(y) \mu_U(du)}{\int_{\mathcal{U}} \frac{dP_u}{d\lambda}(y) \mu_U(du)}
= \int_A \frac{p(y|u)p(u)}{\int_{\mathcal{U}} p(y|u)p(u)du}du
\end{align}
Here, we use the typical Riemann integral notation $\lambda(du) = du$ for
integration with respect to the Lebesgue measure. The last expression implies
that $\mu^y \ll \lambda$ and moreover that the Lebesgue density of
the posterior is given by
\begin{align}
\frac{d\mu^y}{d\lambda}(u) &= \frac{p(y|u)p(u)}{\int_{\mathcal{U}} p(y|u)p(u)du},
\end{align}
which is precisely the typical form in which Bayes' rule is presented. To
obtain the form of Bayes' rule for discrete random variables, we simply
substitute the Lebesgue measure for the counting measure, which has the effect
of replacing densities with mass functions and integrals with sums.
{% endkatexmm %}

## Application: Bayesian Inverse Problems

## Some more technical details: The Disintegration Theorem

## Appendix
### Bayes' Theorem Proof
We recall from the setup above that
{% katexmm %}
\begin{align}
\mathbb{P}(U \in A, Y \in B)
&= \int_A \int_B \frac{dP_u}{d\nu}(y) \nu(dy) \mu_U(du)
= \int_B \int_A \mu^y(du) \mu_Y(dy),
\end{align}
where $A \in \mathcal{B}(\mathcal{U})$, $B \in \mathcal{B}(\mathcal{Y})$.
The general structure of this proof involves manipulating these two terms
so that the integrands can be combined; this requires,
1. Changing the integration order in one of the expressions (we will do this
  for the first of the two).
2. Writing the integrals with respect to a common measure (we will change the
  measure in the second term so that the outer integration is also with respect
  to $\nu$).

There is also the technical concern that the normalizing constant for the
posterior distribution is zero or infinite; it turns out this is not an issue
as it occurs with zero $\mu_U$-probability. We verify this at the end of the
proof to avoid cluttering the main points.

#### Changing the order of integration
We begin by flipping the order of the integration in the first term. Note that
the function $(u, y) \mapsto \frac{dP_u}{d\nu}(y)$ can be shown to
be measurable. Moreover, it is non-negative and thus an application of
[Tonelli's theorem](https://en.wikipedia.org/wiki/Fubini%27s_theorem#Tonelli's_theorem)
gives
\begin{align}
\int_A \left[\int_B \frac{dP_u}{d\nu}(y) \nu(dy)\right] \mu_U(du)
&= \int_B \left[\int_A \frac{dP_u}{d\nu}(y) \mu_U(du) \right] \nu(dy).
\end{align}


#### Changing the measure
To write the second term with respect to $\nu$, we need only confirm that
$\mu_Y \ll \nu$ so that $\frac{d\mu_Y}{d\nu}$ exists. Indeed, if this is true
then
\begin{align}
\int_B \int_A \mu^y(du) \mu_Y(dy)
&= \int_B \mu^y(A) \mu_Y(dy)
= \int_B \mu^y(A) \frac{d\mu_Y}{d\nu}(y) \nu(dy).
\end{align}
To justify this, we can use the expression derived above to obtain
\begin{align}
\mu_Y(B)
&= \mathbb{P}(U \in \mathcal{U}, Y \in B)
= \int_B \left[\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du) \right] \nu(dy),
\end{align}
which tells us both that $\mu_Y \ll \nu$ and gives the specific form for the
Radon-Nikodym derivative,
\begin{align}
\frac{d\mu_Y}{d\nu}(y) &= \int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du).
\end{align}
We can thus plug in this explicit expression to conclude
\begin{align}
\int_B \int_A \mu^y(du) \mu_Y(dy)
&= \int_B \mu^y(A) \frac{d\mu_Y}{d\nu}(y) \nu(dy)
= \int_B \mu^y(A) \left[\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du) \right] \nu(dy).
\end{align}

#### Finishing the proof
We're just about there. All we've done is re-write the joint probability
$\mathbb{P}(U \in A, Y \in B)$ in two ways, one using the conditional
$U|Y$ and the other using $Y|U$. We manipulated the integrals a bit to make them
comparable, and have ended up with the equality
\begin{align}
\int_B \left[\int_A \frac{dP_u}{d\nu}(y) \mu_U(du) \right] \nu(dy)
&= \int_B \mu^y(A) \left[\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du) \right] \nu(dy).
\end{align}
Subtracting one side from the other gives
\begin{align}
\mu^y(A) \left[\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du) \right]
&= \int_A \frac{dP_u}{d\nu}(y) \mu_U(du), && \nu-a.s.
\end{align}
We now divide both sides by the integral in brackets. Recall that this integral
is finite and non-zero $\mu_U$-a.s., so we conclude that
\begin{align}
\mu^y(A)
&= \int_A \left[\frac{\frac{dP_u}{d\nu}(y)}{\int_{\mathcal{U}} \frac{dP_u}{d\nu}(y) \mu_U(du)}\right] \mu_U(du)
\end{align}
holds $\mu_U$-a.s. in $u$ and $\nu$-a.s. in $y$ (following from the fact that the
division is well-defined on a set of $\mu_U$-probability one, as will be confirmed
below). This expression verifies that
$\mu^y \ll \mu_U$ and verifies that the Radon-Nikodym derivative
$\frac{d\mu^y}{d\mu_U}(u)$ is given by the term in brackets.

#### Verifying that the normalizing constant is well-defined  

{% endkatexmm %}
