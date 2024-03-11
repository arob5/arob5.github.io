---
title: Deriving the Metropolis-Hastings Update from the Transition Kernel
subtitle: Given only the Metroplis-Hastings transition kernel, I show how to recover the Metropolis-Hastings update rule.
layout: default
date: 2023-12-31
keywords: MCMC, Prob-Theory, Comp-Stats
published: true
---

The Metropolis-Hastings (MH) Markov Chain Monte Carlo (MCMC) method is typically
introduced in the form of a practical algorithm. In a more theoretically-oriented
context, one might prove that the algorithm defines a Markov chain and derive
the associated transition (i.e. probability) kernel. I have found it insightful
to also work through the derivations in the reverse order; given only the
transition kernel, how could one derive the well-known MH update? In other words,
how can you simulate the Markov chain implied by the transition kernel? In this
post, I work through the required derivations.

## Setup and notation
{% katexmm %}
We consider drawing samples from a target probability distribution $\mu$ supported
on a state space $\mathcal{X} \subseteq \mathbb{R}^D$ with Borel sigma algebra
$\mathcal{A}$. Let $\pi: \mathcal{X} \to [0,\infty]$ denote the Lebesgue density
of $\mu$, i.e.
$$
\mu(A) = \int_A \mu(d\mathbf{x}) = \int_A \pi(\mathbf{x}) d\mathbf{x}, \qquad \forall A \in \mathcal{A}.
$$
Let $Q$ denote the proposal kernel for the MH algorithm, and $q$ its Lebesgue density.
For current state $\mathbf{x} \in \mathcal{X}$ and proposal
$\mathbf{y} \sim Q(\mathbf{x}, \cdot)$ we recall the MH acceptance probability
$$
\alpha(\mathbf{x}, \mathbf{y})
=\min\left(1, \frac{\pi(\mathbf{y})q(\mathbf{y},\mathbf{x})}{\pi(\mathbf{x})q(\mathbf{x},\mathbf{y})} \right).
$$
Throughout this post I will let $A \in \mathcal{B}$ denote an arbitrary Borel set.
The transition kernel $P:\mathcal{X} \times \mathcal{B} \to [0,1]$ implied by the MH algorithm is
then given by
\begin{align}
P(\mathbf{x},A)
= \int_A q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y} + \delta_{\mathbf{x}}(A) \int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})[1-\alpha(\mathbf{x},\mathbf{y})] d\mathbf{y} \tag{1}
\end{align}
where $\delta_{\mathbf{x}}(A) := \mathbf{1}(\mathbf{x} \in A)$ denotes the Dirac
measure. The first term in the kernel is the probability of accepting a proposal in
the set $A$, while the second term accounts for the probability of rejection in the
case that the current state $\mathbf{x}$ is already in $A$. I will denote the
overall probability of acceptance by
$$
\overline{a}(\mathbf{x})
:= \int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y}. \tag{2}
$$
{% endkatexmm %}

## Mixture of Kernels
{% katexmm %}
Given the task of writing an algorithm to draw a sample from the distribution
$P(\mathbf{x},\cdot)$ defined in (1), a reasonable place to start is to try
writing $P(\mathbf{x},\cdot)$ as a mixture of kernels from which we already
know how to sample. Let's first try this approach, attempting to write
$$
P(\mathbf{x},A) = wP_1(\mathbf{x},A) + (1-w)P_2(\mathbf{x},A), \qquad w \in [0,1], \tag{3}
$$
with $P_1$, $P_2$ transition kernels we already know how to sample from. If
we are able to do this, then we could easily sample from $P(\mathbf{x},\cdot)$
via the following simple algorithm:
1. Select $P_1$ with probability $w$, else select $P_2$.
2. Sample from the selected kernel.

To this end, let's manipulate (1) to write it in the form of a kernel mixture. We have
\begin{align}
P(\mathbf{x},A)
&= \int_A q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y} + \delta_{\mathbf{x}}(A) \int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})[1-\alpha(\mathbf{x},\mathbf{y})] d\mathbf{y} \newline
&= \overline{a}(\mathbf{x}) \int_A \frac{q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})}{\overline{a}(\mathbf{x})}d\mathbf{y} + \left[1-\overline{a}(\mathbf{x})\right] \delta_{\mathbf{x}}(A) \tag{4}
\end{align}
which is a kernel mixture of the form (3) with  
\begin{align}
w = \overline{a}(\mathbf{x}), \qquad P_1(\mathbf{x},A)=\int_A \frac{q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})}{\overline{a}(\mathbf{x})}d\mathbf{y},
\qquad P_2(\mathbf{x},A)=\delta_{\mathbf{x}}(A).
\end{align}
All we did here was to multiply and divide by the acceptance probability $\overline{a}(\mathbf{x})$
in the first term (under typical assumptions on $Q$ this will be non-zero when $\mathbf{y}$
is in the support of $\pi$) and to rearrange the second term using the fact that
$q(\mathbf{x},\cdot)$ is a probability density; hence,
$$
\int_{\mathcal{X}} q(\mathbf{x},\mathbf{y}) d\mathbf{y} = 1.
$$
Note that the mixture weight $\overline{a}(\mathbf{x})$ is the overall acceptance probability,
and is thus in $[0,1]$ as required. Moreover, $P_2$ is the Dirac measure centered at $\mathbf{x}$
and is thus a valid kernel. To check that $P_1(\mathbf{x},\cdot)$ is a valid probability
measure, we recall the form of $\overline{a}(\mathbf{x})$ from (2) to verify that
\begin{align}
P_1(\mathbf{x},A)
&= \int_{\mathcal{X}} \frac{q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})}{\overline{a}(\mathbf{x})}d\mathbf{y} \newline
&= \frac{\int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y}}{\int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y}} \newline
&= 1.
\end{align}
The other required properties (countable additivity and non-negativity) are similarly
verified. Thus, $P_1(\mathbf{x},\cdot)$ is a probability measure with Lebesgue density
proportional to $q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})$; the overall
acceptance probability $\overline{a}(\mathbf{x})$ is the normalizing constant for this
density.

This representation of the MH kernel as a mixture distribution is conceptually useful,
but it does not directly help us determine a sampling algorithm. Indeed, we cannot
implement the simple mixture sampling algorithm described above since (i.) computing
the mixture weight $\overline{a}(\mathbf{x})$ requires evaluating an intractable
integral, and (ii.) we don't know how to directly sample from the probability distribution
with density proportional to $q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})$.
While this approach seems to be a dead end from a practical point of view,
we should keep in mind that the MH algorithm derived below does sample from the
mixture (4), but does so in a way that avoids having to compute the mixture weight
or to directly sample from $P_2(\mathbf{x},\cdot)$.
{% endkatexmm %}

## Marginalized Mixture of Kernels
{% katexmm %}
In the previous section, we feigned ignorance of the MH algorithm in order to
approach the problem of simulating from (1) as a generic sampling problem.
We found that $P$ can indeed be written as a mixture of kernels, but the problem
of sampling from the resulting mixture was also intractable. To take a step in the
right direction, it is useful to cheat a little bit and recall some of the mechanics
of the MH algorithm. The proposal $\mathbf{y} \sim Q(\mathbf{x},\cdot)$
is accepted with probability $\alpha(\mathbf{x},\mathbf{y})$; if rejected, the
next state is set to the current state $\mathbf{x}$. Thus, it seems that we should
be looking for a mixture of kernels of the form
$$
\alpha(\mathbf{x},\mathbf{y})\delta_{\mathbf{y}}(\cdot) + [1-\alpha(\mathbf{x},\mathbf{y})]\delta_{\mathbf{x}}(\cdot). \tag{5}
$$
Of course, this can't represent the whole picture since the mixture weight in (5)
depends on $\mathbf{y}$ and the proposal kernel $Q$ is completely missing from the
expression. The key insight is that the MH kernel $P(\mathbf{x},\cdot)$ can be
viewed as the expectation of (5) averaged with respect to $Q(\mathbf{x},\cdot)$;
i.e. the mixture (5) is *marginalized* over $Q(\mathbf{x},\cdot)$. To show this,
we return to the original expression (1) for the MH kernel. We have
\begin{align}
P(\mathbf{x},A)
&= \int_A q(\mathbf{x},\mathbf{y})\alpha(\mathbf{x},\mathbf{y})d\mathbf{y} + \delta_{\mathbf{x}}(A) \int_{\mathcal{X}} q(\mathbf{x},\mathbf{y})[1-\alpha(\mathbf{x},\mathbf{y})] d\mathbf{y} \newline
&= \int_{\mathcal{X}} \alpha(\mathbf{x},\mathbf{y}) \mathbf{1}(\mathbf{y} \in A) q(\mathbf{x},\mathbf{y}) d\mathbf{y} +  \int_{\mathcal{X}} [1-\alpha(\mathbf{x},\mathbf{y})] \delta_{\mathbf{x}}(A) q(\mathbf{x},\mathbf{y}) d\mathbf{y} \newline
&= \int_{\mathcal{X}} \alpha(\mathbf{x},\mathbf{y}) \delta_{\mathbf{y}}(A) q(\mathbf{x},\mathbf{y}) d\mathbf{y} +  \int_{\mathcal{X}} [1-\alpha(\mathbf{x},\mathbf{y})] \delta_{\mathbf{x}}(A) q(\mathbf{x},\mathbf{y}) d\mathbf{y} \newline
&= \int_{\mathcal{X}} \left[\alpha(\mathbf{x},\mathbf{y})\delta_{\mathbf{y}}(A) + [1-\alpha(\mathbf{x},\mathbf{y})] \delta_{\mathbf{x}}(A) \right] q(\mathbf{x},\mathbf{y}) d\mathbf{y}, \tag{6}
\end{align}
which is precisely the mixture (5) averaged (with respect to $\mathbf{y}$) over
$q(\mathbf{x},\mathbf{y})$. We now have three different representations of the MH
transition kernel $P$: (1) is the most natural to derive when starting from the MH
algorithm, (4) represents $P$ as a mixture of two distributions, and (6) represents
$P$ as a marginalized mixture of two distributions. It is this final representation which
proves useful for developing a practical simulation algorithm.

All that remains is to recall how to sample from a marginalized distribution. First
note that $\mathbf{x}$ is essentially a fixed parameter in the integral (6); the
averaging is done with respect to $\mathbf{y}$. Now, if we condition on a fixed
 $\mathbf{y}$ as well, then the expression
$$
\alpha(\mathbf{x},\mathbf{y})\delta_{\mathbf{y}}(A) + [1-\alpha(\mathbf{x},\mathbf{y})] \delta_{\mathbf{x}}(A)
$$
is simply a mixture of two distributions, which we know how to sample from. Thus,
a sample can be drawn from $P(\mathbf{x},\cdot)$ via the following algorithm:
1. Sample $\mathbf{y} \sim Q(\mathbf{x}, \cdot)$.
2. Conditional on $\mathbf{y}$, sample from $\alpha(\mathbf{x},\mathbf{y})\delta_{\mathbf{y}} + [1-\alpha(\mathbf{x},\mathbf{y})] \delta_{\mathbf{x}}$.

For the second step, the mixture can be sampled from using the simple algorithm
discussed in the previous section: randomly select one of the two kernels with
probabilities equal to their mixture weights, then return a sample from the
selected kernel. Since sampling from the Dirac measure $\delta_{\mathbf{x}}$ simply
means returning the value $\mathbf{x}$ (and similarly for $\delta_{\mathbf{y}}$)
then this step will simply return $\mathbf{x}$ or $\mathbf{y}$ according to
their respective probabilities $\alpha(\mathbf{x},\mathbf{y})$ and
$1-\alpha(\mathbf{x},\mathbf{y})$. This is precisely the MH accept-reject
mechanism!

It might be helpful to make this more concrete by letting $\mathbf{X}_k$ denote the
value of the MCMC algorithm at iteration $k$ and letting
 $\mathbf{Y}|\mathbf{X}_k \sim Q(\mathbf{X}_k, \cdot)$ be the random variable
 representing the proposal. Then the above mixture corresponds to the probability
$\mathbb{P}\left(\mathbf{X}_{k+1} \in A | \mathbf{X}_k=\mathbf{x}, \mathbf{Y}=\mathbf{y}\right)$
so we can re-write (6) as
$$
P(\mathbf{x},A)
= \int_{\mathcal{X}} \mathbb{P}\left(\mathbf{X}_{k+1} \in A | \mathbf{X}_k=\mathbf{x}, \mathbf{Y}=\mathbf{y}\right) q(\mathbf{x},\mathbf{y}) d\mathbf{y}.
$$
Once we condition on $\mathbf{X}_k$ and $\mathbf{Y}$, the only remaining randomness in
the probability above is coming from the selection of one of the two kernels.

## Conclusion
While all of this might appear to be overcomplicating the very simply MH algorithm,
I have found it a quite worthwhile exercise to contemplate some different perspectives
on the method, as well as to get some practice manipulating expressions involving
probability kernels and thinking through sampling schemes. The MH transition kernel
(1) can easily be derived by thinking through the mechanics of the MH algorithm.
In this post, I showed in (4) how the kernel can be re-written as mixture of two
distributions and in (6) as a marginalized mixture of two distributions. It is
this final expression which provides the basis for a tractable simulation algorithm.  
{% endkatexmm %}
