---
title: Some Handy LogSumExp Tricks
subtitle: Derivation of some variations of the LogSumExp trick for numerically stable averaging.
layout: default
date: 2024-08-29
keywords:
published: false
---

{% katexmm %}
## The Basic LogSumExp Trick
$x := {x_1, \dots, x_n}$. Consider that the sum might also be weighted be some
weights $w := \{w_1, \dots, w_n\}$. We will account for the fact that these
weights may only be available on the log scale $\ell_i := \log w_i$. For example,
this is often the case when working with very small probabilities.

## Accounting for Negative Numbers
Now suppose $x$ contains a mix of positive and non-positive values. The sum
$x_1 + \dots x_n$ is of course well-defined, by the LogSumExp trick will fail
since $\log(x_i)$ is undefined when $x_i \leq 0$. A slight adjustment of the
method can deal with this case, subject to a requirement that we will note below.
Let $\text{sgn}(x_i)$ denote the sign of
$x_i$ (plus or minus one). Also define,
$$
M := \text{max}_{1 \leq i \leq n} \left\{\text{sgn}(x_i) \log\lvert x_i\rvert \right\},
$$
which is the maximum of the *positive* elements in $x$.
We can then proceed as follows:
\begin{align}
\log\left[\sum_{i=1}^{n} x_i \right]
&= \log\left[\sum_{i=1}^{n} \text{sgn}(x_i)\lvert x_i \rvert \right] \newline
&= \log\left[\sum_{i=1}^{n} \text{sgn}(x_i)\exp\left(\log \lvert x_i\rvert\right) \right] \newline
&= M + \log\left[\sum_{i=1}^{n} \text{sgn}(x_i)\exp\left(\log \lvert x_i\rvert - M\right) \right]
\end{align}
This looks good, so long as
$$
\sum_{i=1}^{n} \text{sgn}(x_i)\exp\left(\log \lvert x_i\rvert - M\right) > 0 \tag{3}
$$
so that the log in the final expression is well-defined. If we find that this quantity
is not positive, we could flip the sign of all of the values at the beginning of
this procedure, re-defining $x := -x$. The result would then be the log of the
sum of the negated values. If the scale of the result allows for it to be exponentiated,
one could then exponentiate and flip the sign back. A neater way to accomplish this is
to compute
$$
M + \log\big\lvert\sum_{i=1}^{n} \text{sgn}(x_i)\exp\left(\log \lvert x_i\rvert - M\right) \tag{4} \big\rvert.
$$
This gives the desired log-sum when (3) is positive. When (3) is negative, it returns
the log-sum of the negated values. The quantity inside the large absolute value
signs in (4) can be returned so that the user knows whether they computed the negated
version or not.
{% endkatexmm %}

## LogDiffExp

## Log Exp Plus 1 Trick I use


## References:
- https://discourse.mc-stan.org/t/how-to-use-log-sum-exp-function-to-deal-with-a-negative-value/24809/6
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
