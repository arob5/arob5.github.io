---
title: A look under the hood at Gaussian process software
subtitle: Exploring and explaining the source code of some popular GP packages.
layout: default
date: 2025-01-05
keywords: GP
published: true
---

There are an overwhelming number of Gaussian process (GP) packages implemented
across various programming languages. It can be a daunting task to weigh
the pros and cons of different packages in order to select one for a particular
project. Even after reading the documentation and walking through tutorials,
I often find that the limitations of certain GP implementations only become
fully apparent after playing around with the software for a while. I plan
to use this post as a place to store my thoughts on various packages I've tried
out, and walk through some source code to better understand GP software from
the ground up.

My focus here is mostly on general-purpose GP software; that is, implementations
that provide a wide array of features for experimenting with different GP models.
There are many speciality packages out there targeting specific use cases
(scalable/approximate GPs, GPs for time series or spatial settings, etc.), but
these are beyond the scope of this post.

I don't intend this to be comprehensive in any way. There is
already a nice [list](https://en.wikipedia.org/wiki/Comparison_of_Gaussian_process_software)
of GP software on Wikipedia. I also recommend [this](https://danmackinlay.name/notebook/gp_implementation)
post on GP software from Dan MacKinlay.

# Desirable Features in GP Software

## The Essentials
- Closed-form inference for latent GP with Gaussian likelihood.
- Numerically stable implementation of GP predictive mean and (co)variance.
- Ability to estimate hyperparameters via numerical optimization algorithms
(via maximum marginal likelihood)
- Ability to update GP model (i.e., condition on more data) without having
to re-fit the whole model.
- Scaling/normalization

## Nice to Have
- Option for using GPs as a building block of a larger model/system.
- Inference for non-Gaussian likelihoods.
- Ability to place priors on hyperparameters (rather than just interval bound constraints)
- Option to optimize hyperparameters or sample from full Bayesian posterior
- Lot's of kernels, as well as ability to define your own and perform "kernel algebra"
- Standard mean functions (e.g., polynomial), and ability to define your own.
- Ability to leverage closed-form computations when possible
- Easy to manually fix certain parameters that may be known
- Easy to integrate with external code (potential downside of PPLs)
- Approximate/scalable GPs
- Automatic differentiation
- Sequential design
- Multi-output GPs
- Nice default, out-of-the-box behavior (e.g., parameter bounds)

## Convenience functions
- Model evaluation for the GP prior
- Model evaluation for GP posterior
- Cross validation methods for model checking
- Plotting helper functions
  - Specialty functions to make the classic GP plots in one and two dimensions.
  - Emphasis on plots for higher-dimensional input spaces.

# List of Gaussian Process Packages
To start, here is a long list of GP software that I am away of, and some brief
notes on the functionality they provide. In general, I find that Python offers
the most options as far as flexible GP software, which is largely the
result of efforts by the machine learning community. Julia is newer and hence
the software tends to be less developed, but it does have some GP toolboxes.
R lacks the kind of general-purpose GP toolbox that is available in Python or
Julia. An R user requiring the ability to fit flexible GP models is probably
best off looking to Stan, though there are tradeoffs in opting for software
that is not specifically designed for GPs.

## Python
- GPy
- PyMC3
- GPFlow
  - Trieste
  - GPFlux
- scikit-learn

## R
- kergp
- hetGP
- laGP
- mlegp
- DiceKriging

## Julia
- [GaussianRandomFields](https://pieterjanrobbe.github.io/GaussianRandomFields.jl/stable/API/)

## Cross Language
- Stan
- [celerite](https://celerite.readthedocs.io/en/stable/) and [celerite2](https://celerite2.readthedocs.io/en/latest/): (Python, C++, and Julia)

# Very Quick GP Review

# kergp (R)
Disregarding cross-platform PPLs (e.g., Stan), `kergp` offers the closest thing
to a flexible GP toolbox in R that I am aware of. Unfortunately, it does
not seem to be very actively maintained; its documentation mentions some
updates that seem to be stuck in limbo.

### Mean Functions
`kergp` allows for specification of a linear model for the mean function
using `R` formulas, the same way you would do for `lm()`. This covers
many of the common mean functions used in practice, including constant,
linear, and polynomial. The covariates in the mean function are allowed to
differ from the inputs to the kernel, which is a common scenario in certain
settings (e.g., geostatistics).

### Kernels
`kergp` defines a kernel via the `CovMan` class, which is short for
"manual covariance function". The package offers some pre-defined common
kernels (e.g., Gaussian and Matérn), as well as some more specialized
options (including kernels for qualitative inputs). Users can define their
own kernels via the `CovMan` class, and the package also provides the ability
to combine kernels via the typical algebraic operations. The `CovMan` class
stores a method for computing (cross) covariance matrices between sets of
inputs, as well as the gradient with respect to the kernel hyperparameters.

### Parameter Estimation
`kergp` estimates hyperparameters via maximum marginal likelihood. The only
constraints supported are bound constraints on the kernel parameters and noise
variance. This is a bit of a bummer, as it would be nice to be able to
regularize the optimization via more flexible priors. No constraints are
allowed on the coefficients of the mean function. The reason for this is
due to the fact that the optimization procedure leverages the fact that,
conditional on the other hyperparameters, the mean coefficients can be
optimized in closed-form. Indeed, the optimum is simply a generalized
least squares estimator, a result I derive
[here](https://arob5.github.io/blog/2024/01/11/GP-specifications/). If you're
fine with the mean coefficients being unregularized, then its nice that
`kergp` simplifies the optimization by leveraging this closed-form computation.
However, it also restrictive in that it doesn't allow the definition of
priors on the coefficients (including bound constraints). The objective
function being numerically optimized under the hood is thus the
*concentrated* marginal likelihood; that is, the marginal likelihood
as a function of the kernel parameters and noise variance, with the closed-form
estimate for the mean coefficients (as a function of the other parameters),
plugged in. One other thing that is a bit annoying is that there is currently
no ability to fix the kernel parameters or noise variance at some desired
value. The user can only control whether the kernel parameters, mean parameters,
or both are optimized vs. fixed. Thus, all kernel parameters are either fixed or
not. If a parameter is included in the `kernParNames` attribute of
`covMan`, and you initialize your GP via `gp(..., estim = TRUE)`,
then it is going to be optimized. I found this a bit frustrating when
manually creating a quadratic kernel, which is of the form
$$
k(x,z) = \left(\langle x-a, z-a\rangle + c \right)^2.
$$
Sometimes I want to fix one of $a$ or $c$ instead of estimating them both,
but as far as I can tell the only way to do this is when actually
defining the class. The documentation actually notes the future addition
of a `parFixed` argument to `gp()` that would address this complaint, but
it has yet to be added.

If you instead run `gp(..., estim = FALSE)` then the kernel parameters and
noise variance are fixed (you must supply their values), and only the
generalized least squares estimator for the mean coefficients is computed - so
no numerical optimization required in this case. Another oddity is that
it appears there is no easy way to manually fix the noise variance when
`estim = TRUE`. It seems to be lumped in with the kernel parameters as far as
whether they are all estimated together or all fixed. I ran into this issue
when I wanted to fix the noise to a small "jitter" for numerical stability.
To get around this, I've used a little hack in the past: setting the lower
and upper bounds for the noise variance to the desired noise variance
value. I would hope the underlying optimization routine has no trouble with
this, but no promises that this is actually a good idea.

TODO: note two different optimization routines.
TODO: note that there is no update method.

### Prediction


{% highlight bash %}
cd ~
{% endhighlight %}

<pre>
  <code class="ruby">
    puts "hello"
  </code>
</pre>

```
Test
```

# hetGP (R)
`hetGP` is another `R` package I've worked with. The name is short for
"heteroscedastic", implying that this package has support for GP models
where the noise term is of the form $\epsilon(x) \sim \mathcal{N}(0, \sigma^2(x))$;
that is, the noise can vary across the input space. Fitting models of this form
thus typically requires having replicate observations of $y(x)$ at the same $x$
value. Since this post is focused on more standard GP models, we instead focus on
reviewing `hetGP`'s support for traditional homoscedastic noise models.

### Mean Functions

### Kernels

### Parameter Estimation

### Prediction

# PyMC (Python)
