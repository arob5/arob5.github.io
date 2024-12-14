---
title: Precision Matrices, Partial Correlations, and Conditional Independence
subtitle: Understanding the inverse covariance matrix in general, and in the Gaussian special case.
layout: default
date: 2024-12-09
keywords:
published: false
---

# Setup and Notation
{% katexmm %}
Throughout this post we will consider the set of random variables
$x := \{x_1, x_2, \dots, x_n\}$ taking values in $\R$.
We will generically write $p(x_1, x_2, \dots, x_n)$
to denote the joint distribution of these variables, and abuse notation by
using the same symbol $p(\cdot)$ to indicate marginal and conditional
distributions. For example $p(x_i, x_j)$ is the marginal distribution for
$(x_i, x_j)$ and $p(x_i, x_j|x_k)$ is the distribution of $(x_i, x_j)$,
conditional on $x_k$. We will find it convenient to introduce some shorthand
to avoid listing these random variables all the time. For and index set
$A \subseteq \{1, 2, \dots, n\}$ we define $x_A := \{x_i\}_{i \in A}$.
We also write
$\tilde{x}_A := x \setminus x_A := \{x_i\}_{i \notin A}$ to indicate the subset
of all variables in $x$ excluding $x_A$. Finally, we introduce the shorthand
$x_{ij} := \{x_i, x_j\}$ and $x_{i:j} := \{x_i, \dots, x_j\}$ (for $i \leq j$).
Finally, while we have introduced the above notation using sets, we will
utilize the same notation for vectors when relevant. For example, when relevant,
$x_A$ will denote the column vector $[x_i]_{i \in A}$ with entries ordered
according to the specified order of the index set $A$.
{% endkatexmm %}

# Geometry induced by the precision

# Partial correlations

# Gaussian special case

# Graphical models

1. Estimation and Model Identification for Continuous Spatial Processes (Vecchia)
2. Graphical Models in Applied Mathematical Multivariate Statistics
3. Graphical Models (Lauritzen)
4. https://stats.stackexchange.com/questions/10795/how-to-interpret-an-inverse-covariance-or-precision-matrix
5. http://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
6. https://stats.stackexchange.com/questions/140080/why-does-inversion-of-a-covariance-matrix-yield-partial-correlations-between-ran
