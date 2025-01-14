---
title: The Reversed Cholesky Decomposition
subtitle: Cholesky-like decomposition with upper-triangular matrices.
layout: default
date: 2025-01-14
keywords:
published: true
---

{% katexmm %}
The Cholesky decomposition of a positive definite matrix $C$ is the unique
factorization of the form
$$
C = LL^\top, \tag{1}
$$
such that $L$ is lower triangular with positive entries on its diagonal. We
refer to $L$ as the Cholesky factor of $C$, and denote this relationship by
$L := \text{chol}(C)$.
Intuitively, it seems that the focus on lower triangular matrices is just a
convention, and that we could alternatively consider factorizations of the form
$$
C = UU^\top , \tag{2}
$$
where $U$ is upper triangular with positive diagonal entries. This is no longer
what is commonly called the Cholesky decomposition, but is similar in spirit.
We will refer to this as the *reversed Cholesky decomposition*, or
*rCholesky* for short. We call $U$ the rCholesky factor, and write
$U := \text{rchol}(C)$. In this post, we explore this alternative factorization,
and demonstrate (i.) its close connections to the Cholesky factorization of
$C^{-1}$; and (ii.) its interpretation as the Cholesky factorization under
a reverse ordering of the variable indices. These connections have various
applications, including in high-dimensional covariance estimation; see, e.g.,
{% cite SparseCholeskyVecchia %} for one such example.
{% endkatexmm %}

{% katexmm %}
# Connections to the Inverse Matrix
The first order of business is to ensure that we have the same uniqueness and
existence properties for (2) as we do for (1). The below result shows that
the rCholesky factorization inherits these properties from the Cholesky
factorization.

<blockquote>
  <p><strong>Existence and Uniqueness.</strong> <br>
  If $C$ is positive definite, then there exists a unique decomposition of the
  form $C = UU^\top$ such that $U$ is upper triangular with positive entries
  on the diagonal.
  </p>
</blockquote>

**Proof.** Since $C$ is positive definite, then $C^{-1}$ exists and is itself
positive definite. Thus, there is a unique factorization of the form
$C^{-1} = LL^{\top}$, where $L$ is lower triangular with positive diagonal
entries. Therefore, $L$ is invertible and
$$
C = L^{-\top}L^{-1}. \tag{3}
$$
Setting $U := L^{-\top}$, we obtain $C = UU^\top$. $\qquad \blacksquare$

We see from the above proof that the rCholesky factor of $C$ is closely related
to the Cholesky factor of $C^{-1}$. This corollary is summarized below.

<blockquote>
  <p><strong>Connection to Inverse Matrix.</strong> <br>
  Let $C$ be a positive definite matrix. Then
  \begin{align}
  \text{rchol}(C) &= \text{chol}(C)^{-\top} \tag{4} \newline
  \text{chol}(C) &= \text{rchol}(C)^{-\top}. \tag{5}
  \end{align}
  </p>
</blockquote>

A consequence of (4) and (5) is that we can easily transform between the
Cholesky factorization of a matrix and the rCholesky factorization of its
inverse.
{% endkatexmm %}

# Reverse Ordering
{% katexmm %}

## The Reversal Operator
As we will see, the rCholesky decomposition can be interpreted as a Cholesky
decomposition of a matrix under reverse ordering. By reverse ordering, we
mean that the order of both the rows and the columns of $C$ are reversed.
This notion is more intuitive when viewing $C$ as a $n \times n$ covariance
matrix for some random variables $x_1, \dots, x_n$, such that
$C_{ij} = \text{Cov}[x_i,x_j]$. We thus see that the ordering of the variables
determines the ordering of the matrix. Let $x := (x_1, \dots, x_n)^\top$ be the
vector of variables such that $C = \text{Cov}[x]$. We will denote by
$$
\tilde{x} := (x_n, \dots, x_1)^\top \tag{6}
$$
the reversed vector. The reversal operation $x \mapsto \tilde{x}$ is linear
and can thus be represented by a matrix. In particular, $\tilde{x}$ is given
by
$$
\tilde{x} = Px \tag{7}
$$
where $P$ is the square permutation matrix with ones on the
[anti-diagonal](https://en.wikipedia.org/wiki/Anti-diagonal_matrix); i.e.,
the non-main diagonal going from the lower-left to the upper-right corner.
For example, if $n=3$ then
\begin{align}
P &= \begin{bmatrix}
0 & 0 & 1 \newline 0 & 1 & 0 \newline 1 & 0 & 0
\end{bmatrix}. \tag{8}
\end{align}

We will make use of the following properties of the matrix $P$.

<blockquote>
  <p><strong>Properties of Reversal Operator.</strong> <br>
  The matrix $P$ that reverses the order of a vector satisfies
  $$
  P = P^\top, \qquad P = P^{-1}, \qquad P^2 = P \tag{9}
  $$
  </p>
</blockquote>

The first property is true of any anti-diagonal matrix, and the latter two
simply reflect the fact that applying the reversal operation twice results in
the original vector. With these properties in hand, note that
$$
\tilde{C} := \text{Cov}[\tilde{x}] = \text{Cov}[Px]
= P\text{Cov}[x]P^\top = PCP^\top = PCP. \tag{10}
$$

In words, this says that the covariance matrix of the reversed vector is given
by $PCP$, where $C$ is the covariance of the original vector. If you prefer
to avoid probabilistic language, then $PCP$ is simply the result of reversing
the order of the columns and rows of $C$. Reversing $C$ induces the same
operation on its inverse, since
$$
(\tilde{C})^{-1} := (PCP)^{-1} = P^{-1}C^{-1}P = PC^{-1}P, \tag{11}
$$
where we have used (9).

## Cholesky Factorization of Reversed Matrix
We now derive the form of the Cholesky and rCholesky decompositions of the
reversed matrix $\tilde{C}$. Notation becomes a bit confusing here, so we
separate the two results for clarity.

<blockquote>
  <p><strong>Cholesky under reverse ordering.</strong> <br>
  Let $C = UU^\top$ be the rCholesky decomposition of $C$. Then the Cholesky
  decomposition of $\tilde{C} = PCP$ is given by
  $$
  \tilde{C} = (PUP)(PUP)^\top. \tag{12}
  $$
  This can be equivalently be written as
  $$
  \text{chol}(\tilde{C}) = \text{chol}(PCP) = PUP = P\text{rchol}(C)P. \tag{13}
  $$
  In words, this says that the Cholesky factor of $\tilde{C}$ is given by
  reversing the rCholesky factor of $C$.
  </p>
</blockquote>

**Proof.** Using the fact that $P^2 = I$ we have
$$
\tilde{C} = PCP = P(UU^\top)P = PUP^2 U^\top P = (PUP)(PUP)^\top.
$$
The result is now immediate upon noticing that $PUP$ is lower triangular
and has positive diagonal entries. $\qquad \blacksquare$

<blockquote>
  <p><strong>rCholesky under reverse ordering.</strong> <br>
  Let $C = LL^\top$ be the Cholesky decomposition of $C$. Then the rCholesky
  decomposition of $\tilde{C} = PCP$ is given by
  $$
  \tilde{C} = (PLP)(PLP)^\top. \tag{14}
  $$
  This can be equivalently be written as
  $$
  \text{rchol}(\tilde{C}) = \text{rchol}(PCP) = PLP = P\text{chol}(C)P. \tag{15}
  $$
  In words, this says that the rCholesky factor of $\tilde{C}$ is given by
  reversing the Cholesky factor of $C$.
  </p>
</blockquote>

**Proof.** Using the fact that $P^2 = I$ we have
$$
\tilde{C} = PCP = P(LL^\top)P = PLP^2 L^\top P = (PLP)(PLP)^\top.
$$
The result is now immediate upon noticing that $PLP$ is upper triangular
and has positive diagonal entries. $\qquad \blacksquare$

These results tell us that we can use a Cholesky factorization of $C$
to immediately compute the rCholesky factorization of the reversed matrix
$\tilde{C}$. This statement is the same with the roles of Cholesky and
rCholesky swapped. In (13) and (15) we can left and right multiply by $P$ to
obtain the equivalent expressions
\begin{align}
\text{chol}(C) &= P\text{rchol}(PCP)P \tag{16} \newline
\text{rchol}(C) &= P\text{chol}(PCP)P. \tag{17} \newline
\end{align}
See remark 1 in {% cite SparseCholeskyVecchia %} for an example of an expression
of this form, though the authors are decomposing $C^{-1}$ in place of $C$.

# Other Resources
- Relevant StackExchange posts: [here](https://math.stackexchange.com/questions/2039477/cholesky-decompostion-upper-triangular-or-lower-triangular), [here](https://math.stackexchange.com/questions/712993/cholesky-decomposition-of-the-inverse-of-a-matrix), and [here](https://mathoverflow.net/questions/230808/computing-the-inverse-of-a-cholesky-decomposition)
- Permutation and Grouping Methods for Sharpening Gaussian Process Approximations (Guiness, 2018)
- Iterative methods for sparse linear systems (Saad, 2003)

{% endkatexmm %}
