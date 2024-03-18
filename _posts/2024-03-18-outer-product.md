---
title: Generalizing the Outer Product to Hilbert Space
subtitle: I briefly discuss how the outer product, familiar in Euclidean space, can be viewed as a linear operator which can readily be generalized to Hilbert space.
layout: default
date: 2024-03-18
keywords: Functional-Analysis, Linear-Algebra
published: true
---

## Finite Dimensional Euclidean Space
{% katexmm %}
In Euclidean space, where $x, y \in \mathbb{R}^n$, the outer product is defined
via the matrix multiplication $xy^\top \in \mathbb{R}^{n \times n}$. Since the
resulting $n \times n$ matrix can be viewed as a linear map,
it is natural to consider the outer product as an operation which accepts two vectors
and uses them to construct a linear map $\mathcal{L}(\mathbb{R}^n, \mathbb{R}^n)$
(throughout this post, I use the notation $\mathcal{L}(H_1, H_2)$ to denote the
space of linear maps from a Hilbert space $H_1$ to another Hilbert space $H_2$).
Given this viewpoint, let us denote this operator by
$$
\otimes(\cdot, \cdot): \mathbb{R}^n \times \mathbb{R}^n \to \mathcal{L}(\mathbb{R}^n, \mathbb{R}^n),
$$
though we will favor the binary operator notation
$$
x \otimes y = \otimes(x, y), \qquad x, y \in \mathbb{R}^n.
$$
Letting $x, y, z \in \mathbb{R}^n$, the defining relation of this operator
can be uncovered by observing
$$
(x \otimes y)(z) = (xy^\top) z = x(y^\top z) = x \langle y, z \rangle,
$$
where associativity allowed us to re-write things in terms on an inner, rather
than outer, product. While the definition $x \otimes y = xy^\top$ may not appear
amenable to generalization in the case where $x$ and $y$ are infinite-dimensional
vectors, the defining relation
\begin{align}
(x \otimes y)(z) &= \langle y, z \rangle x \tag{1}
\end{align}
only requires an inner product, and hence is readily generalizable.
{% endkatexmm %}

## Generalizing to Hilbert Space
{% katexmm %}
We now consider a Hilbert space equipped with inner product $\langle \cdot, \cdot \rangle.$
Given vectors $x, y \in H$, we define the operation $x \otimes y: H \to H$ by

\begin{align}
(x \otimes y)(z) &= \langle y, z \rangle x, \qquad z \in H
\end{align}

thus generalizing (1). Given the constructive definition, this operator clearly
exists and inherits linearity (in $z$) from the inner product; i.e.,
$x \otimes y \in \mathcal{L}(H, H)$.

The operator $x \otimes y$ is not unique in the sense that different choices
of $x$ and $y$ can yield the same operator in $\mathcal{L}(H, H)$; e.g.,
for $\alpha \in \mathbb{R}$,
$$
(\alpha x \otimes \alpha^{-1} y)(z) = \langle \alpha^{-1} y, z \rangle (\alpha x)
= \langle y, z \rangle x = (x \otimes y)(z).
$$

I'm not going to get into the applications of this operator here, but I will
briefly note that (just like the familiar outer product) this operation is
closely related to projection. If we set $x=y$ then
$$
(x \otimes x)(z) = \langle x, z \rangle x,
$$
which is precisely the projection of $z$ onto $x$ (when $x$ has unit norm).
{% endkatexmm %}

## Formulation via the Adjoint
{% katexmm %}
While we have generalized the outer product via the defining relation (1), we
might still wonder if we can make meaning of the expression
$$
x \otimes y = xy^\top \tag{2}
$$
when $x,y \in H$. This provides a direct definition of the operator $x \otimes y$
rather than indirectly defining it through its action on $z \in H$. Our plan
of attack will be to replace the transpose in (2) with the adjoint,
$$
x \otimes y = xy^*, \tag{3}
$$
but we still need to provide a precise interpretation of this statement.

We first recall that the adjoint (in Hilbert space; there is also a more general
notation of adjoint in Banach space) of a linear map
$L \in \mathcal{L}(H_1, H_2)$ is a linear map in the reverse direction, denoted
$L^* \in \mathcal{L}(H_2, H_1)$, satisfying the defining relation
\begin{align}
\langle Lx_1, x_2 \rangle_{H_2} &= \langle x_1, L^* x_2 \rangle_{H_1}, \qquad x_1 \in H_1, x_2 \in H_2. \tag{4}
\end{align}

Note that the adjoint is defined for a linear operator, while in (3) we have written
$y^*$ where $y \in H$ is just a vector. The key here will be associating $x$ and $y$
with suitable linear operators. When working with *inner* products, it is common to
associate a vector $x \in H$ with the linear functional $\ell_x(z) = \langle x, z \rangle$.
However, when working with *outer* products we need to flip things around. Instead
of viewing $x$ as the map $\ell_x \in \mathcal{L}(H, \mathbb{R})$, we will view it
as the map $L_x \in \mathcal{L}(\mathbb{R}, H)$ defined by
\begin{align}
L_x(\alpha) &= \alpha x, \qquad \alpha \in \mathbb{R}.
\end{align}

I claim that the adjoint $L^*_x \in \mathcal{L}(H, \mathbb{R})$ is given by
\begin{align}
L^*_x z = \langle x, z \rangle, \qquad z \in H. \tag{5}
\end{align}
Indeed, by definition (4), the adjoint satisfies

\begin{align}
\langle \alpha, L_x^* z \rangle_{\mathbb{R}} = \langle L_x \alpha, z \rangle_H
\end{align}
The inner product on the lefthand side is simply just multiplication, and
on the righthand side, we can plug in the definition of $L_x$. Making these
modifications, this reduces to
$$
\alpha L^*_x z = \langle \alpha x, z\rangle_H = \alpha \langle x, z\rangle_H.
$$
Since $\alpha$ is arbitrary, This proves that $L^*_x z = \langle x, z\rangle_H$.

With this result in hand, I next claim that
\begin{align}
x \otimes y = L_x L_y^*. \tag{6}
\end{align}

Indeed, we have
$$
(L_x L_y^*)z = L_x(L_y^* z) = L_x \langle y, z \rangle = \langle y, z \rangle x = (x \otimes y)(z),
$$
where the middle inequality uses (5). This verifies the claim and shows that we can
write (3) with the understanding that the expression is to be formally interpreted
as (6).
{% endkatexmm %}

## Summary
{% katexmm %}
In this post, we generalized the outer product $x \otimes y = xy^\top$ on
$\mathbb{R}^n$ to Hilbert space in two different, but equivalent, ways. Both
approaches rely on viewing $x \otimes y$ as a linear operator living in
$\mathcal{L}(H, H)$. The first approach indirectly defines $x \otimes y$ through
its action on other vectors $z$ and resulted in the defining relation (1). The
second directly generalizes $xy^\top$ from the finite-dimensional case by replacing
the transpose with the adjoint, and viewing the expression $xy^*$ formally, with
the underlying rigorous interpretation given by (6).
{% endkatexmm %}
