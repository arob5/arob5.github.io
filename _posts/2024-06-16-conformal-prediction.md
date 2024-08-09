---
title: Conformal Prediction
subtitle: Exploring a popular method for computing prediction sets for black-box models.
layout: default
date: 2024-06-16
keywords: Statistics
published: false
---

## Setting and Assumptions
{% katexmm %}
In order to precisely define *confidence set*, we need to specify what probability
space we're actually working with. The conformal inference setting is quite
generic, assuming only that input-output pairs
$(x,y) \in \mathcal{X} \times \mathcal{Y}$ are sampled from some joint probability
distribution on $\mathcal{X} \times \mathcal{Y}$. We formalize this joint
distribution via a probability measure $\mathbb{P}$ defined on the measurable space
$\left(\mathcal{X} \times \mathcal{Y}, \mathcal{A}\right)$, for some
sigma field $\mathcal{A}$.  


{% endkatexmm %}
