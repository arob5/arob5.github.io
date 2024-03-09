---
title: Implementations of Gaussian Processes - Python, R, and Julia
subtitle: Following up on the previous post on variations in specifying and fitting Gaussian processes, I show how some of these decisions play out in popular Gaussian process implementations.
layout: default
date: 2024-01-12
keywords: GP, Python, R, Julia, Comp-Stats
published: false
---

# Introduction

## Things to Look Out for As Users
1. Is the *predict* function including the nugget variance or not?
2. How is the covariance function parameterized?
(important if trying to interpret lengthscales, for example)
3. If using a plug-in prior mean estimate, does the package bias-correct the
GP predictive mean or not? (i.e. does it perform the universal kriging
correction).
4. Does the package offer scaling and normalization functionality for the
regression inputs and outputs?


# High-Level Implementation Choices

## Computing the predictive equations
