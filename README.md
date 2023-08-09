# Ping pong meets hierarchical models (hopefully)
The goal of this small project is to build a hierarchical bayesian model to infer players skills and predict results of table tennis matches.
It currently implements a Gibbs sampling method based on a truncated Gaussian prior and a Negative Binomial likelihood. Since the prior is not conjugate to the likelihood, slice sampling is implemented for the update step.
