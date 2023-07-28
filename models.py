import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import pandas as pd
from matplotlib import pyplot as plt
import random

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def pyro_model(winner_id, loser_id, loser_score_obs=None):
    # priors
    mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
    sd = pyro.sample("sd", dist.LogNormal(0.0, 1.0))

    nt = len(np.unique(winner_id))

    with pyro.plate("plate_players", nt):
        strength = pyro.sample("strength", dist.Normal(mu, sd))
    epsilon=0.01
    p=(strength[loser_id]+epsilon)/(strength[loser_id]+strength[winner_id]+2*epsilon)

    with pyro.plate("results", len(winner_id)):
        pyro.sample("s1", dist.NegativeBinomial(11,p), obs=loser_score_obs)