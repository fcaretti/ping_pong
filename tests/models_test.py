import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import pandas as pd
from matplotlib import pyplot as plt
import random
import os
import sys
import gdown

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models

print(torch.__version__)
print(np.__version__)
print('Passed import tests')

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTltw5xQVB_sSBCLiA5nzbiDc1srkInw5TDBWcYy50A-Yhr7zKMTgJUy0aoE1q0uCo5WUJSJSVhR_SY/pub?gid=0&single=true&output=csv"
df = pd.read_csv(url)
winners=np.array(df['Winner'])
losers=df['Loser']
loser_scores=df['Points']
names = pd.concat([df['Winner'], df['Loser']]).unique()
ids = list(range(len(names)))
dict_to_id = dict(zip(names,ids))
dict_to_name=dict(zip(ids,names))
winner_ids = [dict_to_id[i] for i in winners]
loser_ids = [dict_to_id[i] for i in losers]

print('Passed reading scores')

winner_ids=torch.Tensor(winner_ids).int()
loser_ids=torch.Tensor(loser_ids).int()
loser_scores=torch.Tensor(loser_scores).int()

# Run inference using NUTS (No-U-Turn Sampler) in MCMC
nuts_kernel = NUTS(models.pyro_model)
mcmc_run = MCMC(nuts_kernel, num_samples=2, warmup_steps=1)
mcmc_run.run(winner_ids, loser_ids, loser_scores)

# Get posterior samples
posterior_samples = mcmc_run.get_samples()

print('Pyro model running')