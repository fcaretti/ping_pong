import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import pandas as pd
from matplotlib import pyplot as plt
import random

from scipy.stats import nbinom,norm,truncnorm

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def pyro_model(winner_id, loser_id, loser_score_obs=None):
    # priors
    #mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
    #sd = pyro.sample("sd", dist.LogNormal(0.0, 1.0))
    mu=0.5
    sd=1.
    nt = len(torch.unique(torch.cat((winner_id,loser_id))))

    with pyro.plate("plate_players", nt):
        strength = pyro.sample("strength", dist.Normal(mu, sd))
    epsilon=0.01
    p=(strength[loser_id]+epsilon)/(strength[loser_id]+strength[winner_id]+2*epsilon)

    with pyro.plate("results", len(winner_id)):
        pyro.sample("s1", dist.NegativeBinomial(11,p), obs=loser_score_obs)


epsilon=1e-12
def unnorm_product_function(l1,lambdas_wins, lambdas_losses, scores, std=0.3, epsilon=1e-12):
    if l1<0 or l1>1:
        return 0
    else:
        #p=np.concatenate(((l1+epsilon)/(lambdas_wins+l1+epsilon),(lambdas_losses+epsilon)/(lambdas_losses+l1+epsilon)))
        p=np.concatenate(((l1+epsilon)/(lambdas_losses+l1+epsilon),(lambdas_wins+epsilon)/(lambdas_wins+l1+epsilon)))
        func=np.prod(nbinom.pmf(11,scores,p))*norm.pdf(l1,0.5,std)
        return func


def slice_sampler(x_init=0.5,custom_function=unnorm_product_function, n_samples=10):
    x = x_init
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        # Draw a vertical line
        y = np.random.uniform(0, custom_function(x))
        # Create a horizontal “slice” (i.e., an interval)
        x_left = x - 0.1
        while y < custom_function(x_left):
            x_left -= 0.1
        x_right = x + 0.1
        while y < custom_function(x_right):
            x_right += 0.1
        # Draw new sample
        if x_left<0:
            x_left=0.01
        if x_right>1:
            x_right=0.99
        #print(x_left,x_right)
        while True:
            x_new = np.random.uniform(x_left, x_right)
            if y < custom_function(x_new):
                break
            elif x_new > x:
                x_right = x_new
            elif x_new < x:
                x_left = x_new
            else:
                raise Exception("Slice sampler shrank to zero!")
        x = x_new
        samples[i] = x
    return samples[-1]

def init_strengths(n,std):
    mean = 0.5
    # replace with your value
    lower, upper = 0, 1
    # Because truncnorm takes its bounds in the standard Gaussian space, 
    # you should convert your bounds
    lower_std, upper_std = (lower - mean) / std, (upper - mean) / std
    # Create the truncated Gaussian distribution
    truncated_gaussian = truncnorm(lower_std, upper_std, loc=mean, scale=std)
    # Sample from the distribution
    sample = truncated_gaussian.rvs(n)
    return sample

def simulate_match(p):
    count_heads = np.random.binomial(11, p)
    count_tails = 11-count_heads
    while count_heads < 11 and count_tails < 11:
        # draw a sample from a Bernoulli distribution
        sample = np.random.binomial(1, p)
        if sample == 1:
            count_heads += 1
        else:
            count_tails += 1
    if count_heads ==11:
        return 1
    else:
        return 0

class gibbs_model:
    def __init__(self, winner_ids, loser_ids, loser_scores, prior='gaussian', prior_std=0.2,names=None):
        #find the number of players
        self.n_players=len(np.unique(np.concatenate((winner_ids,loser_ids))))
        #initialize the strengths; you can also put a wider prior
        self.strengths=init_strengths(self.n_players,prior_std)
        self.winner_ids=winner_ids
        self.loser_ids=loser_ids
        self.loser_scores=loser_scores
        self.prior_std=prior_std
        self.names=names
        self.posterior_samples=None
        self.prior=prior
        
    def posterior_sampling(self,n_iterations=100, warmup=20, verbose=True):
        strengths_time=[]
        ids = list(range(self.n_players))
        thermalized=False
        for i in range(n_iterations):
            if verbose:
                print("\r" + "Progress: [" + "#" * i + " " * (n_iterations - i) + f"] {int(100 * i / n_iterations)}%", end="")
            if thermalized==False:
                if i==warmup:
                    thermalized=True
            random.shuffle(ids)
            for current_id in ids:
                #print(current_id)
                l1=self.strengths[current_id]
                #print(l1)
                #print(np.where(self.winner_ids == current_id))
                lambdas_wins=self.strengths[self.loser_ids[np.where(self.winner_ids == current_id)]]
                scores=self.loser_scores[np.where(self.winner_ids == current_id)]
                lambdas_losses=self.strengths[self.winner_ids[np.where(self.loser_ids == current_id)]]
                scores=np.concatenate((scores,self.loser_scores[np.where(self.loser_ids == current_id)]))
                custom_function = lambda l1: unnorm_product_function(l1, lambdas_wins, lambdas_losses, scores, self.prior_std, epsilon)
                self.strengths[current_id] = slice_sampler(x_init=self.strengths[current_id], custom_function=custom_function)
                #self.strengths[current_id]=slice_sampler()
            if thermalized:
                strengths_time.append(self.strengths.copy())
        self.posterior_samples=np.array(strengths_time)
        
    def predict_match(self, player_1_id, player_2_id, n_matches=1000, explicit=False):
        player_1_wins=0
        l1=np.random.choice(self.posterior_samples[:,player_1_id], size=n_matches, replace=True, p=None)
        l2=np.random.choice(self.posterior_samples[:,player_2_id], size=n_matches, replace=True, p=None)
        for match in range(n_matches):
            player_1_wins+=simulate_match(l1[match]/(l1[match]+l2[match]))
        ratio=player_1_wins/n_matches
        if explicit:
            print(f'Player 1 has {ratio*100}% probability of winning')
        return ratio
    
    def print_table(self):
        # Print the posterior mean and standard deviation of player strengths
        mean=np.mean(self.posterior_samples,axis=0)
        std=np.std(self.posterior_samples,axis=0)
        for i in range(self.n_players):
            if self.names is not None:
                player = f'Player {i+1}: {self.names[i]}'
            else:
                player = f'Player {i+1}'
            print(player)
            print("Posterior mean of strength:", mean[i])
            print("Posterior std dev of strength:", std[i])
            print()
            
        for i in range(self.n_players):
            for j in range(self.n_players)[i+1:]:
                if self.names is not None:
                    print(f'{self.names[i]} has {int(self.predict_match(i,j)*100)}% of probability of winning against {self.names[j]}')
                else:
                    print(f'Player {i+1} has {int(self.predict_match(i,j)*100)}% of probability of winning against Player {j+1}')
