import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pprint

import torch
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import random
import time
import models
import gdown

print('Finished imports')


#Authorize the API
scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
    ]
file_name = 'pypong.json'
creds = ServiceAccountCredentials.from_json_keyfile_name(file_name,scope)
client = gspread.authorize(creds)

#Get results
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTltw5xQVB_sSBCLiA5nzbiDc1srkInw5TDBWcYy50A-Yhr7zKMTgJUy0aoE1q0uCo5WUJSJSVhR_SY/pub?gid=1190540286&single=true&output=csv"
df = pd.read_csv(url)

#Create arrays
winners =np.array(df['Winner\'s name'])
winners = np.array([str(item) for item in winners])
winners = np.char.capitalize(np.char.replace(winners, " ", ""))
losers = np.array(df['Loser\'s name'])
losers = np.array([str(item) for item in losers])
losers = np.char.capitalize(np.char.replace(losers, " ", ""))

loser_scores=np.array(df['Loser\'s score'])
names = np.unique(np.concatenate((winners, losers)))
ids = list(range(len(names)))
dict_to_id = dict(zip(names,ids))
dict_to_name=dict(zip(ids,names))
winner_ids = np.array([dict_to_id[i] for i in winners])
loser_ids = np.array([dict_to_id[i] for i in losers])

#Create model
g=models.gibbs_model(winner_ids, loser_ids, loser_scores,names=names,prior_std=0.3)
print('Starting sampling')
g.posterior_sampling(n_iterations=500, warmup=100,verbose=False)
print('Finished sampling')

#Get to the Posteriors sheet
spreadsheet_name = "Table Tennis Results"
spreadsheet = client.open(spreadsheet_name)
worksheet = spreadsheet.worksheet("Posteriors")
worksheet.clear()

means=['Posterior Means']+list(np.mean(g.posterior_samples,axis=0))
stds=['Standard Deviations']+list(np.std(g.posterior_samples,axis=0))
names = ['Names']+ list(g.names)

# Write data to columns
worksheet.update('A1', [[name] for name in names])
worksheet.update('B1', [[mean] for mean in means])
worksheet.update('C1', [[std] for std in stds])


