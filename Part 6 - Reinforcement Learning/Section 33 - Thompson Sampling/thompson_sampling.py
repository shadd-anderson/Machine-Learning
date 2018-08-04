import random

import matplotlib
import matplotlib.pyplot as plot
import pandas as pd
matplotlib.use("TkAGG")

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
num_rewards_1 = [0] * d
num_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(num_rewards_1[i] + 1, num_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        num_rewards_1[ad] += 1
    else:
        num_rewards_0[ad] += 1
    total_reward += reward

plot.hist(ads_selected)
plot.title("Histogram of ads selections")
plot.xlabel("Ads")
plot.ylabel("Number of selections")
plot.show()
