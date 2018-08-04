import math
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
matplotlib.use("TkAGG")

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
selections = [0] * d
sum_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if selections[i] > 0:
            average_reward = sum_rewards[i] / selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_rewards[ad] += reward
    total_reward += reward

plot.hist(ads_selected)
plot.title("Histogram of ads selections")
plot.xlabel("Ads")
plot.ylabel("Number of selections")
plot.show()
