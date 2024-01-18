import math
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo valuation of a European call option
np.random.seed(42)      # fix random seed for reproducibility

# parameters
S0 = 50     # initial index level
K = 55      # strike price
r = 0.05    # risk-less short rate
sigma = 0.4     # volatility
T = 1       # time-to-maturity
t = 30      # number of time steps by which we divide the interval [0, T]
dt = T / t      # length of time interval
M = 1000     # number of paths

# simulate M random walks with t time steps

S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt)* np.random.standard_normal((t + 1, M)), axis=0))

# calculate payoff at maturity
P_call = sum(np.maximum(S[-1] - K, 0)) / M

print("Value of the European call option: {:0.2f} ZAR.\n".format(P_call))

# plot the simulated paths
num_paaths_to_plot = 100
plt.figure(figsize= (12,8))
plt.plot(S[:, :num_paaths_to_plot])
plt.grid(True)
plt.xlabel('time steps')
plt.ylabel('index level')
plt.title('Simulated random walks')
plt.show()

# let's investigate the frequency of the simulated index vals at the end of
# the simulation period

plt.figure(figsize=(10,5))
plt.hist(S[-1], bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title('Histogram of simulated end index level')
plt.show()

# all simulated index vals at the end of the simulation period
plt.figure(figsize=(10,5))
plt.hist(np.maximum(S[-1] - K, 0), bins=50)
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.title('Histogram of simulated end option inner value')
plt.show()

# THE QUANTUM APPROACH
