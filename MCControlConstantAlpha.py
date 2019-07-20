import numpy as np

# This is the sequence (corresponding to successively sampled returns). 
# Feel free to change it!
x = np.hstack((np.ones(10), 10*np.ones(10)))

# These are the different step sizes alpha that we will test.  
# Feel free to change it!
alpha_values = np.arange(0,.3,.01)+.01

#########################################################
# Please do not change any of the code below this line. #
#########################################################

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + (1.0/(k+1))*(x[k] - mu)
        mean_values.append(mu)
    return mean_values
    
def forgetful_mean(x, alpha):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + alpha*(x[k] - mu)
        mean_values.append(mu)
    return mean_values

def print_results():
    """
    prints the mean of the sequence "x" (as calculated by the
    running_mean function), along with analogous results for each value of alpha 
    in "alpha_values" (as calculated by the forgetful_mean function).
    """
    print('The running_mean function returns:', running_mean(x)[-1])
    print('The forgetful_mean function returns:')
    for alpha in alpha_values:
        print(np.round(forgetful_mean(x, alpha)[-1],4), \
        '(alpha={})'.format(np.round(alpha,2)))

print_results()

# The running_mean function returns: 5.5
# The forgetful_mean function returns:
# 1.0427 (alpha=0.01)
# 1.9787 (alpha=0.02)
# 2.8194 (alpha=0.03)
# 3.5745 (alpha=0.04)
# 4.2529 (alpha=0.05)
# 4.8624 (alpha=0.06)
# 5.4099 (alpha=0.07)
# 5.9018 (alpha=0.08)
# 6.3436 (alpha=0.09)
# 6.7403 (alpha=0.1)
# 7.0964 (alpha=0.11)
# 7.4159 (alpha=0.12)
# 7.7025 (alpha=0.13)
# 7.9593 (alpha=0.14)
# 8.1894 (alpha=0.15)
# 8.3953 (alpha=0.16)
# 8.5795 (alpha=0.17)
# 8.7441 (alpha=0.18)
# 8.891 (alpha=0.19)
# 9.0221 (alpha=0.2)
# 9.1389 (alpha=0.21)
# 9.2428 (alpha=0.22)
# 9.3352 (alpha=0.23)
# 9.4173 (alpha=0.24)
# 9.49 (alpha=0.25)
# 9.5544 (alpha=0.26)
# 9.6114 (alpha=0.27)
# 9.6616 (alpha=0.28)
# 9.706 (alpha=0.29)
# 9.745 (alpha=0.3)