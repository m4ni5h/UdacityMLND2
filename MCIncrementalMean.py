import numpy as np

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + (1.0/(k+1))*(x[k] - mu)
        mean_values.append(mu)
    return mean_values
x = [2, 3, 4, 7, 3, 5, 7, 8, 9, 6, 4]
print(running_mean(x))

#The correct answer is
#[2.0, 2.5, 3.0, 4.0, 3.8, 4.0, 4.428571428571429, 4.875, 5.333333333333333, 5.3999999999999995, 5.2727272727272725]