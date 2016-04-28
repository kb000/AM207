import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from cleaning import rides

rides = rides[rides['start_weekday'] < 5]

def examine_weights(times, ws):
    plt.scatter(times, ws[0], c='red')
    plt.scatter(times, ws[1], c='blue')
    plt.scatter(times, ws[2], c='green')
    plt.xlim((0, 86400))
    plt.ylim((0, 1))
    plt.show()


# Find weights for each c_i given the data and current mixture parameters.
def e_step(data, mu, sigma, lam):
    """
    Find weights for each c_i given the data and current mixture parameters
    :data: pandas dataframe with 'start_seconds'
    :mu: array of 3 of mixture time means.
    :sigma: array of 3 of mixture time deviations.
    :return: weights [c_0, c_1, c_2] for each item in data.
    """
    td = np.array(data['start_seconds']) # Time of departure
    c0 = stats.norm.pdf(td, mu[0], sigma[0]) * lam[0]
    c1 = stats.norm.pdf(td, mu[1], sigma[1]) * lam[1]
    c2 = stats.norm.pdf(td, mu[2], sigma[2]) * lam[2]
    cs = np.array([c0, c1, c2])/np.sum(np.array([c0,c1,c2]), axis=0)  # Normalized array of c weights.
    return cs


guess_mus = [14 * 60 * 60, 8 * 60 * 60, 17 * 60 * 60]
guess_sigmas = [3 * 60 * 60, 1 * 60 * 60, 1 * 60 * 60]
guess_lams = [1, 1, 1]
c_weights = e_step(rides, guess_mus, guess_sigmas, guess_lams)

# Update mixture parameters given the data and the current mixture weights.
def m_step(data, cs):
    """
    Update mixture parameters given the data and the current mixture weights.
    :param data: pandas dataframe with 'start_seconds'
    :param cs: array of 3 mixture weights.
    :return: hash {'mu', 'sigma', 'lambda'}
    """
    #Update mixture parameters using MLE.
    td = np.array(data['start_seconds'])

    lmbdas = np.mean(cs, axis=1)
    mus = np.array([np.sum(td*w)/np.sum(w) for w in cs])
    sigs = np.sqrt(np.array([np.sum(cs[j]*(td-mus[j])*(td-mus[j]))/np.sum(cs[j]) for j in range(len(mus))]))

    return {'mu': mus, 'sigma': sigs, 'lambda': lmbdas}




# Run E-M iteration until the weights don't change, or we lose patience.
itr = 0
old_c_weights = np.zeros(c_weights.shape)
parms = {}
while np.sum(np.abs(c_weights - old_c_weights)) > .0001:
    itr += 1
    old_c_weights = c_weights
    parms = m_step(rides, c_weights)
    print 'Model parameters: %s' % str(parms)
    c_weights = e_step(rides, parms['mu'], parms['sigma'], parms['lambda'])
    print 'Weights: %s' % str(c_weights)
    if itr > 10000:
        print "Incomplete!"
        break


#examine_weights(rides['start_seconds'], c_weights)

x = np.linspace(0,60*60*24, num=100000)
plt.plot(x, stats.norm.pdf(x, parms['mu'][0], parms['sigma'][0])*parms['lambda'][0], color='blue', label="joyriders")
plt.plot(x, stats.norm.pdf(x, parms['mu'][1], parms['sigma'][1])*parms['lambda'][1], color='green', label="morning")
plt.plot(x, stats.norm.pdf(x, parms['mu'][2], parms['sigma'][2])*parms['lambda'][2], color='red', label="evening")
plt.legend()
plt.show()
