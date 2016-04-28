import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from cleaning import rides
from simulated_annealer import simulate_annealing
from mix_model import MixModel

rides = rides[:10000:4]

def examine_weights(times, ws):
    plt.scatter(times, ws[0], c='red')
    plt.scatter(times, ws[1], c='blue')
    plt.scatter(times, ws[2], c='green')
    plt.xlim((0, 86400))
    plt.ylim((0, 1))
    plt.show()


def e_step(data, model):
    """
    Find weights for each c_i given the data and current mixture parameters
    :data: pandas dataframe with 'start_seconds'
    :model: A MixModel.
    :return: A new model with weights adjusted.
    """
    td = np.array(data['start_seconds'])  # Time of departure
    c0 = stats.norm.pdf(td, model.mu[0], model.s[0])
    c1 = stats.norm.pdf(td, model.mu[1], model.s[1])
    c2 = stats.norm.pdf(td, model.mu[2], model.s[2])
    cs = np.array([c0, c1, c2]) / np.sum(np.array([c0, c1, c2]), axis=0)  # Normalized array of c weights.
    new_model = model.copy()
    new_model.w = cs
    return new_model


# Update mixture parameters given the data and the current mixture weights.
def m_step(data, model):
    """
    Update mixture parameters given the data and the current mixture weights.
    :param data: pandas dataframe with 'start_seconds'
    :param model: A MixModel
    :return: A new model with adjusted parameters.
    """
    td = np.array(data['start_seconds'])
    #burnt = simulate_annealing(model.copy(),
    #                           energy_meter=lambda m: -m.em_log_like(td)/100000,
    #                           stepper=lambda m, size: m.mh_step(size),
    #                           init_temp=2.0,
    #                           max_iterations=1000, stopping_energy=1000)

    optimal = simulate_annealing(model.copy(),
                                 energy_meter=lambda m: -m.em_log_like(td),
                                 stepper=lambda m, size: m.mh_step(size),
                                 max_iterations=100000)

    return optimal


# Initialize weights from a guess on parameters.
guess_mus = [14 * 60 * 60, 8 * 60 * 60, 17 * 60 * 60]
guess_sigmas = [3 * 60 * 60, 1 * 60 * 60, 1 * 60 * 60]
init_model = MixModel(weights=np.zeros(3), mus=guess_mus, sigmas=guess_sigmas)
e_model = e_step(rides, init_model)

# Run E-M iteration until the weights don't change, or we hit 100 iterations.
itr = 0
old_model = MixModel(np.zeros(e_model.w.shape), np.zeros(3), np.zeros(3))
while np.sum(np.abs(e_model.w - old_model.w)) > .0001:
    itr += 1
    old_model = e_model.copy()
    m_model = m_step(rides, e_model)
    parm_hash = {'mu': m_model.mu, 'sigma': m_model.s}
    print 'Model parameters: %s' % str(parm_hash)
    e_model = e_step(rides, m_model)
    print 'Weights: %s' % str(e_model.w)
    examine_weights(rides['start_seconds'], e_model.w)
    if itr > 100:
        break
