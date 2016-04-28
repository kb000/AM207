import numpy as np
import scipy.stats as stats

import simulated_annealer as sa

class MixModel:
    """
    A Mixture Model for use in EM with Simulated Annealing.
    """
    _axis = 0

    def __init__(self, weights, mus, sigmas):
        self.mu = np.array(mus)
        self.s = np.array(sigmas)
        self.w = np.array(weights)

    def __getitem__(self, key):
        if 0 <= key < 3:
            return self.mu[key]
        else:
            return self.s[key - 3]

    def __setitem__(self, key, val):
        if 0 <= key < 3:
            self.mu[key] = val
        else:
            self.s[key-3] = val

    def __len__(self):
        return 6

    def copy(self):
        other = MixModel(self.w.copy(), self.mu.copy(), self.s.copy())
        other._axis = self._axis
        return other

    def em_log_like(self, data):
        """Calculates the EM log likelihood for the mixture model"""
        m = self
        ll_part = lambda w, m, s: np.sum(w*(stats.norm.logpdf(data, m, s) - np.log(w)))
        c0part = ll_part(m.w[0], m.mu[0], m.s[0])
        c1part = ll_part(m.w[1], m.mu[1], m.s[1])
        c2part = ll_part(m.w[2], m.mu[2], m.s[2])
        return c0part + c1part + c2part

    def mh_step(self, step_size):
        axis_scales = [180, 180, 180, 60, 60, 60]  # Scale axes to seconds.
        axis_domains = [(0, 86400), (0, 86400), (0, 86400), (1, np.inf), (1, np.inf),  (1, np.inf)]
        i = self._axis % len(self)
        old = self[i]
        new = np.nan
        while not axis_domains[i][0] < new < axis_domains[i][1]:
            new = old + np.random.normal(0, step_size * axis_scales[i])
        self._axis += 1
        self[i] = new
        return self
