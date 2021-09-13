import numpy as np
import scipy.stats as stats
import scipy.special as sps
import scipy.misc as spm

import time


class StreamingInfinite:
    def __init__(self, dim, num_features, alpha, sigma_a, sigma_x, t0=1, kappa=0.5):
        self.dim = dim
        self.num_features = num_features

        self.alpha = alpha
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = np.full(num_features, alpha)
        self.tau_2 = np.ones(num_features)
        self.phi = stats.norm.rvs(scale=0.01, size=(dim, num_features))
        self.Phi = stats.norm.rvs(scale=0.1, size=(dim, num_features))

        self.iteration = 0

        self.nu_sum = np.zeros(num_features)
        self.nu_comp_sum = np.zeros(num_features)
        self.nu_cross_sum = np.zeros((num_features, num_features))
        self.nu_data_sum = np.zeros((num_features, dim))

    def compute_logq_unnormalized(self, feature_index):
        tau_1 = self.tau_1[:(feature_index + 1)]
        tau_2 = self.tau_2[:(feature_index + 1)]

        # select our relevant set of tau's (up to k)
        digamma_tau_1 = sps.digamma(tau_1)
        digamma_tau_2 = sps.digamma(tau_2)
        digamma_sum = sps.digamma(tau_1, tau_2)

        # compute the unnormalized optimal log(q) distribution
        digamma_tau1_cumsum = np.append(0.0, np.cumsum(digamma_tau_1[:-1]))
        digamma_sum_cumsum = np.cumsum(digamma_sum)
        logq_unnormalized = digamma_tau_2 + digamma_tau1_cumsum - digamma_sum_cumsum

        return logq_unnormalized

    # Compute Elogstick variable for a given feature index
    def compute_Elogstick(self, feature_index):
        tmp = self.compute_logq_unnormalized(feature_index)
        qk_log = tmp - sps.logsumexp(tmp)
        qk = np.exp(qk_log)
        Elogstick = np.sum(qk * (tmp - qk_log))

        # return
        return Elogstick

    def compute_sufficient_stats(self, rho, nu, data):
        size = data.shape[0]

        b_nu_sum = np.sum(nu, axis=0)
        nu_sum = rho * self.nu_sum + b_nu_sum
        nu_comp_sum = rho * self.nu_comp_sum + (size - b_nu_sum)

        nu_cross_sum = rho * self.nu_cross_sum
        for i in range(size):
            nu_i = nu[i, :]
            nu_cross_sum += np.outer(nu_i, nu_i)

        nu_data_sum = rho * self.nu_data_sum
        for i in range(size):
            nu_data_sum += np.outer(nu[i, :], data[i, :])

        return nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum

    def train(self, data, convergence_iters=1):
        k_indices = np.arange(self.num_features)
        minibatch_size = data.shape[0]

        rho = (1.0 - (self.iteration + self.t0) ** -self.kappa)

        nu = stats.uniform.rvs(size=(minibatch_size, self.num_features))

        prior = np.cumprod(self.tau_1 / (self.tau_1 + self.tau_2))

        for t in range(convergence_iters):
            nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

            # Update tau's
            for k in range(self.num_features):
                logq_unnormalized = self.compute_logq_unnormalized(self.num_features - 1)

                qs = np.zeros((self.num_features, self.num_features))
                for m in range(k, self.num_features):
                    tmp = logq_unnormalized[:(m + 1)]
                    qs[m, :(m + 1)] = np.exp(tmp - sps.logsumexp(tmp))

                self.tau_1[k] = self.alpha + np.sum(nu_sum[k:]) \
                                + np.dot(nu_comp_sum[(k + 1):], np.sum(qs[(k + 1):, (k + 1):], axis=1))
                self.tau_2[k] = 1.0 + np.dot(nu_comp_sum[k:], qs[k:, k])

            # Update phi's
            for k in range(self.num_features):
                non_k_indices = k_indices[k_indices != k]

                var = 1.0 / (1.0 / self.var_a + nu_sum[k] / self.var_x)
                self.phi[:, k] = (nu_data_sum[k, :] - np.sum(nu_cross_sum[k, non_k_indices] * self.phi[:, non_k_indices], axis=1)) / self.var_x * var
                self.Phi[:, k] = var

            # Update nu's
            for k in range(self.num_features):
                non_k_indices = k_indices[k_indices != k]

                phi_k = self.phi[:, k]
                theta_k = np.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(k) \
                          - (np.sum(self.Phi[:, k]) + np.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                for i in range(minibatch_size):
                    theta = theta_k + np.dot(self.phi[:, k], data[i, :] - np.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], axis=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

        self.nu_sum, self.nu_comp_sum, self.nu_cross_sum, self.nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

        self.iteration += 1

        step_results = {
            'dish_eating_prior': prior,
            'dish_eating_posterior': nu.copy()[0, :],  # remove batch dimension
            'A_mean': self.phi.T,  # transpose because has shape (obs dim, max num features)
            'A_cov': self.Phi.T,  # transpose because has shape (obs dim, max num features)
        }

        return step_results

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = np.arange(self.num_features)
        size = data.shape[0]

        nu = stats.uniform.rvs(size=(size, self.num_features))

        for t in range(convergence_iters):
            nu_orig = np.copy(nu)

            # Update nu's
            for k in range(self.num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = np.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(k)

                for i in range(size):
                    theta = theta_k  \
                            - (np.dot(train_mask[i, :], self.Phi[:, k]) + np.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + np.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - np.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], axis=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if np.sum(np.abs(nu_orig - nu)) < convergence_threshold:
                break

        return nu


class StochasticInfinite:
    def __init__(self, dim, data_size, num_features, alpha, sigma_a, sigma_x, t0=10, kappa=0.5):
        self.dim = dim
        self.data_size = data_size
        self.num_features = num_features

        self.alpha = alpha
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = np.full(num_features, alpha / num_features)
        self.tau_2 = np.ones(num_features)
        self.mu = stats.norm.rvs(scale=0.01, size=(dim, num_features))
        self.tau = stats.norm.rvs(scale=0.1, size=(dim, num_features))
        self.phi = stats.norm.rvs(scale=0.01, size=(dim, num_features))
        self.Phi = stats.norm.rvs(scale=0.1, size=(dim, num_features))

        self.iteration = 0

        self.nu_sum = np.zeros(num_features)
        self.nu_comp_sum = np.zeros(num_features)
        self.nu_cross_sum = np.zeros((num_features, num_features))
        self.nu_data_sum = np.zeros((num_features, dim))

    def compute_logq_unnormalized(self, tau_1, tau_2, feature_index):
        tau_1 = tau_1[:(feature_index + 1)]
        tau_2 = tau_2[:(feature_index + 1)]

        # select our relevant set of tau's (up to k)
        digamma_tau_1 = sps.digamma(tau_1)
        digamma_tau_2 = sps.digamma(tau_2)
        digamma_sum = sps.digamma(tau_1, tau_2)

        # compute the unnormalized optimal log(q) distribution
        digamma_tau1_cumsum = np.append(0.0, np.cumsum(digamma_tau_1[:-1]))
        digamma_sum_cumsum = np.cumsum(digamma_sum)
        logq_unnormalized = digamma_tau_2 + digamma_tau1_cumsum - digamma_sum_cumsum

        return logq_unnormalized

    # Compute Elogstick variable for a given feature index
    def compute_Elogstick(self, tau_1, tau_2, feature_index):
        tmp = self.compute_logq_unnormalized(tau_1, tau_2, feature_index)
        qk_log = tmp - sps.logsumexp(tmp)
        qk = np.exp(qk_log)
        Elogstick = np.sum(qk * (tmp - qk_log))

        # return
        return Elogstick

    def train(self, data, convergence_iters=1):
        k_indices = np.arange(self.num_features)
        size = data.shape[0]

        tau_1 = np.copy(self.tau_1)
        tau_2 = np.copy(self.tau_2)
        mu = np.copy(self.mu)
        tau = np.copy(self.tau)
        phi = np.copy(self.phi)
        Phi = np.copy(self.Phi)

        nu = stats.uniform.rvs(size=(size, self.num_features))

        for t in range(convergence_iters):
            order = [0, 1, 2] if self.iteration == 0 else [2, 0, 1]
            for p in order:
                if p == 0:
                    sum_nu = self.data_size / size * np.sum(nu, axis=0)

                    # Update tau's
                    for k in range(self.num_features):
                        logq_unnormalized = self.compute_logq_unnormalized(tau_1, tau_2, self.num_features - 1)

                        qs = np.zeros((self.num_features, self.num_features))
                        for m in range(k, self.num_features):
                            tmp = logq_unnormalized[:(m + 1)]
                            qs[m, :(m + 1)] = np.exp(tmp - sps.logsumexp(tmp))

                        tau_1[k] = self.alpha + np.sum(sum_nu[k:]) + np.dot(self.data_size - sum_nu[(k + 1):],
                                                                            np.sum(qs[(k + 1):, (k + 1):], axis=1))
                        tau_2[k] = 1.0 + np.dot(self.data_size - sum_nu[k:], qs[k:, k])
                elif p == 1:
                    sum_nu = self.data_size / size * np.sum(nu, axis=0)

                    # Update phi's
                    for k in range(self.num_features):
                        non_k = k_indices[k_indices != k]

                        var = 1.0 / self.var_a + sum_nu[k] / self.var_x

                        mean = np.zeros((1, self.dim))
                        for i in range(size):
                            mean += nu[i, k] * (data[i, :] - np.sum(nu[i, non_k] * phi[:, non_k], axis=1))
                        mean = mean / self.var_x
                        mean = self.data_size / size * mean

                        mu[:, k] = mean
                        tau[:, k] = var

                        phi[:, k] = mean / var
                        Phi[:, k] = 1.0 / var
                elif p == 2:
                    # Update nu's
                    for k in range(self.num_features):
                        non_k = k_indices[k_indices != k]

                        phi_k = phi[:, k]
                        theta_k = np.sum(sps.digamma(tau_1[:(k + 1)]) - sps.digamma(tau_1[:(k + 1)] + tau_2[:(k + 1)])) \
                                  - self.compute_Elogstick(tau_1, tau_2, k) \
                                  - (np.sum(Phi[:, k]) + np.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                        for i in range(size):
                            theta = theta_k + np.dot(phi[:, k], data[i, :] - np.sum(nu[i, non_k] * phi[:, non_k], axis=1)) / self.var_x

                            nu[i, k] = sps.expit(theta)

        rho = (self.iteration + self.t0) ** -self.kappa

        self.tau_1 = (1.0 - rho) * self.tau_1 + rho * tau_1
        self.tau_2 = (1.0 - rho) * self.tau_2 + rho * tau_2
        self.mu = (1.0 - rho) * self.mu + rho * mu
        self.tau = (1.0 - rho) * self.tau + rho * tau
        self.phi = self.mu / self.tau
        self.Phi = 1.0 / self.tau

        self.iteration += 1

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = np.arange(self.num_features)
        size = data.shape[0]

        nu = stats.uniform.rvs(size=(size, self.num_features))

        for t in range(convergence_iters):
            nu_orig = np.copy(nu)

            # Update nu's
            for k in range(self.num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = np.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(self.tau_1, self.tau_2, k)

                for i in range(size):
                    theta = theta_k  \
                            - (np.dot(train_mask[i, :], self.Phi[:, k]) + np.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + np.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - np.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], axis=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if np.sum(np.abs(nu_orig - nu)) < convergence_threshold:
                break

        return nu


class Static:
    def __init__(self, model, data_source, minibatch_size):
        self.model = model
        self.data_source = data_source
        self.minibatch_size = minibatch_size

        self.iteration = 0
        self.time = 0.0

        self.iteration_set = []
        self.data_seen_set = []
        self.time_set = []
        self.ll_mean_set = []
        self.ll_std_set = []

    def features(self):
        return self.model.phi.T

    def step(self, obs_idx):

        # add a batch dimension
        minibatch = self.data_source[obs_idx].reshape(1, -1)

        start_time = time.time()
        step_results = self.model.train(minibatch)
        end_time = time.time()

        self.iteration += 1
        self.time += end_time - start_time

        # if self.iteration % 10 == 0:
        #     self.iteration_set.append(self.iteration)
        #     self.time_set.append(self.time)
        #
        #     ll_mean, ll_std = predictive_log_likelihood(self.model, self.data_source)
        #     self.ll_mean_set.append(ll_mean)
        #     self.ll_std_set.append(ll_std)

        return step_results


def predictive_log_likelihood(model, data_source, samples=100):
    nu = model.test(data_source.test_data, 1 - data_source.test_mask)

    ll = np.empty(samples)
    for s in range(samples):
        Z = np.random.binomial(1, p=nu)
        A = np.random.normal(loc=model.phi, scale=model.Phi).T
        ll[s] = np.sum(data_source.test_mask * ((data_source.test_data - np.dot(Z, A)) ** 2))

    ll = -ll / (2.0 * model.var_x)
    ll -= np.sum(data_source.test_mask) / 2.0 * np.log(2.0 * np.pi * model.var_x)

    return np.mean(ll), np.std(ll)
