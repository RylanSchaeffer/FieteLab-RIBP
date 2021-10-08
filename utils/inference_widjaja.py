# Taken from Widjaja https://dash.harvard.edu/handle/1/38811474
# Converted to PyTorch

import numpy as np
import scipy.stats as stats
import scipy.special as sps
import torch

torch.set_default_tensor_type('torch.FloatTensor')


class OnlineFinite:
    def __init__(self,
                 obs_dim: int,
                 max_num_features: int,
                 alpha: float,
                 beta: float,
                 sigma_a,
                 sigma_x,
                 t0=1,
                 kappa=0.5):

        self.obs_dim = obs_dim
        self.max_num_features = max_num_features

        self.alpha = alpha
        self.beta = beta
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = torch.full(size=(max_num_features,),
                                fill_value=alpha * beta / max_num_features)
        self.tau_2 = torch.full(size=(max_num_features,),
                                fill_value=beta)
        # self.phi = stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))
        # self.Phi = stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features))
        self.phi = torch.zeros(size=(obs_dim, max_num_features))
        self.Phi = torch.zeros(size=(obs_dim, max_num_features))

        self.iteration = 0

        self.nu_sum = torch.zeros(max_num_features)
        self.nu_comp_sum = torch.zeros(max_num_features)
        self.nu_cross_sum = torch.zeros((max_num_features, max_num_features))
        self.nu_data_sum = torch.zeros((max_num_features, obs_dim))

    def compute_sufficient_stats(self, rho, nu, data):
        size = data.shape[0]

        b_nu_sum = torch.sum(nu, dim=0)
        nu_sum = rho * self.nu_sum + b_nu_sum
        nu_comp_sum = rho * self.nu_comp_sum + (size - b_nu_sum)

        nu_cross_sum = rho * self.nu_cross_sum
        for i in range(size):
            nu_i = nu[i, :]
            nu_cross_sum += torch.outer(nu_i, nu_i)

        nu_data_sum = rho * self.nu_data_sum
        for i in range(size):
            nu_data_sum += torch.outer(nu[i, :], data[i, :])

        return nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum

    def train(self, data, convergence_iters=3):

        k_indices = torch.arange(self.max_num_features)
        size = data.shape[0]

        rho = (1.0 - (self.iteration + self.t0) ** -self.kappa)

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.max_num_features))).float()

        prior = self.tau_1 / (self.tau_1 + self.tau_2)
        prior = prior.reshape(1, -1)  # add a batch dimension

        for t in range(convergence_iters):
            nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

            # Update tau's i.e. params for pi_k
            self.tau_1 = self.alpha / self.max_num_features + nu_sum
            self.tau_2 = 1.0 + nu_comp_sum

            # Update phi's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                var = 1.0 / (1.0 / self.var_a + nu_sum[k] / self.var_x)
                self.phi[:, k] = (nu_data_sum[k, :] - torch.sum(nu_cross_sum[k, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x * var
                self.Phi[:, k] = var

            # Update nu's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                phi_k = self.phi[:, k]
                theta_k = sps.digamma(self.tau_1[k]) - sps.digamma(self.tau_2[k]) \
                        - (torch.sum(self.Phi[:, k]) + torch.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                for i in range(size):
                    theta = theta_k + torch.dot(
                        self.phi[:, k],
                        data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices],
                                               dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

        self.nu_sum, self.nu_comp_sum, self.nu_cross_sum, self.nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

        self.iteration += 1

        step_results = {
            'dish_eating_prior': prior,
            'dish_eating_posterior': torch.clone(nu),
            'A_mean': self.phi.T,  # transpose because has shape (obs dim, max num features)
            'A_cov': self.Phi.T,  # transpose because has shape (obs dim, max num features)
            'beta_param_1': torch.clone(self.tau_1),  # add batch dimension
            'beta_param_2': torch.clone(self.tau_2),  # add batch dimension
        }

        return step_results

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = torch.arange(self.max_num_features)
        size = data.shape[0]

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.max_num_features))).float().float()

        for t in range(convergence_iters):
            nu_orig = torch.clone(nu)

            # Update nu's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = sps.digamma(self.tau_1[k]) - sps.digamma(self.tau_2[k])

                for i in range(size):
                    theta = theta_k  \
                            - (torch.dot(train_mask[i, :], self.Phi[:, k]) + torch.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + torch.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if torch.sum(torch.abs(nu_orig - nu)) < convergence_threshold:
                break

        return nu


class OnlineInfinite:
    def __init__(self,
                 obs_dim: int,
                 max_num_features: int,
                 alpha: float,
                 beta: float,
                 sigma_a: float,
                 sigma_x: float,
                 t0=1,
                 kappa=0.5):

        self.obs_dim = obs_dim
        self.max_num_features = max_num_features

        self.alpha = alpha
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = torch.full(size=(max_num_features,), fill_value=alpha)
        self.tau_2 = torch.ones(size=(max_num_features,))
        # self.phi = stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))
        # self.Phi = stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features))
        self.phi = torch.zeros(size=(obs_dim, max_num_features))
        self.Phi = torch.zeros(size=(obs_dim, max_num_features))

        self.iteration = 0

        self.nu_sum = torch.zeros(size=(max_num_features,))
        self.nu_comp_sum = torch.zeros(size=(max_num_features,))
        self.nu_cross_sum = torch.zeros((max_num_features, max_num_features))
        self.nu_data_sum = torch.zeros((max_num_features, obs_dim))

    def compute_logq_unnormalized(self, feature_index):
        tau_1 = self.tau_1[:(feature_index + 1)]
        tau_2 = self.tau_2[:(feature_index + 1)]

        # select our relevant set of tau's (up to k)
        digamma_tau_1 = sps.digamma(tau_1)
        digamma_tau_2 = sps.digamma(tau_2)
        digamma_sum = sps.digamma(tau_1, tau_2)

        # compute the unnormalized optimal log(q) distribution
        digamma_tau1_cumsum = torch.append(0.0, torch.cumsum(digamma_tau_1[:-1]))
        digamma_sum_cumsum = torch.cumsum(digamma_sum)
        logq_unnormalized = digamma_tau_2 + digamma_tau1_cumsum - digamma_sum_cumsum

        return logq_unnormalized

    # Compute Elogstick variable for a given feature index
    def compute_Elogstick(self, feature_index):
        tmp = self.compute_logq_unnormalized(feature_index)
        qk_log = tmp - sps.logsumexp(tmp)
        qk = torch.exp(qk_log)
        Elogstick = torch.sum(qk * (tmp - qk_log))

        # return
        return Elogstick

    def compute_sufficient_stats(self, rho, nu, data):
        size = data.shape[0]

        b_nu_sum = torch.sum(nu, dim=0)
        nu_sum = rho * self.nu_sum + b_nu_sum
        nu_comp_sum = rho * self.nu_comp_sum + (size - b_nu_sum)

        nu_cross_sum = rho * self.nu_cross_sum
        for i in range(size):
            nu_i = nu[i, :]
            nu_cross_sum += torch.outer(nu_i, nu_i)

        nu_data_sum = rho * self.nu_data_sum
        for i in range(size):
            nu_data_sum += torch.outer(nu[i, :], data[i, :])

        return nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum

    def train(self, data, convergence_iters=1):
        k_indices = torch.arange(self.max_num_features)
        minibatch_size = data.shape[0]

        rho = (1.0 - (self.iteration + self.t0) ** -self.kappa)

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(minibatch_size, self.max_num_features))).float()

        prior = torch.cumprod(self.tau_1 / (self.tau_1 + self.tau_2))
        prior = prior.reshape(1, -1)  # add a batch dimension

        for t in range(convergence_iters):
            nu_sum, nu_comp_sum, nu_cross_sum, nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

            # Update tau's
            for k in range(self.max_num_features):
                logq_unnormalized = self.compute_logq_unnormalized(self.max_num_features - 1)

                qs = torch.zeros((self.max_num_features, self.max_num_features))
                for m in range(k, self.max_num_features):
                    tmp = logq_unnormalized[:(m + 1)]
                    qs[m, :(m + 1)] = torch.exp(tmp - sps.logsumexp(tmp))

                self.tau_1[k] = self.alpha + torch.sum(nu_sum[k:]) \
                                + torch.dot(nu_comp_sum[(k + 1):], torch.sum(qs[(k + 1):, (k + 1):], dim=1))
                self.tau_2[k] = 1.0 + torch.dot(nu_comp_sum[k:], qs[k:, k])

            # Update phi's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                var = 1.0 / (1.0 / self.var_a + nu_sum[k] / self.var_x)
                self.phi[:, k] = (nu_data_sum[k, :] - torch.sum(nu_cross_sum[k, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x * var
                self.Phi[:, k] = var

            # Update nu's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                phi_k = self.phi[:, k]
                theta_k = torch.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(k) \
                          - (torch.sum(self.Phi[:, k]) + torch.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                for i in range(minibatch_size):
                    theta = theta_k + torch.dot(self.phi[:, k], data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

        self.nu_sum, self.nu_comp_sum, self.nu_cross_sum, self.nu_data_sum = self.compute_sufficient_stats(rho, nu, data)

        self.iteration += 1

        step_results = {
            'dish_eating_prior': prior,
            'dish_eating_posterior': torch.clone(nu),
            'A_mean': self.phi.T,  # transpose because has shape (obs dim, max num features)
            'A_cov': self.Phi.T,  # transpose because has shape (obs dim, max num features)
            'stick_param_1': torch.clone(self.tau_1),  # add batch dimension
            'stick_param_2': torch.clone(self.tau_2),  # add batch dimension
        }

        return step_results

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = torch.arange(self.max_num_features)
        size = data.shape[0]

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.max_num_features))).float()

        for t in range(convergence_iters):
            nu_orig = torch.clone(nu)

            # Update nu's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = torch.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(k)

                for i in range(size):
                    theta = theta_k  \
                            - (torch.dot(train_mask[i, :], self.Phi[:, k]) + torch.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + torch.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if torch.sum(torch.abs(nu_orig - nu)) < convergence_threshold:
                break

        return nu


class OfflineFinite:
    def __init__(self,
                 obs_dim: int,
                 num_obs: int,
                 max_num_features: int,
                 alpha: float,
                 beta: float,
                 sigma_a: float,
                 sigma_x: float,
                 t0: float = 1,
                 kappa: float = 0.5):

        self.dim = obs_dim
        self.num_obs = num_obs
        self.num_features = max_num_features

        self.alpha = alpha
        self.beta = beta
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = torch.full(size=(max_num_features,), fill_value=alpha * beta / max_num_features)
        self.tau_2 = torch.full(size=(max_num_features,), fill_value=beta)
        self.mu = torch.from_numpy(
            stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))).float()
        self.tau = torch.from_numpy(
            stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features)))
        # self.phi = stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))
        # self.Phi = stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features))
        self.phi = torch.zeros(size=(obs_dim, max_num_features))
        self.Phi = torch.zeros(size=(obs_dim, max_num_features))

        self.iteration = 0

        self.nu_sum = torch.zeros(max_num_features)
        self.nu_comp_sum = torch.zeros(max_num_features)
        self.nu_cross_sum = torch.zeros((max_num_features, max_num_features))
        self.nu_data_sum = torch.zeros((max_num_features, obs_dim))

    def train(self, data, convergence_iters=2):
        k_indices = torch.arange(self.num_features)
        size = data.shape[0]

        tau_1 = torch.clone(self.tau_1)
        tau_2 = torch.clone(self.tau_2)
        mu = torch.clone(self.mu)
        tau = torch.clone(self.tau)
        phi = torch.clone(self.phi)
        Phi = torch.clone(self.Phi)

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.num_features))).float()

        for t in range(convergence_iters):
            order = [0, 1, 2] if self.iteration == 0 else [2, 0, 1]
            for p in order:
                if p == 0:
                    sum_nu = self.num_obs / size * torch.sum(nu, dim=0)

                    # Update tau's
                    tau_1 = self.alpha / self.num_features + sum_nu
                    tau_2 = 1 + self.num_obs - sum_nu
                elif p == 1:
                    sum_nu = self.num_obs / size * torch.sum(nu, dim=0)

                    # Update phi's
                    for k in range(self.num_features):
                        non_k = k_indices[k_indices != k]

                        var = 1.0 / self.var_a + sum_nu[k] / self.var_x

                        mean = torch.zeros((1, self.dim))
                        for i in range(size):
                            mean += nu[i, k] * (data[i, :] - torch.sum(nu[i, non_k] * phi[:, non_k], dim=1))
                        mean = mean / self.var_x
                        mean = self.num_obs / size * mean

                        mu[:, k] = mean
                        tau[:, k] = var

                        phi[:, k] = mean / var
                        Phi[:, k] = 1.0 / var
                elif p == 2:
                    # Update nu's
                    for k in range(self.num_features):
                        non_k = k_indices[k_indices != k]

                        phi_k = phi[:, k]
                        theta_k = sps.digamma(tau_1[k]) - sps.digamma(tau_2[k]) \
                                  - (torch.sum(Phi[:, k]) + torch.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                        for i in range(size):
                            theta = theta_k + torch.dot(phi[:, k], data[i, :] - torch.sum(nu[i, non_k] * phi[:, non_k], dim=1)) / self.var_x

                            nu[i, k] = sps.expit(theta)

        rho = (self.iteration + self.t0) ** -self.kappa

        self.tau_1 = (1.0 - rho) * self.tau_1 + rho * tau_1
        self.tau_2 = (1.0 - rho) * self.tau_2 + rho * tau_2
        self.mu = (1.0 - rho) * self.mu + rho * mu
        self.tau = (1.0 - rho) * self.tau + rho * tau
        self.phi = self.mu / self.tau
        self.Phi = 1.0 / self.tau

        self.iteration += 1

        dish_eating_posterior = torch.clone(nu)
        dish_eating_prior = torch.full_like(dish_eating_posterior, fill_value=np.nan)

        step_results = {
            'dish_eating_prior': dish_eating_prior,
            'dish_eating_posterior': dish_eating_posterior,
            'A_mean': self.phi.T,  # transpose because has shape (obs dim, max num features)
            'A_cov': self.Phi.T,  # transpose because has shape (obs dim, max num features)
            'beta_param_1': torch.clone(self.tau_1),  # add batch dimension
            'beta_param_2': torch.clone(self.tau_2),  # add batch dimension
        }

        return step_results

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = torch.arange(self.num_features)
        size = data.shape[0]

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.num_features))).float()

        for t in range(convergence_iters):
            nu_orig = torch.clone(nu)

            # Update nu's
            for k in range(self.num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = sps.digamma(self.tau_1[k]) - sps.digamma(self.tau_2[k])

                for i in range(size):
                    theta = theta_k  \
                            - (torch.dot(train_mask[i, :], self.Phi[:, k]) + torch.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + torch.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if torch.sum(torch.abs(nu_orig - nu)) < convergence_threshold:
                break

        return nu


class OfflineInfinite:
    def __init__(self,
                 obs_dim: int,
                 num_obs: int,
                 max_num_features: int,
                 alpha: float,
                 beta: float,
                 sigma_a: float,
                 sigma_x: float,
                 t0=10,
                 kappa=0.5):

        self.obs_dim = obs_dim
        self.num_obs = num_obs
        self.max_num_features = max_num_features

        self.alpha = alpha
        self.var_a = sigma_a ** 2
        self.var_x = sigma_x ** 2

        self.t0 = t0
        self.kappa = kappa

        self.tau_1 = torch.full(size=(max_num_features,), fill_value=alpha / max_num_features)
        self.tau_2 = torch.ones(max_num_features)
        self.mu = stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))
        self.tau = stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features))
        # self.phi = stats.norm.rvs(scale=0.01, size=(obs_dim, max_num_features))
        # self.Phi = stats.norm.rvs(scale=0.1, size=(obs_dim, max_num_features))
        self.phi = torch.zeros(size=(obs_dim, max_num_features))
        self.Phi = torch.zeros(size=(obs_dim, max_num_features))

        self.iteration = 0

        self.nu_sum = torch.zeros(max_num_features)
        self.nu_comp_sum = torch.zeros(max_num_features)
        self.nu_cross_sum = torch.zeros((max_num_features, max_num_features))
        self.nu_data_sum = torch.zeros((max_num_features, obs_dim))

    def compute_logq_unnormalized(self, tau_1, tau_2, feature_index):
        tau_1 = tau_1[:(feature_index + 1)]
        tau_2 = tau_2[:(feature_index + 1)]

        # select our relevant set of tau's (up to k)
        digamma_tau_1 = sps.digamma(tau_1)
        digamma_tau_2 = sps.digamma(tau_2)
        digamma_sum = sps.digamma(tau_1, tau_2)

        # compute the unnormalized optimal log(q) distribution
        digamma_tau1_cumsum = torch.append(0.0, torch.cumsum(digamma_tau_1[:-1]))
        digamma_sum_cumsum = torch.cumsum(digamma_sum)
        logq_unnormalized = digamma_tau_2 + digamma_tau1_cumsum - digamma_sum_cumsum

        return logq_unnormalized

    # Compute Elogstick variable for a given feature index
    def compute_Elogstick(self, tau_1, tau_2, feature_index):
        tmp = self.compute_logq_unnormalized(tau_1, tau_2, feature_index)
        qk_log = tmp - sps.logsumexp(tmp)
        qk = torch.exp(qk_log)
        Elogstick = torch.sum(qk * (tmp - qk_log))

        # return
        return Elogstick

    def train(self, data, convergence_iters=1):
        k_indices = torch.arange(self.max_num_features)
        num_obs = data.shape[0]

        tau_1 = torch.clone(self.tau_1)
        tau_2 = torch.clone(self.tau_2)
        mu = torch.clone(self.mu)
        tau = torch.clone(self.tau)
        phi = torch.clone(self.phi)
        Phi = torch.clone(self.Phi)

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(num_obs, self.max_num_features))).float()

        for t in range(convergence_iters):
            order = [0, 1, 2] if self.iteration == 0 else [2, 0, 1]
            for p in order:
                if p == 0:
                    sum_nu = self.num_obs / num_obs * torch.sum(nu, dim=0)

                    # Update tau's
                    for k in range(self.max_num_features):
                        logq_unnormalized = self.compute_logq_unnormalized(tau_1, tau_2, self.max_num_features - 1)

                        qs = torch.zeros((self.max_num_features, self.max_num_features))
                        for m in range(k, self.max_num_features):
                            tmp = logq_unnormalized[:(m + 1)]
                            qs[m, :(m + 1)] = torch.exp(tmp - sps.logsumexp(tmp))

                        tau_1[k] = self.alpha + torch.sum(sum_nu[k:]) + torch.dot(self.num_obs - sum_nu[(k + 1):],
                                                                            torch.sum(qs[(k + 1):, (k + 1):], dim=1))
                        tau_2[k] = 1.0 + torch.dot(self.num_obs - sum_nu[k:], qs[k:, k])
                elif p == 1:
                    sum_nu = self.num_obs / num_obs * torch.sum(nu, dim=0)

                    # Update phi's
                    for k in range(self.max_num_features):
                        non_k = k_indices[k_indices != k]

                        var = 1.0 / self.var_a + sum_nu[k] / self.var_x

                        mean = torch.zeros((1, self.obs_dim))
                        for i in range(num_obs):
                            mean += nu[i, k] * (data[i, :] - torch.sum(nu[i, non_k] * phi[:, non_k], dim=1))
                        mean = mean / self.var_x
                        mean = self.num_obs / num_obs * mean

                        mu[:, k] = mean
                        tau[:, k] = var

                        phi[:, k] = mean / var
                        Phi[:, k] = 1.0 / var
                elif p == 2:
                    # Update nu's
                    for k in range(self.max_num_features):
                        non_k = k_indices[k_indices != k]

                        phi_k = phi[:, k]
                        theta_k = torch.sum(sps.digamma(tau_1[:(k + 1)]) - sps.digamma(tau_1[:(k + 1)] + tau_2[:(k + 1)])) \
                                  - self.compute_Elogstick(tau_1, tau_2, k) \
                                  - (torch.sum(Phi[:, k]) + torch.dot(phi_k, phi_k)) / (2.0 * self.var_x)

                        for i in range(num_obs):
                            theta = theta_k + torch.dot(phi[:, k], data[i, :] - torch.sum(nu[i, non_k] * phi[:, non_k], dim=1)) / self.var_x

                            nu[i, k] = sps.expit(theta)

        rho = (self.iteration + self.t0) ** -self.kappa

        self.tau_1 = (1.0 - rho) * self.tau_1 + rho * tau_1
        self.tau_2 = (1.0 - rho) * self.tau_2 + rho * tau_2
        self.mu = (1.0 - rho) * self.mu + rho * mu
        self.tau = (1.0 - rho) * self.tau + rho * tau
        self.phi = self.mu / self.tau
        self.Phi = 1.0 / self.tau

        self.iteration += 1

        posterior = torch.clone(nu)  # remove batch dimension
        prior = torch.full_like(posterior, fill_value=torch.nan)

        step_results = {
            'dish_eating_prior': prior,
            'dish_eating_posterior': posterior,
            'A_mean': self.phi.T,  # transpose because has shape (obs dim, max num features)
            'A_cov': self.Phi.T,  # transpose because has shape (obs dim, max num features)
            'stick_param_1': torch.clone(self.tau_1),  # add batch dimension
            'stick_param_2': torch.clone(self.tau_2),  # add batch dimension
        }

        return step_results

    def test(self, data, train_mask, convergence_iters=10, convergence_threshold=1e-3):
        k_indices = torch.arange(self.max_num_features)
        size = data.shape[0]

        nu = torch.from_numpy(
            stats.uniform.rvs(size=(size, self.max_num_features))).float()

        for t in range(convergence_iters):
            nu_orig = torch.clone(nu)

            # Update nu's
            for k in range(self.max_num_features):
                non_k_indices = k_indices[k_indices != k]

                theta_k = torch.sum(sps.digamma(self.tau_1[:(k + 1)]) - sps.digamma(self.tau_1[:(k + 1)] + self.tau_2[:(k + 1)])) \
                          - self.compute_Elogstick(self.tau_1, self.tau_2, k)

                for i in range(size):
                    theta = theta_k  \
                            - (torch.dot(train_mask[i, :], self.Phi[:, k]) + torch.dot(train_mask[i, :], self.phi[:, k] ** 2)) / (2.0 * self.var_x) \
                            + torch.dot(train_mask[i, :] * self.phi[:, k], data[i, :] - torch.sum(nu[i, non_k_indices] * self.phi[:, non_k_indices], dim=1)) / self.var_x

                    nu[i, k] = sps.expit(theta)

            if torch.sum(torch.abs(nu_orig - nu)) < convergence_threshold:
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

    def step(self, obs_indices):

        # add a batch dimension
        minibatch = self.data_source[obs_indices]

        step_results = self.model.train(minibatch)

        self.iteration += 1

        # if self.iteration % 10 == 0:
        #     self.iteration_set.append(self.iteration)
        #     self.time_set.append(self.time)
        #
        #     ll_mean, ll_std = predictive_log_likelihood(self.model, self.data_source)
        #     self.ll_mean_set.append(ll_mean)
        #     self.ll_std_set.append(ll_std)

        return step_results

