import numpy as np
import torch


class LinearGaussian(torch.nn.Module):

    def __init__(self, num_obs, obs_dim, max_num_features):
        super(LinearGaussian, self).__init__()

        # dict mapping variables to variational parameters
        # we use half covariance because we want to numerically optimize
        A_half_cov = torch.stack([torch.eye(obs_dim, obs_dim)
                                  for _ in range((num_obs + 1) * max_num_features)])
        A_half_cov = A_half_cov.view(num_obs + 1, max_num_features, obs_dim, obs_dim)
        A_half_cov.requires_grad = True
        self.variable_variational_params = torch.nn.ModuleDict({
            'Z': dict(  # variational parameters for binary indicators
                prob=torch.full(
                    size=(num_obs + 1, max_num_features),
                    fill_value=np.nan,
                    dtype=torch.float64,
                    requires_grad=False)
            ),
            'A': dict(  # variational parameters for Gaussian features
                params=dict(
                    mean=torch.full(
                        size=(num_obs + 1, max_num_features, obs_dim),
                        fill_value=0.,
                        dtype=torch.float64,
                        requires_grad=True),
                    # mean=torch.from_numpy(
                    #     np.random.normal(size=(num_obs + 1, max_num_features, obs_dim)),
                    #     # requires_grad=True
                    # ),
                    half_cov=A_half_cov),
                optimize_fns=dict(),
            )
        })
