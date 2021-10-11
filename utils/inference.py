import logging
import numpy as np
import torch
from typing import Dict, Tuple, Union

import inference.factor_analysis
import inference.linear_gaussian


# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')

inference_algs = [
    'HMC-Gibbs',
    'Doshi-Velez-Finite',
    'Doshi-Velez-Infinite',
    'R-IBP',
    'Widjaja-Finite',
    'Widjaja-Infinite',
]


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      model_str: str,
                      gen_model_params: dict,
                      plot_dir: str = None):
    if model_str == 'linear_gaussian':
        assert 'IBP' in gen_model_params
        assert 'feature_prior_params' in gen_model_params
        assert 'likelihood_params' in gen_model_params

        # select inference alg and add kwargs as necessary
        if inference_alg_str.startswith('Doshi-Velez'):
            gen_model_params['t0'] = 0.
            gen_model_params['kappa'] = 0.
            num_coordinate_ascent_steps = 17
            logging.info(f'Number of coordinate ascent steps: '
                         f'{num_coordinate_ascent_steps}')
            inference_alg = inference.linear_gaussian.DoshiVelezLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                num_coordinate_ascent_steps=num_coordinate_ascent_steps,
                use_infinite=True if 'Infinite' in inference_alg_str else False)
        elif inference_alg_str == 'R-IBP':
            inference_alg = inference.linear_gaussian.RecursiveIBPLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                plot_dir=None,
                # plot_dir=plot_dir,
            )
        elif inference_alg_str.startswith('Widjaja'):
            gen_model_params['t0'] = 0
            gen_model_params['kappa'] = 0.
            inference_alg = inference.linear_gaussian.WidjajaLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                use_infinite=True if 'Infinite' in inference_alg_str else False)
        # elif inference_alg_str == 'Online CRP':
        #     inference_alg = online_crp
        # elif inference_alg_str == 'SUSG':
        #     inference_alg = sequential_updating_and_greedy_search
        # elif inference_alg_str == 'VSUSG':
        #     inference_alg = variational_sequential_updating_and_greedy_search
        # elif inference_alg_str.startswith('DP-Means'):
        #     inference_alg = dp_means
        #     if inference_alg_str.endswith('(offline)'):
        #         inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan
        #     elif inference_alg_str.endswith('(online)'):
        #         inference_alg_kwargs['num_passes'] = 1
        #     else:
        #         raise ValueError('Invalid DP Means')
        elif inference_alg_str.startswith('HMC-Gibbs'):
            inference_alg = inference.linear_gaussian.HMCGibbsLinearGaussian(
                model_str=model_str,
                model_params=gen_model_params,
                num_samples=100000,
                num_warmup_samples=50000,
                num_thinning_samples=1000)

            # Suppose inference_alg_str is 'HMC-Gibbs (5000 Samples)'. We want to extract
            # the number of samples. To do this, we use the following
            # num_samples = int(inference_alg_str.split(' ')[1][1:])

        # elif inference_alg_str.startswith('SVI'):
        #     inference_alg = stochastic_variational_inference
        #     learning_rate = 5e-4
        #     # suppose the inference_alg_str is 'SVI (5k Steps)'
        #     substrs = inference_alg_str.split(' ')
        #     num_steps = 1000 * int(substrs[1][1:-1])
        #     inference_alg_kwargs['num_steps'] = num_steps
        #     # Note: these are the ground truth params
        #     if likelihood_model == 'dirichlet_multinomial':
        #         inference_alg_kwargs['model_params'] = dict(
        #             dirichlet_inference_params=10.)  # same as R-CRP
        #     elif likelihood_model == 'multivariate_normal':
        #         inference_alg_kwargs['model_params'] = dict(
        #             gaussian_mean_prior_cov_scaling=6.,
        #             gaussian_cov_scaling=0.3)
        #     else:
        #         raise ValueError
        # elif inference_alg_str.startswith('Variational Bayes'):
        #     inference_alg = variational_bayes
        #     # Suppose we have an algorithm string 'Variational Bayes (10 Init, 10 Iterations)',
        #     substrs = inference_alg_str.split(' ')
        #     num_initializations = int(substrs[2][1:])
        #     max_iters = int(substrs[4])
        #     inference_alg_kwargs['num_initializations'] = num_initializations
        #     inference_alg_kwargs['max_iter'] = max_iters
        else:
            raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')


    elif model_str == 'factor_analysis':
        raise NotImplementedError
    elif model_str == 'nonnegative_matrix_factorization':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # run inference algorithm
    inference_alg_results = inference_alg.fit(
        observations=observations)

    # Add inference alg object to results, for later generating predictions
    inference_alg_results['inference_alg'] = inference_alg

    return inference_alg_results
