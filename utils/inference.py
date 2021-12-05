import logging
import numpy as np
import torch

import utils.prob_models.factor_analysis
import utils.prob_models.linear_gaussian


# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')

inference_algs = [
    'HMC-Gibbs',
    'Collapsed-Gibbs',
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
            inference_alg = utils.prob_models.linear_gaussian.DoshiVelezLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                num_coordinate_ascent_steps=num_coordinate_ascent_steps,
                use_infinite=True if 'Infinite' in inference_alg_str else False)
        elif inference_alg_str == 'R-IBP':
            inference_alg = utils.prob_models.linear_gaussian.RecursiveIBPLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                plot_dir=None,
                # plot_dir=plot_dir,
            )
        elif inference_alg_str.startswith('Widjaja'):
            gen_model_params['t0'] = 0
            gen_model_params['kappa'] = 0.
            inference_alg = utils.prob_models.linear_gaussian.WidjajaLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                use_infinite=True if 'Infinite' in inference_alg_str else False)
        elif inference_alg_str.startswith('HMC-Gibbs'):
            inference_alg = utils.prob_models.linear_gaussian.HMCGibbsLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
                num_samples=50000,  # 100000
                num_warmup_samples=1000,  # 10000
                num_thinning_samples=500,  # 1000
            )
        elif inference_alg_str == 'Collapsed-Gibbs':
            inference_alg = utils.prob_models.linear_gaussian.CollapsedGibbsLinearGaussian(
                model_str=model_str,
                gen_model_params=gen_model_params,
            )
        else:
            raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    elif model_str == 'factor_analysis':
        assert 'IBP' in gen_model_params
        assert 'feature_prior_params' in gen_model_params
        assert 'scale_prior_params' in gen_model_params
        assert 'likelihood_params' in gen_model_params

        if inference_alg_str == 'R-IBP':
            inference_alg = utils.prob_models.factor_analysis.RecursiveIBPFactorAnalysis(
                model_str=model_str,
                gen_model_params=gen_model_params,
                plot_dir=None,
                num_coord_ascent_steps_per_obs=5,
                # plot_dir=plot_dir,
            )
        else:
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
