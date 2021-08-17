import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(1)

alpha = 3.7
num_customers = 100
num_dishes = int(np.round(np.log(num_customers)))
obs_dim = 2
sigma_a = 1e1
sigma_x = 1e-10

betas = np.random.beta(a=alpha, b=1, size=num_dishes)
sticks = np.cumprod(betas)

Z = np.stack([np.random.binomial(n=1, p=stick, size=num_customers) for stick in sticks]).T
Z_prior = np.repeat(np.expand_dims(sticks, axis=0), repeats=num_customers, axis=0)
A = np.random.multivariate_normal(
    mean=np.zeros(shape=obs_dim),
    cov=np.square(sigma_a) * np.eye(obs_dim),
    size=num_dishes)
epsilon = np.random.normal(
    loc=0,
    scale=sigma_x,
    size=(num_customers, obs_dim))
X = np.matmul(Z, A) + epsilon

ax = sns.scatterplot(x=X[:, 0], y=X[:, 1])
for dish_idx in range(num_dishes):
    sns.lineplot(x=[0, A[dish_idx, 0]],
                 y=[0, A[dish_idx, 1]],
                 ax=ax,
                 label=f'Dish {dish_idx + 1}',
                 alpha=0.5)
ax.legend()
plt.show()


def squared_error(X: np.ndarray, Z: np.ndarray, A: np.ndarray, sigma_x: float = 1.) -> np.float:
    """
    Compute summed (non-mean) squared error.

    :param X: shape (num samples, obs dim)
    :param Z: shape (num samples, num features)
    :param A: (num features, obs dim)
    :param sigma_x: float
    """
    resid = X - np.matmul(Z, A)
    summed_squared_error = 0.5 * np.trace(resid @ resid.T) / sigma_x
    return summed_squared_error


def l1_regularization(X: np.ndarray) -> np.float:
    """
    Compute |X|_1

    :param X: shape (num samples, obs dim)
    :return:
    """
    abs_X = np.abs(X)
    l1_sum = np.sum(abs_X)
    return l1_sum


def prior_regularization(Z: np.ndarray, Z_prior: np.ndarray, adjustment: float = 1e-8) -> np.float:
    """
    Compute the Bernoulli regularize

    :param Z: shape (num samples, num features
    :param Z_prior: shape (num_samples, num features)
    :param adjustment: used to ensure Z_prior is within (0, 1)
    """
    Z_prior[Z_prior == 0.] += adjustment
    Z_prior[Z_prior == 1.] -= adjustment

    one_minus_Z_prior = 1. - Z_prior
    log_one_minus_Z_prior = np.log(one_minus_Z_prior)
    elementwise_result = np.add(
        np.multiply(Z, np.log(np.divide(Z_prior, one_minus_Z_prior))),
        log_one_minus_Z_prior
    )
    summed_result = np.sum(elementwise_result)
    return summed_result


def loss(Z: np.ndarray,
         X: np.ndarray,
         A: np.ndarray,
         lambda_squared_error: float = 1.,
         lambda_prior_regularization: float = 1.,
         lambda_l1_regularization: float = 1.) -> np.float:
    term_1 = squared_error(X=X, Z=Z, A=A)
    term_2 = prior_regularization(Z=Z, Z_prior=Z_prior)
    term_3 = l1_regularization(X=X)
    return lambda_squared_error * term_1 \
           + lambda_prior_regularization * term_2 \
           + lambda_l1_regularization * term_3


# import scipy.optimize

# Z0 = 0.5 * np.ones_like(Z)
# result = scipy.optimize.least_squares(
#     fun=loss,
#     x0=Z0.flatten(),
#     bounds=(0., 1.),
#     kwargs={'X': X, 'A': A})


import cvxpy as cp


# add parameters for faster refitting
# https://www.cvxpy.org/tutorial/intro/index.html#parameters
# m = cp.Parameter(nonneg=True)

Z_cp = cp.Variable(shape=(num_customers, num_dishes))
one_minus_Z_prior = 1. - Z_prior
log_one_minus_Z_prior = np.log(one_minus_Z_prior)
sse_fn = 0.5*cp.sum_squares(X - Z_cp @ A)
prior_fn = cp.sum(
    cp.multiply(Z_cp, np.log(np.divide(Z_prior, one_minus_Z_prior))) + log_one_minus_Z_prior)
l1_fn = cp.norm1(Z_cp)

constraints = [0 <= Z_cp, Z_cp <= 1]
objectives = {
    'sse': cp.Minimize(cp.sum_squares(X - Z_cp @ A)),
    'sse_prior': cp.Minimize(sse_fn - prior_fn),
    'sse_prior_l1': cp.Minimize(sse_fn - prior_fn + l1_fn),
}

problems = {}
for objective_str, objective_cp in objectives.items():
    prob = cp.Problem(objective=objective_cp, constraints=constraints)
    prob.solve()
    problems[objective_str] = dict(
        prob=prob,
        Zhat=Z_cp.value)
    sse = np.square(np.linalg.norm(Z_cp.value - Z))
    problems[objective_str]['sse'] = sse
    print(f'Objective Str: {objective_str}\t\t\tMSE: {sse}')


print(10)







