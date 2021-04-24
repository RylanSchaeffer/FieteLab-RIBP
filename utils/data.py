import numpy as np


def sample_sequence_from_ibp(T: int,
                             alpha: float):

    # shape: (number of customers, number of dishes)
    # heuristic: 10 * expected number
    max_dishes = int(10 * alpha * np.sum(1 / (1 + np.arange(T))))
    customers_dishes_draw = np.zeros(shape=(T, max_dishes), dtype=np.int)

    current_num_dishes = 0
    for t in range(T):

        # sample old dishes for new customer
        frac_prev_customers_sampling_dish = np.sum(customers_dishes_draw[:t, :], axis=0) / (t + 1)
        dishes_for_new_customer = np.random.binomial(n=1, p=frac_prev_customers_sampling_dish[np.newaxis, :])[0]
        customers_dishes_draw[t, :] = dishes_for_new_customer.astype(np.int)

        # sample number of new dishes for new customer
        # add +1 to t because of 0-based indexing
        num_new_dishes = np.random.poisson(alpha / (t + 1))
        customers_dishes_draw[t, current_num_dishes:current_num_dishes + num_new_dishes] = 1

        # increment current num dishes
        current_num_dishes += num_new_dishes

    return customers_dishes_draw


vectorized_sample_sequence_from_ibp = np.vectorize(sample_sequence_from_ibp,
                                                   otypes=[np.ndarray])
