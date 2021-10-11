import matplotlib.pyplot as plt
import numpy as np
import os


def plot_images_in_clusters(inference_alg_str: str,
                            concentration_param: float,
                            images: np.ndarray,
                            table_assignment_posteriors: np.ndarray,
                            table_assignment_posteriors_running_sum: np.ndarray,
                            plot_dir):

    table_indices = np.arange(len(table_assignment_posteriors))

    # as a heuristic
    confident_class_predictions = table_assignment_posteriors > 0.95
    summed_confident_predictions_per_table = np.sum(confident_class_predictions, axis=0)

    plt.plot(table_indices, table_assignment_posteriors_running_sum[-1, :], label='Total Prob. Mass')
    plt.plot(table_indices, summed_confident_predictions_per_table, label='Confident Predictions')
    plt.ylabel('Prob. Mass at Table')
    plt.xlabel('Table Index')
    plt.xlim(0, 150)
    plt.legend()

    plt.savefig(os.path.join(plot_dir,
                             '{}_alpha={:.2f}_mass_per_table.png'.format(inference_alg_str,
                                                                         concentration_param)),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    table_indices_by_decreasing_summed_prob_mass = np.argsort(
        table_assignment_posteriors_running_sum[-1, :])[::-1]

    num_rows = 6
    num_images_per_table = 11
    num_rows = num_rows
    num_cols = num_images_per_table
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             sharex=True,
                             sharey=True)
    axes[0, 0].set_title(f'Cluster Means')
    axes[0, int(num_images_per_table / 2)].set_title('Observations')

    for row_idx in range(num_rows):

        table_idx = table_indices_by_decreasing_summed_prob_mass[row_idx]

        # plot table's mean parameters
        # axes[row_idx, 0].imshow(pca_proj_means[table_idx], cmap='gray')
        axes[row_idx, 0].set_ylabel(f'Cluster: {1 + row_idx}',
                                    rotation=0,
                                    labelpad=40)
        # axes[row_idx, 0].axis('off')

        # images_at_table = images[confident_class_predictions[:, table_idx]]
        posteriors_at_table = table_assignment_posteriors[:, table_idx]
        customer_indices_by_decreasing_prob_mass = np.argsort(
            posteriors_at_table)[::-1]

        for image_num in range(num_images_per_table):
            customer_idx = customer_indices_by_decreasing_prob_mass[image_num]
            customer_mass = posteriors_at_table[customer_idx]
            # only plot high confidence
            # if customer_mass < 0.9:
            #
            # else:
            try:
                if customer_mass < 0.4:
                    axes[row_idx, image_num].axis('off')
                else:
                    axes[row_idx, image_num].imshow(images[customer_idx], cmap='gray')
                    axes[row_idx, image_num].set_title(f'{np.round(customer_mass, 2)}')
            except IndexError:
                axes[row_idx, image_num].axis('off')

    # remove tick labels
    plt.setp(axes, xticks=[], yticks=[])
    plt.savefig(os.path.join(plot_dir, '{}_alpha={:.2f}_images_per_table.png'.format(inference_alg_str,
                                                                                     concentration_param)),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()