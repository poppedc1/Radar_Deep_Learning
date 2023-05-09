import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatterplot_mse_comparison(x, y):
    plt.scatter(x, y, s=5)
    max_lim = np.max(x)
    plt.xlim([0, max_lim])
    plt.ylim([0, max_lim])
    plt.show()

def plot_3d_targets_and_predictions(targets, predictions, radar_array, sample_idx, num_targets):

    target_points = targets[sample_idx, :, :]
    predicted_points = predictions[sample_idx, :, :]
    radar_points = radar_array[sample_idx, :, :]

    markers = ['o', 's', '^', 'D', 'X']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot true perturbed radar location
    for i in range(num_targets):
        ax.scatter(target_points[i, 0], target_points[i, 1], target_points[i, 2], c='b', marker=markers[i], label=f'Target {i+1}')

    # Plot predicted radar location
    for i in range(num_targets):
        ax.scatter(predicted_points[i, 0], predicted_points[i, 1], predicted_points[i, 2], c='r', marker=markers[i], label=f'Prediction {i+1}')
        
    # Plot the original radar array
    for i in range(num_targets):
        ax.scatter(radar_points[i, 0], radar_points[i, 1], radar_points[i, 2], c='g', marker=markers[i], label=f'Original Radar Position {i+1}')

    # Connect everything with a bunch of dotted lines
    for i in range(num_targets):
        for j in range(i+1, num_targets):
            ax.plot([target_points[i, 0], target_points[j, 0]], [target_points[i, 1], target_points[j, 1]], [target_points[i, 2], target_points[j, 2]], 'b--')

    for i in range(num_targets):
        for j in range(i+1, num_targets):
            ax.plot([predicted_points[i, 0], predicted_points[j, 0]], [predicted_points[i, 1], predicted_points[j, 1]], [predicted_points[i, 2], predicted_points[j, 2]], 'r--')

    for i in range(num_targets):
        for j in range(i+1, num_targets):
            ax.plot([radar_points[i, 0], radar_points[j, 0]], [radar_points[i, 1], radar_points[j, 1]], [radar_points[i, 2], radar_points[j, 2]], 'g--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Predicted vs. Perturbed vs. Original Radar Array Position')

    plt.show()