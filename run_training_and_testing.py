import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np

from radar_learning_models import RadarFeedForward, RadarLSTM, train_model, evaluate_model
from generate_data import generate_dataloaders
from plot_tools import plot_3d_targets_and_predictions, scatterplot_mse_comparison

def run_training_and_testing(N, T, target_scaling, perturbation_scaling, model_flag, noise_flag, temporal_flag, rotation_flag, plot_flag, num_samples, train_ratio, batch_size, epochs, learning_rate, hidden_dim, num_layers_ff, dropout):
    # Load the data
    train_loader, test_loader = generate_dataloaders(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, train_ratio, batch_size, model_flag, rotation_flag, temporal_flag, plot_flag)

    # Initialize model
    input_dim = train_loader.dataset.dataset.inputs.shape[1]
    target_dim = train_loader.dataset.dataset.targets.shape[1]
    num_targets = int(target_dim / 3)
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_flag == 0:
        model = RadarFeedForward(input_dim, target_dim, hidden_dim, num_layers_ff, dropout).to(dtype).to(device)
    else: 
        model = RadarLSTM(input_dim, hidden_dim, target_dim, num_layers_ff)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, num_targets)

    # Evaluate the model
    predictions, targets, radar_array = evaluate_model(model, test_loader, device)

    # Find the percent reduction in error
    radar_array_mat = np.reshape(radar_array, (radar_array.shape[0], num_targets, 3))
    predictions_mat = np.reshape(predictions, (predictions.shape[0], num_targets, 3))
    targets_mat = np.reshape(targets, (targets.shape[0], num_targets, 3))
    original_mse_array = np.sum((np.sum((radar_array_mat - targets_mat)**2, axis=2)), axis=1)
    updated_mse_array = np.sum((np.sum((predictions_mat - targets_mat)**2, axis=2)),axis=1)
    original_mse = np.mean(original_mse_array) #Average of the sum of the MSE for the entire radar array
    updated_mse = np.mean(updated_mse_array)
    percent_reduction = (1 - updated_mse / original_mse) * 100
    percent_reduction_string = "{:.2f}".format(percent_reduction)
    print('Percentage Reduction in Displacement MSE after Training and Prediction: ' + str(percent_reduction_string) + '%')

    # Scatterplot of original vs. new MSE
    scatterplot_mse_comparison(original_mse_array, updated_mse_array)

    # Plot examples
    sample_idx = [0, 1, 2] 
    for i in range(len(sample_idx)):
        plot_3d_targets_and_predictions(targets_mat, predictions_mat, radar_array_mat, sample_idx[i], num_targets)