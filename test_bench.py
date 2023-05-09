from run_training_and_testing import run_training_and_testing

#You will need these packages to run the code:
# numpy
# torch
# scipy
# matplotlib
# mpl_toolkits.mplot3d

### Hyperparmameters ###

# Training parameters
num_samples = 10000
train_ratio = 0.9
batch_size = 256
epochs = 100
learning_rate = 0.001

# Model parameters
model_flag = 0 #0 for feedforward, 1 for LSTM. Note that lstm w/ temporal_flag = 0 will be useless because data is formatted as one=steps
hidden_dim = 128
num_layers_ff = 4
dropout = 0.1

# Scenario parameters
N = 4 #Number of radars
T = 1 #Number of targets
temporal_flag = 0 #0 to use one-step prediction with random targets, 1 to use continuous wind model for motion tracking

# One-Step prediction parameters (FF only - set model_flag = 0)
target_scaling = 5 #Ratio of maximum target location to maximum radar array location (1)
perturbation_scaling = 2 #Ratio of maximum perturbation to maximum radar array location (1) for uniform noise, otherwise is just the variance
noise_flag = 0 #0 is uniform, 1 is gaussian (not recommended)
rotation_flag = 1 #0 to train and test without rotations, 1 to train and test with rotations, 2 to train with rotations and test without (py_flag = 1 only)

# Multi-Step prediction paramters (LSTM or FF, model_flag = 0 or 1)
plot_flag = 0

run_training_and_testing(N, T, target_scaling, perturbation_scaling, model_flag, noise_flag, temporal_flag, rotation_flag, plot_flag, num_samples, train_ratio, batch_size, epochs, learning_rate, hidden_dim, num_layers_ff, dropout)
