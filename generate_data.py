import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import scipy.spatial.distance
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal

### Create classes to hold the datasets depending on type ###

class RadarDataset(Dataset):
    def __init__(self, inputs, targets, radar_array):
        self.inputs = inputs
        self.targets = targets
        self.radar_array = radar_array

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.radar_array[idx]

class OverlappingSequencesRadarDataset(Dataset):
    def __init__(self, inputs, targets, radar_array, window_size):
        self.inputs = inputs
        self.targets = targets
        self.radar_array = radar_array
        self.window_size = window_size

    def __len__(self):
        return len(self.inputs) - self.window_size

    def __getitem__(self, idx):
        input_seq = self.inputs[idx:idx + self.window_size]
        target_seq = self.targets[idx + self.window_size]
        radar_array_seq = self.radar_array[idx + self.window_size]
        
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32), torch.tensor(radar_array_seq, dtype=torch.float32)


############################################################################################

### Generate Wind Path Data ###

def dryden_wind_turbulence(v, L, dt, n_samples):
    T = n_samples * dt
    t = np.arange(0, T, dt)
    std_dev = np.sqrt(2*v/L)
    num = [std_dev]
    den = [L/v, 1]
    system = signal.TransferFunction(num, den)
    d_system = signal.cont2discrete((system.num, system.den), dt, method='bilinear')
    _, w = signal.dlsim(d_system, np.random.normal(0, 1, n_samples))
    return t, np.squeeze(w)

def particle_motion(t, wind, initial_position=0, dt=0.01):
    position = np.zeros_like(wind)
    position[0] = initial_position
    velocity = wind.copy()
    for i in range(1, len(t)):
        position[i] = position[i-1] + velocity[i-1] * dt
    return position

def dryden_wind_turbulence_3d(v, L, dt, n_samples):
    t, wind_x = dryden_wind_turbulence(v, L, dt, n_samples)
    _, wind_y = dryden_wind_turbulence(v, L, dt, n_samples)
    _, wind_z = dryden_wind_turbulence(v, L, dt, n_samples)
    return t, wind_x, wind_y, wind_z

def particle_motion_3d(t, wind_x, wind_y, wind_z, initial_position=(0, 0, 0), dt=0.01):
    position_x = particle_motion(t, wind_x, initial_position[0], dt)
    position_y = particle_motion(t, wind_y, initial_position[1], dt)
    position_z = particle_motion(t, wind_z, initial_position[2], dt)
    return position_x, position_y, position_z

# Calculate the net force on the structure due to the wind
def net_force_on_structure(structure_points, wind_x, wind_y, wind_z):
    force_x = np.sum(wind_x)
    force_y = np.sum(wind_y)
    force_z = np.sum(wind_z)
    return np.array([force_x, force_y, force_z])

# Calculate the torque on the structure from wind
def net_torque_on_structure(structure_points, center_of_mass, wind_x, wind_y, wind_z):
    torque = np.zeros(3)
    for i in range(structure_points.shape[0]):
        r = structure_points[i] - center_of_mass
        f = np.array([wind_x, wind_y, wind_z])
        torque += np.cross(r, f)
    return torque

# Update the position and orientation of the structure
def update_structure(structure_points, center_of_mass, force, torque, dt):
    # Update center of mass
    new_center_of_mass = center_of_mass + force * dt
    
    # Update rotation from torque
    rotation_angle = np.linalg.norm(torque) * dt
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],[np.sin(rotation_angle),  np.cos(rotation_angle), 0], [0, 0, 1]])
    
    # rotate
    new_structure_points = np.dot(structure_points - center_of_mass, rotation_matrix) + new_center_of_mass
    return new_structure_points

def run_wind_model(radar_array, num_samples, plot_flag):
    # Parameters for the Dryden Wind Turbulence Model
    v = 30  # Wind speed (m/s), increased from 10 m/s
    L = 50  # Turbulence scale length m)
    dt = 0.1  # Time step (s)

    # Simulate 3D wind turbulence
    t, wind_x, wind_y, wind_z = dryden_wind_turbulence_3d(v, L, dt, num_samples)

    # Initialize the structure ponts
    structure_points = radar_array
    center_of_mass = np.mean(structure_points, axis=0)
    structure_positions = [structure_points.copy()]

    # Simulate
    for i in range(num_samples):
        # Calculate the net force and torque on the structure
        force = net_force_on_structure(structure_points, wind_x[i], wind_y[i], wind_z[i])
        torque = net_torque_on_structure(structure_points, center_of_mass, wind_x[i], wind_y[i], wind_z[i])

        # Update the structure's position and orientation
        structure_points = update_structure(structure_points, center_of_mass, force, torque, dt)
        structure_positions.append(structure_points.copy())

    if plot_flag:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

        # Subsample
        step = 2000
        for i in range(0, len(structure_positions), step):
            structure_points = structure_positions[i]
            num_points = len(structure_points)
            for j in range(num_points):
                for k in range(j + 1, num_points):
                    ax1.plot(structure_points[[j, k], 0], structure_points[[j, k], 1], structure_points[[j, k], 2], linestyle=':', marker='o')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Structure at Sub-sampled Time Points')

        plt.show()

        # Visualize the path of the first element
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')

        element_index = 0
        element_positions = np.array([position[element_index] for position in structure_positions])
        ax2.plot(element_positions[:, 0], element_positions[:, 1], element_positions[:, 2], linestyle='-', color='red', linewidth=1)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Path of First Element')

        plt.show()

    return structure_positions

############################################################################################

### Generate Radar data ###

# Generate a random radar array and assign a random rotation and translation to it
def generate_radar_array(N):
    radar_array = np.random.rand(N, 3)
    q, r = np.linalg.qr(np.random.rand(3, 3))
    radar_array = np.array(np.matmul(radar_array, q))

    return radar_array

#Create a dataset for only one time step
def create_dataset_one_step(N, T, num_samples, radar_array, noise_flag, target_scaling, perturbation_scaling, rotation_flag):
    # Initialize the large arrays for storing data
    radar_data = np.zeros((N, 3, num_samples));
    radar_data_perturbed = np.zeros((N, 3, num_samples));
    target_data = np.zeros((T, 3, num_samples));
    distances_original = np.zeros((N, T, num_samples));
    distances_perturbed = np.zeros((N, T, num_samples));

    for i in range(num_samples):
        # Create a uniformly randomly generated target array of size T by 3. Ensure it has a maximum dimension value greater than 1 for normalization
        target_array = (np.random.rand(T, 3) -.5) * target_scaling;
        m = np.max(np.abs(target_array))
        while m < 1:
            target_array = np.random.rand(T, 3) * target_scaling;
            m = np.max(target_array)

        # Add a random Gaussian perturbation to the entire radar array (translation)
        if noise_flag == 1:
            translation_perturbation = np.random.randn(1, 3) * perturbation_scaling;
        else:
            translation_perturbation = (np.random.rand(1, 3) - .5) * perturbation_scaling;
        perturbed_radar_array_translation = radar_array + translation_perturbation;

        # Add a random rotation to the radar array
        if rotation_flag == 1: #Completely uniformly random rotation
            q, r = np.linalg.qr(np.random.rand(3, 3))
            r_mean = np.mean(perturbed_radar_array_translation, axis=0) # Make the average point of the radar array the point about which rotation happens, then add the center back
            centered_perturbed_radar_array = perturbed_radar_array_translation - r_mean
            perturbed_radar_array_rotation = np.array(np.matmul(centered_perturbed_radar_array, q)) + r_mean
        elif rotation_flag == 2:
            perturbed_radar_array_rotation = np.zeros((N, 3))
            rotation_scaling = .1
            x_rot = rotation_scaling * np.random.rand()
            y_rot = rotation_scaling * np.random.rand()
            z_rot = rotation_scaling * np.random.rand()
            r = R.from_euler('xyz', [[x_rot, 0, 0], [0, y_rot, 0], [0, 0, z_rot]], degrees = True)
            r_mean = np.mean(perturbed_radar_array_translation, axis=0) # Make the average point of the radar array the point about which rotation happens, then add the center back
            centered_perturbed_radar_array = perturbed_radar_array_translation - r_mean
            for j in range(N):
                perturbed_radar_array_rotation[j, :] = r.apply(centered_perturbed_radar_array[j, :])[2, :] + r_mean
        else:
            perturbed_radar_array_rotation = np.array(perturbed_radar_array_translation) #no rotation

        # Find the original distance matrix before perturbation is applied
        distance_matrix_original = scipy.spatial.distance.cdist(radar_array, target_array, metric='euclidean');

        # Capture the Euclidean distance from each perturbed radar to each target
        distance_matrix_perturbed = scipy.spatial.distance.cdist(perturbed_radar_array_rotation, target_array, metric='euclidean');

        # Store data in large arrays
        radar_data[:, :, i] = radar_array;
        target_data[:, :, i] = target_array;
        distances_original[:, :, i] = distance_matrix_original;
        distances_perturbed[:, :, i] = distance_matrix_perturbed;
        radar_data_perturbed[:, :, i] = perturbed_radar_array_rotation;
        
    return radar_data, radar_data_perturbed, target_data, distances_original, distances_perturbed

#create a dataset for a sequence of points from the wind model
def create_dataset_temporal(N, T, num_samples, radar_array, target_scaling, plot_flag):
    # Initialize the large arrays for storing data
    radar_data = np.zeros((N, 3, num_samples));
    radar_data_perturbed = np.zeros((N, 3, num_samples));
    target_data = np.zeros((T, 3, num_samples));
    distances_original = np.zeros((N, T, num_samples));
    distances_perturbed = np.zeros((N, T, num_samples));

    radar_data = np.asarray(run_wind_model(radar_array, num_samples, plot_flag))
    radar_data = np.transpose(radar_data, (1, 2, 0))

    target_array = (np.random.rand(T, 3) -.5) * target_scaling

    for i in range(num_samples):
        radar_data_perturbed[:, :, i] = radar_data[:, :, i+1]
        
        # Find the original distance matrix before perturbation is applied
        distance_matrix_original = scipy.spatial.distance.cdist(radar_data[:, :, i], target_array, metric='euclidean')  
        
        # Capture the Euclidean distance from each perturbed radar to each target
        distance_matrix_perturbed = scipy.spatial.distance.cdist(radar_data_perturbed[:, :, i], target_array, metric='euclidean');

        # Store data in large arrays
        target_data[:, :, i] = target_array;
        distances_original[:, :, i] = distance_matrix_original;
        distances_perturbed[:, :, i] = distance_matrix_perturbed;
    
        #radar_data[:, :, i] = radar_array
        radar_data[:, :, i] = radar_data[:, :, 0]

    radar_data = radar_data[:, :, :-1] #drop the final point

    return radar_data, radar_data_perturbed, target_data, distances_original, distances_perturbed

#Format the data into our input and output vectors for the model
def generate_inputs_targets(radar_data, target_data, distances_original, distances_perturbed, radar_data_perturbed, N, T, num_samples):
    # Calculate the distance differences
    distances_delta = distances_perturbed - distances_original;
    
    # Concatenate the inputs to one vector
    inputs = np.concatenate((np.reshape(radar_data, (3*N, num_samples)), np.reshape(target_data, (3*T, num_samples)), np.reshape(distances_delta, (N*T, np.size(distances_delta, 2)))), axis=0);
    #inputs = np.concatenate((np.reshape(target_data, (3*T, num_samples)), np.reshape(distances_delta, (N*T, np.size(distances_delta, 2)))), axis=0);

    # Generate the target output vectors
    targets = np.reshape(radar_data_perturbed, [N*3, num_samples]);

    # Export initial radar array position to use for constraints
    radar_data = np.reshape(radar_data, [N*3, num_samples]);

    return inputs, targets, radar_data

# Generate all the data
def generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry=[], rotation_flag=2):
    # Generate radar array
    if len(radar_geometry) > 0:
        radar_array = radar_geometry
    else:
        radar_array = generate_radar_array(N)

    #Generate the dataset using the generated radar array
    if temporal_flag == 0:
        radar_data, radar_data_perturbed, target_data, distances_original, distances_perturbed = create_dataset_one_step(N, T, num_samples, radar_array, noise_flag, target_scaling, perturbation_scaling, rotation_flag)
    else:
        radar_data, radar_data_perturbed, target_data, distances_original, distances_perturbed = create_dataset_temporal(N, T, num_samples, radar_array, target_scaling, plot_flag)

    #Generate inputs and targets
    inputs, targets, radar_data = generate_inputs_targets(radar_data, target_data, distances_original, distances_perturbed, radar_data_perturbed, N, T, num_samples)
    return inputs, targets, radar_data

def load_data(train_ratio, batch_size, model_flag, inputs=[], targets=[], radar_array=[], window_size=[]):
    window_size = 10

    inputs = np.transpose(inputs)
    targets = np.transpose(targets)
    radar_array = np.transpose(radar_array)

    if model_flag == 0:
        dataset = RadarDataset(inputs, targets, radar_array)
    else:
       dataset = OverlappingSequencesRadarDataset(inputs, targets, radar_array, window_size)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def generate_dataloaders(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, train_ratio, batch_size, model_flag, rotation_flag, temporal_flag, plot_flag):

    #Temporal flag = 0 means we are doing one-step predictions
    if temporal_flag == 0:

        if rotation_flag == 0: #No rotation
            inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry=[], rotation_flag=2)
            train_loader, test_loader = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data)
        elif rotation_flag == 1: #Unbounded rotation
            inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry=[], rotation_flag=2)
            train_loader, test_loader = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data)
        elif rotation_flag == 2: #Bounded slight rotation
            inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry=[], rotation_flag=2)
            train_loader, test_loader = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data)
        else: #Combine various rotation styles
            #We need to define a radar geometry outside the function for this task
            radar_geometry = np.random.rand(N, 3)

            inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry, rotation_flag=2)
            train_loader_1, test_loader_1 = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data) #keep the training data with rotations

            inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry, rotation_flag=0)
            train_loader_2, test_loader_2 = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data) #keep the testing data without rotations

            train_loader = train_loader_1
            test_loader = test_loader_2

    #Temporal flag = 1 means we are generating sequences of windy radar arrays
    else: 

        radar_geometry = np.random.rand(N, 3)

        inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry, rotation_flag=2)
        train_loader_1, test_loader_1 = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data) 

        inputs, targets, radar_data = generation_wrapper(N, T, num_samples, noise_flag, target_scaling, perturbation_scaling, plot_flag, temporal_flag, radar_geometry, rotation_flag=2)
        train_loader_2, test_loader_2 = load_data(train_ratio, batch_size, model_flag, inputs, targets, radar_data) 

        train_loader = train_loader_1
        test_loader = test_loader_2

    return train_loader, test_loader