# Radar: Deep Learning and Sparse Recovery

Sparse Recovery Project Submission:
Simply run the sparse_recovery_radar_positions_runscript.m. The simulation parameters are at the top, and the input_select.m function allows you to pick a pulse type for reconstruction. Single and multi-target scenarios can be run or commented out at the top of the run script.

Deep Learning Project Submission:

There are five files: test_bench.py, generate_data.py, plot_tools.py, radar_learning_models.py, and run_training_and_testing.py. 

In order to run the code, simply run test_bench.py. Ensure you have all the packages at the top of the file. It is in this file that you can adjust the hyperparameters for training and for running various scenarios. The most important flags are "temporal_flag" and "model_flag". Set model_flag to 0 to train a feedforward linear network and set to 1 to train an LSTM. Set "temporal_flag" to 0 to train and test on single-step random perturbations, and set it to 1 to train and test on continous noise data generated by a Dryden wind model. As should be expected, the LSTM will only work when temporal_flag is set to 1.

Generate_data.py generates either one-step or multi-step datasets depending on the flags given to it. It also contains the code we used for the Dryden wind model.

Plot_tools.py allows you to look at the noise profiles generated by the Dryden wind model, as well as view how torque rotates the array. 

Radar_learning_models.py generates our models based on the hyperparameters from the test bench and has the functions to do training and testing.

Run_training_and_testing.py is the main run script, calling the other functions to generate data, build a model, train it, test it, and print and plot the results.
