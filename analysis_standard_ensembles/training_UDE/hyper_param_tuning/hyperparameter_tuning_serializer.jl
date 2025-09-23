cd(@__DIR__)

using Serialization

###################### LV 1.0 ################################

learning_rate_adam = 0.005
neural_network_dimension = 32
activation_function = 4
regularization = 0
regularization_coefficient_1 = 0.0
regularization_coefficient_2 = 0.0
gain = 1.0
error_level = 1
number_ensembles = 10
number_threads = 1
output_folder = "initialization_difference"
model = 1
hyper_ms_segment = 15
hyper_ms_lambda = 0.1


hyperparameters = (learning_rate_adam, neural_network_dimension, activation_function, regularization, regularization_coefficient_1, regularization_coefficient_2, gain, error_level, number_ensembles, number_threads, output_folder, model, hyper_ms_segment, hyper_ms_lambda)

serialize("hyperparameters_LV_UDE.jld", hyperparameters)

###################### DAMPED 1.0 ################################
learning_rate_adam = 0.01
neural_network_dimension = 32
activation_function = 4
regularization = 0
regularization_coefficient_1 = 0.0
regularization_coefficient_2 = 0.0
gain = 1.0
error_level = 1
number_ensembles = 3
number_threads = 1
output_folder = "initialization_difference"
model = 3
hyper_ms_segment = 5
hyper_ms_lambda = 0.1

hyperparameters = (learning_rate_adam, neural_network_dimension, activation_function, regularization, regularization_coefficient_1, regularization_coefficient_2, gain, error_level, number_ensembles, number_threads, output_folder, model, hyper_ms_segment, hyper_ms_lambda)

serialize("hyperparameters_DAMPED_UDE.jld", hyperparameters)

###################### LORENZ 1.0 ################################
learning_rate_adam = 0.01
neural_network_dimension = 32
activation_function = 4
regularization = 0
regularization_coefficient_1 = 0.00
regularization_coefficient_2 = 0.00
gain = 1.0
error_level = 1
number_ensembles = 3
number_threads = 1
output_folder = "initialization_difference"
model = 2
hyper_ms_segment = 5
hyper_ms_lambda = 10.0

hyperparameters = (learning_rate_adam, neural_network_dimension, activation_function, regularization, regularization_coefficient_1, regularization_coefficient_2, gain, error_level, number_ensembles, number_threads, output_folder, model, hyper_ms_segment, hyper_ms_lambda)

serialize("hyperparameters_LORENZ_UDE.jld", hyperparameters)