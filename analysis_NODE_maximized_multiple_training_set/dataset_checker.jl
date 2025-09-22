cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase

################################### loads the data ##############################################
training_data_structure = deserialize("./data_generator/lotka_volterra_in_silico_data_no_noise.jld")
training_data_structure_training = deserialize("../analysis_standard_ensembles/data_generator_traj_2/lotka_volterra_in_silico_data_no_noise.jld")

isequaldf = isequal(training_data_structure, training_data_structure_training)

################################ damped oscillator ###############################################
training_data_structure = deserialize("./data_generator/damped_oscillator_in_silico_data_no_noise.jld")
training_data_structure_training = deserialize("../analysis_standard_ensembles/data_generator_traj_2/damped_oscillator_in_silico_data_no_noise.jld")

isequaldf = isequal(training_data_structure, training_data_structure_training)

################################# lorenz system ####################################################
training_data_structure = deserialize("./data_generator/lorenz_in_silico_data_no_noise.jld")
training_data_structure_training = deserialize("../analysis_standard_ensembles/data_generator_traj_2/lorenz_in_silico_data_no_noise.jld")

isequaldf = isequal(training_data_structure, training_data_structure_training)
