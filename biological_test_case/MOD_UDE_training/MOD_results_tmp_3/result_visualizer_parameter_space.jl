cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, DiffEqFlux, Distributions


neural_network_dimension = 32
activation_function = 4

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-4
reltol = 1e-5
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#load the data and split into training and validation
datafile = "../../data_generator/cell_apoptosis_silico_data.jld"
in_dim = 6
out_dim = 1

observables = [4,]

include("../../cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../../cell_apoptosis_settings/cell_apop_model_settings.jl")

cd(@__DIR__)

#### DEATH CONDITION ANALYSIS  ######

naive_ODE_parameters = []
mod_ODE_parameters = []
initial_MOD_parameters = []

for ensemble_index in 1:10

  global solution_dataframes
  global max_oscillations


  #load the training results 
  results = deserialize("result_ca$ensemble_index/results.jld")
  MOD_trajectories = results.trajectories
  MOD_ensemble = results.ensemble_original
  
  if MOD_ensemble == nothing
    println("No MOD ensemble found for ensemble index $ensemble_index")
    continue
  end

  naive_ensemble = results.naive_ensemble_reference

  ode_par_naive = []
  for naive_model in naive_ensemble 
    push!(ode_par_naive, naive_model.ode_par)
  end

  mod_trajectories = []
  for MOD_trajectory in MOD_trajectories
    push!(mod_trajectories, MOD_trajectory)
  end

  push!(initial_MOD_parameters, MOD_ensemble[1].ode_par)

  push!(naive_ODE_parameters, ode_par_naive)
  push!(mod_ODE_parameters, mod_trajectories)
end


#plot single parameters 
plot_parameters = []

for i in 1:9
    tmp_plot = Plots.plot(title="Parameter $i", xlabel="Epoch", ylabel="Parameter value")

    #plot a horizontal line on the value of the naive parameters 
    for j in 1:length(naive_ODE_parameters)
        naive_ensemble_parameter = naive_ODE_parameters[j]
        for parameterization in naive_ensemble_parameter
             Plots.hline!(tmp_plot, [parameterization[i]], color=:red, label="")
        end
    end

    for MOD_index in 1:10
        MOD_original_ensemble_parameter = initial_MOD_parameters[MOD_index]
        MOD_ensemble_parameter = mod_ODE_parameters[MOD_index]
        for trajectory in MOD_ensemble_parameter

              trajectory_parameter =  []
              epochs = 1:(length(trajectory)-1)
              for parameter_step in trajectory[1:(end-1)]
                tmp_ode_par = parameter_step.ode_par[i] * MOD_original_ensemble_parameter[i]
                push!(trajectory_parameter, tmp_ode_par)
              end
              Plots.plot!(tmp_plot, epochs, trajectory_parameter, color=:blue, label="")
        end
    end

    push!(plot_parameters, tmp_plot)
end