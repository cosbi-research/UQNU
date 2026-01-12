cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, DiffEqFlux


neural_network_dimension = 32
activation_function = 4

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#load the training results 
results = deserialize("cell_apop_UDE_results/ensemble_results.jld")


#load the data and split into training and validation
datafile = "../data_generator/cell_apoptosis_silico_data.jld"
in_dim = 6
out_dim = 1

include("../cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../cell_apoptosis_settings/cell_apop_model_settings.jl")

upper_parameter_boundaries = original_parameters * 2.0
lower_parameter_boundaries = original_parameters * 0.5
original_ude_parameters = ones(length(original_parameters))

solutions_dataframe = deserialize(datafile)
training_data = Array(solutions_dataframe[!, 2:(end)])'
training_datas = [training_data,]

variable_plots = []
losses = []

for res in results

  global variable_plots

  if res.status != "success"
    continue
  end

  println("Visualizing result with loss: ", res)
  ########################################## NODE neural network ############################################################
  activation_function_fun = gelu
  my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=1.0)
  approximating_neural_network = Lux.Chain(
    Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  )
  max_oscillations = [maximum(training_dataframe[1:end, i]) - minimum(training_dataframe[1:end, i]) for i in 2:(size(training_dataframe, 2))]

  solution_dataframes = [solutions_dataframe,]
  training_dataframes = [training_dataframe,]
  validation_dataframes = [validation_dataframe,]
  max_oscillations = [max_oscillations,]

  observables = [4,]

  local_rng = StableRNG(0)
  p_net, st = Lux.setup(local_rng, approximating_neural_network)

  tspan = extrema(solution_dataframes[1].t)
  uode_derivative_function = get_uode_model_function(approximating_neural_network, st, lower_parameter_boundaries, upper_parameter_boundaries)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solution_dataframes[1, 2:(end)]), tspan)

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, pred, max_oscillation)
    original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
    return 1 / size(data, 2) * original_cost
  end

  θ = res.training_res
  sol = Array(solve(
    remake(
      prob_uode_pred;
      p=θ.p,
      tspan=tspan,
      u0=max.(θ.u0[1, :, 1], 0.0)
      #u0=u0_original[i, :, 1]
    ),
    integrator;
    saveat=solutions_dataframe.t,
    reltol=reltol,
    abstol=abstol,
    sensealg=sensealg,
    verbose=true
  ))

  loss = loss_function(training_datas[1], sol, max_oscillations[1])
  push!(losses, loss)

  #plot the results towards the training data 
  if length(variable_plots) == 0
    for var_index in 1:size(training_datas[1], 1)

      p = Plots.scatter(
        solutions_dataframe.t,
        solutions_dataframe[:, var_index + 1],
        label="training data",
        lw=2,
        xlabel="time",
        ylabel="concentration",
        title="Variable $(var_index)",
      )
      plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend =nothing)
      #range 0-1 for y axis
      y_min = minimum(solutions_dataframe[:, var_index + 1])
      y_max = maximum(solutions_dataframe[:, var_index + 1])
      y_range = y_max - y_min
      y_lower = y_min - 0.1 * y_range
      y_upper = y_max + 0.1 * y_range
      Plots.plot!(p, ylims = (y_lower, y_upper))
      push!(variable_plots, p)
    end
  else
    for var_index in 1:size(training_datas[1], 1)
      p = variable_plots[var_index]
      plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend =nothing)
            y_min = minimum(solutions_dataframe[:, var_index + 1])
      y_max = maximum(solutions_dataframe[:, var_index + 1])
      y_range = y_max - y_min
      y_lower = y_min - 0.1 * y_range
      y_upper = y_max + 0.1 * y_range
      Plots.plot!(p, ylims = (y_lower, y_upper))
    end
  end
end

#plot in a grid layout
Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
#circle the forth variable which is the observable
savefig("cell_apop_UDE_results/variable_plots.png")


#trying and simulate the model on the other initial conditions (cell death)
