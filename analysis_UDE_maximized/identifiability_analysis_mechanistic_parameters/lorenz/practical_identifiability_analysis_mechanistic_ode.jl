cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase

include("../../ConfidenceEllipse.jl")
using .ConfidenceEllipse

integrator = Vern7()
abstol = 1e-6
reltol = 1e-5
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
maxiters = 10000

################################### loads the data ##############################################
training_data_structure = deserialize("../../data_generator/lorenz_training_data_structure_err_1.jld")

rng = Random.default_rng()
Random.seed!(rng, 0)

original_parameters = Float64[10, 28, 8/3]
#function to generate the Data
function damped_oscillator_ground_truth(du, u, pars, t)
    σ, r, b = original_parameters .* pars
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

tspan = extrema(training_data_structure.solution_dataframes[1].t)
prob_uode_pred = ODEProblem{true}(damped_oscillator_ground_truth, Array(training_data_structure.solution_dataframes[1][1, 2:(end-1)]), tspan)

df = training_data_structure.solution_dataframes[1]
first_traj_df = df[df.traj.==1, :]
initial_condition_trajectory_1 = Array(first_traj_df[1, 2:(end-1)])
df = training_data_structure.solution_dataframes[2]
second_traj_df = df[df.traj.==2, :]
initial_condition_trajectory_2 = Array(second_traj_df[1, 2:(end-1)])
df = training_data_structure.solution_dataframes[3]
third_traj_df = df[df.traj.==3, :]
initial_condition_trajectory_3 = Array(third_traj_df[1, 2:(end-1)])


function model_simulation(θ, t, trajectory, integrator=integrator, sensealg=sensealg, prob_uode_pred=prob_uode_pred)
    if trajectory == 1
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_condition_trajectory_1
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol,
            sensealg=sensealg,
            maxiters=maxiters
        )
    elseif trajectory == 2
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_condition_trajectory_2
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol,
            sensealg=sensealg,
            maxiters=maxiters
        )
    elseif trajectory == 3
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_condition_trajectory_3
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol,
            sensealg=sensealg,
            maxiters=maxiters
        )
    end

    if trajectory_sol.retcode != :Success
        return Inf
    end

    return Array(trajectory_sol)
end

times = training_data_structure.solution_dataframes[1].t
original_times = deepcopy(times)
times = times[1:1:end]

function get_Hessian_not_proportional(parameters_to_consider, times, training_data_structure)

    #first trajectory
    sensitivity_matrix_first_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 1), parameters_to_consider)[1]
    #second trajectory
    sensitivity_matrix_second_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 2), parameters_to_consider)[1]
    #third trajectory
    sensitivity_matrix_third_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 3), parameters_to_consider)[1]

    sensitivity_matrix = vcat(sensitivity_matrix_first_trajectory, sensitivity_matrix_second_trajectory, sensitivity_matrix_third_trajectory)

    multiplicative_factor_array = repeat(training_data_structure.max_oscillations[1], outer=size(times, 1))
    multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[2], outer=size(times)))
    multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[3], outer=size(times)))

    multiplicative_factor_matrix = Diagonal(multiplicative_factor_array)

    hessian = sensitivity_matrix' * multiplicative_factor_matrix * sensitivity_matrix

    return hessian
end

function getEigenDempositionHessianNotProportional(par, times, training_data_structure)
    hessian = get_Hessian_not_proportional(par, times, training_data_structure)
    eigen_decomposition = eigen(Symmetric(hessian))
    return eigen_decomposition
end

#modify the arguments!
eigenDecomposition = getEigenDempositionHessianNotProportional([1.0, 1.0, 1.0], times, training_data_structure)

eigeinvalues = eigenDecomposition.values
eigenvectors = eigenDecomposition.vectors

threshold_on_eigenvalues = 10^(-5)
indices = findall(eigeinvalues .< threshold_on_eigenvalues)

