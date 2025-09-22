cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase

include("../../ConfidenceEllipse.jl")
using .ConfidenceEllipse

include("../../configurations_lv.jl")

################################### loads the data ##############################################
training_data_structure = deserialize("../../data_generator/lotka_volterra_training_data_structure_err_1.jld")

rng = Random.default_rng()
Random.seed!(rng, 0)

#bounding box of interest 
boundig_box_vect_field = deserialize("../../data_generator/lotka_volterra_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../../data_generator/lotka_volterra_in_silico_data_no_noise.jld")

#load the result of the single-parameter training
trained_ensemble = deserialize("../../training_UDE_results/lv/ensemble_results_model_1_with_seed.jld")
#get one standard model only for the u0
single_parameter_training = trained_ensemble[1]
initial_states = deepcopy(single_parameter_training.training_res.u0)


in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../../results_maximized/lv/result_lv$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(ensembles, tmp_ensemble)
end

projections_α_null_space = []
projections_δ_null_space = []

maxiters = 1000

for ensemble in ensembles
    for model in ensemble
        original_α = model.α
        original_δ = model.δ

        model.α = 1.0
        model.δ = 1.0

        original_parameters_ude = [original_α, original_δ]

        p_net = model.p_net
        tmp, st = Lux.setup(rng, approximating_neural_network)

        tspan = extrema(training_data_structure.solution_dataframes[1].t)

        uode_derivative_function = get_uode_model_function(approximating_neural_network, st, original_parameters_ude)

        prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(training_data_structure.solution_dataframes[1][1, 2:(end-1)]), tspan)

        function model_simulation(θ, t, trajectory, initial_states, integrator=integrator, sensealg=sensealg, prob_uode_pred=prob_uode_pred)
            if trajectory == 1
                trajectory_sol = solve(
                    remake(
                        prob_uode_pred;
                        p=θ,
                        tspan=extrema(t),
                        u0=initial_states[1, :, 1]
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
                        u0=initial_states[2, :, 1]
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
                        u0=initial_states[3, :, 1]
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

        function get_Hessian_not_proportional(parameters_to_consider, times, initial_states, training_data_structure)

            #first trajectory
            sensitivity_matrix_first_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 1, initial_states), parameters_to_consider)[1]
            #second trajectory
            sensitivity_matrix_second_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 2, initial_states), parameters_to_consider)[1]
            #third trajectory
            sensitivity_matrix_third_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 3, initial_states), parameters_to_consider)[1]

            sensitivity_matrix = vcat(sensitivity_matrix_first_trajectory, sensitivity_matrix_second_trajectory, sensitivity_matrix_third_trajectory)

            multiplicative_factor_array = repeat(training_data_structure.max_oscillations[1], outer=size(times, 1))
            multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[2], outer=size(times)))
            multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[3], outer=size(times)))

            multiplicative_factor_matrix = Diagonal(multiplicative_factor_array)

            hessian = sensitivity_matrix' * multiplicative_factor_matrix * sensitivity_matrix

            return hessian
        end

        function getEigenDempositionHessianNotProportional(par, times, initial_states, training_data_structure)
            hessian = get_Hessian_not_proportional(par, times, initial_states, training_data_structure)
            eigen_decomposition = eigen(Symmetric(hessian))
            return eigen_decomposition
        end

        #modify the arguments!
        eigenDecomposition = getEigenDempositionHessianNotProportional(model, times, initial_states, training_data_structure)

        eigeinvalues = eigenDecomposition.values
        eigenvectors = eigenDecomposition.vectors


        threshold_on_eigenvalues = 10^(-5)
        indices = findall(eigeinvalues .< threshold_on_eigenvalues)

        null_space_eigenvectors = eigenvectors[:, indices]

        #get the projection of the last parameters on the null space
        projection_α_null_space = nothing
        projection_δ_null_space = nothing
        for i in 1:size(null_space_eigenvectors, 2)

            eigenvector = null_space_eigenvectors[:, i]


            tmp_α = eigenvector[end-1] .* eigenvector
            tmp_δ = eigenvector[end] .* eigenvector

            if isnothing(projection_α_null_space)
                projection_α_null_space = tmp_α
                projection_δ_null_space = tmp_δ
            else
                projection_α_null_space .+= tmp_α
                projection_δ_null_space .+= tmp_δ
            end
        end


        #get the norms of the projections 
        projection_α_null_space = norm(projection_α_null_space)
        projection_δ_null_space = norm(projection_δ_null_space)

        push!(projections_α_null_space, projection_α_null_space)
        push!(projections_δ_null_space, projection_δ_null_space)
    end
end

#plot the histogram of the projections_α_null_space
binwidth = 0.05
StatsPlots.histogram(projections_α_null_space, bins=0:binwidth:1, title="", xlabel="Projection on the null space of α", ylabel="Frequency", legend=false, dpi=400)
#VLINE DASHED RED FOR THE THRESHOLD 0.05

# X AXIS BETWEEN 0 AND 1
xlims!(0, 1)
vline!([0.05], color=:red, linestyle=:dash, label="", linewidth=2)
Plots.savefig("histogram_projections_α_null_space.svg")


#plot the histogram of the projections_δ_null_space
StatsPlots.histogram(projections_δ_null_space, bins=0:binwidth:1, title="", xlabel="Projection on the null space of δ", ylabel="Frequency", legend=false, dpi=400)
#VLINE DASHED RED FOR THE THRESHOLD 0.05
# X AXIS BETWEEN 0 AND 1
xlims!(0, 1)
vline!([0.05], color=:red, linestyle=:dash, label="", linewidth=2)
Plots.savefig("histogram_projections_δ_null_space.svg")