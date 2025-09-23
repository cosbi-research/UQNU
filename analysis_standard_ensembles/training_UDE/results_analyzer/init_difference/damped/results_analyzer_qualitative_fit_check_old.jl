cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using PlotlyJS

experimental_data = deserialize("../../../data_generator/damped_oscillator_in_silico_data_no_noise.jld")
trajectories = unique(experimental_data.traj)

experimental_data_1 = experimental_data[experimental_data.traj .== trajectories[1], :]
experimental_data_2 = experimental_data[experimental_data.traj .== trajectories[2], :]
experimental_data_3 = experimental_data[experimental_data.traj .== trajectories[3], :]

experimental_datas = [experimental_data_1, experimental_data_2, experimental_data_3]

max_oscillations_1 = [maximum(experimental_data_1.x1) - minimum(experimental_data_1.x1), maximum(experimental_data_1.x2) - minimum(experimental_data_1.x2)]
max_oscillations_2 = [maximum(experimental_data_2.x1) - minimum(experimental_data_2.x1), maximum(experimental_data_2.x2) - minimum(experimental_data_2.x2)]
max_oscillations_3 = [maximum(experimental_data_3.x1) - minimum(experimental_data_3.x1), maximum(experimental_data_3.x2) - minimum(experimental_data_3.x2)]

max_oscillations = [max_oscillations_1, max_oscillations_2, max_oscillations_3]

initial_time_training = 0.0f0
end_time_training = 25.0f0

integrator = Tsit5()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state
            û = appr_neural_network(u, p, st)[1]
            @inbounds du[1] = û[1]
            @inbounds du[2] = û[2]
        end
end

results_1 = deserialize("ensemble_results_model_3_gain_1.0_reg_0.jld")
results_01 = deserialize("ensemble_results_model_3_gain_0.1_reg_0.jld")
results_001 = deserialize("ensemble_results_model_3_gain_0.01_reg_0.jld")

#computes the likelihood in the additive noise model
function compute_likelihood_on_trajectory(experimental_data_tmp, model_simulation, max_oscillations_tmp)

    # estimate sigma for the likelihood
    residuals_first = (experimental_data_tmp.x1 .- model_simulation[1, :]) / max_oscillations_tmp[1]
    residuals_second = (experimental_data_tmp.x2 .- model_simulation[2, :]) / max_oscillations_tmp[2]

    residuals_total = [residuals_first residuals_second]
    alpha_estimate = std(residuals_total)
    sigma_1 = alpha_estimate * max_oscillations_tmp[1]
    sigma_2 = alpha_estimate * max_oscillations_tmp[2]

    # compute the loglikelihood assuming non correlation between the variables 
    n = length(experimental_data.t)

    #loglikelihhhood first variable 
    loglikelihood_1 = -n / 2 * log(2 * pi) - n * log(sigma_1) - 1 / (2 * sigma_1^2) * sum((experimental_data_tmp.x1 .- model_simulation[1, :]) .^ 2)
    #loglikelihood second variable
    loglikelihood_2 = -n / 2 * log(2 * pi) - n * log(sigma_2) - 1 / (2 * sigma_2^2) * sum((experimental_data_tmp.x2 .- model_simulation[2, :]) .^ 2)

    return loglikelihood_1 + loglikelihood_2
end

# threashold based on chi squared distribution
function chi_squared_threshold(df, alpha)
    dist = Chisq(df)
    return quantile(dist, 1 - alpha)
end

df = 1  # degrees of freedom
alpha = 0.01  # significance level
threshold = chi_squared_threshold(df, alpha)

############################################################### early stopping ###################################################################

function analyze_results(results, extended_name, validation_likelihood_threshold)

    validation_likelihoods = [result.validation_likelihood for result in results if result.status == "success"]

    in_dim = 2
    out_dim = 2
    #approximating neural network
    neural_network_dimension = 32
    activation_function_fun = gelu

    my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)
    approximating_neural_network = Lux.Chain(
        Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
        Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
        Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform),
    )

    seed = 0
    local_rng = StableRNG(seed)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)

    tspan = [initial_time_training, end_time_training]
    uode_derivative_function = get_uode_model_function(approximating_neural_network, st)
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, [0.0, 0.0], tspan)

    #plot the dynamics obtained
    simulations_1 = []
    simulations_2 = []
    simulations_3 = []
    likelihoods = []
    indexes = []
    seeds = []
    #TODO remove
    for i in 1:length(results)
        result = results[i]
        if result.status == "success"
            status = result.net_status
            parameters = result.training_res

            tmp_likelihoods = [0.0, 0.0, 0.0]
            for traj in trajectories
                node_parameters = parameters.p
                node_initial_time = parameters.u0[traj, :, 1]

                prob = remake(
                    prob_uode_pred;
                    p=node_parameters,
                    tspan=tspan,
                    u0=node_initial_time
                    #u0 = original_u0#u0=ode_data[:, 1]
                )
                solutions = solve(prob, integrator, saveat=0.001, abstol=abstol, reltol=reltol, verbose=false)

                solutions_at_experimental_times = Array(solve(prob, integrator, p=node_parameters, saveat=experimental_datas[traj].t, abstol=abstol, reltol=reltol,
                    sensealg=sensealg, verbose=false))
                #
                simulated = solutions_at_experimental_times
                likelihood = compute_likelihood_on_trajectory(experimental_datas[traj], simulated, max_oscillations[traj])

                tmp_likelihoods[traj] = likelihood

                if traj == 1
                    push!(simulations_1, solutions)
                elseif traj == 2
                    push!(simulations_2, solutions)
                else
                    push!(simulations_3, solutions)
                end
            end

            push!(likelihoods, tmp_likelihoods[1] + tmp_likelihoods[2] + tmp_likelihoods[3])
            push!(indexes, i)
            push!(seeds, result.seed)
        end
    end

    likelihoods_tmp = copy(likelihoods)

    #get the models to be considered for the ensemble_results
    likelihood_to_keep = validation_likelihoods.< validation_likelihood_threshold
    gr()

    plts = [Plots.plot(title = "First variable traj 1"), Plots.plot(title = "Second variable traj 1")]
    #print the experimental data 
    Plots.scatter!(plts[1], experimental_data_1.t, experimental_data_1.x1, label="Prey", color="green")
    Plots.scatter!(plts[2], experimental_data_1.t, experimental_data_1.x2, label="Predator", color="green")

    indexes_to_keep = []
    for i in 1:length(simulations_1)

        if likelihood_to_keep[i]
            simulation = simulations_1[i]
            likelihood = likelihoods[i]
            index = indexes[i]

            push!(indexes_to_keep, index)

            name_ensamble_ind = string(index) *", loglikelihood "*string(round(likelihood))


            #if likelihood_to_keep[i] 
            Plots.plot!(plts[1], simulation.t, [point[1] for point in simulation.u], lw=3, label=false, color="blue", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
            Plots.plot!(plts[2], simulation.t, [point[2] for point in simulation.u], lw=3, label=false, color="red", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
        end
    end
    plt = Plots.plot(plts[1], plts[2], layout=(2, 1), size=(1000, 500), xlabel="", ylabel="Population", legend=:bottomright)
    Plots.savefig(plt, "damped_ensemble_"*extended_name*"_traj_1.png")

    plts = [Plots.plot(title = "First variable traj 2"), Plots.plot(title = "Second variable traj 2")]
    #print the experimental data 
    Plots.scatter!(plts[1], experimental_data_2.t, experimental_data_2.x1, label="Prey", color="green")
    Plots.scatter!(plts[2], experimental_data_2.t, experimental_data_2.x2, label="Predator", color="green")

    indexes_to_keep = []
    for i in 1:length(simulations_2)

        if likelihood_to_keep[i]
            simulation = simulations_2[i]
            likelihood = likelihoods[i]
            index = indexes[i]

            push!(indexes_to_keep, index)

            name_ensamble_ind = string(index) *", loglikelihood "*string(round(likelihood))


            #if likelihood_to_keep[i] 
            Plots.plot!(plts[1], simulation.t, [point[1] for point in simulation.u], lw=3, label=false, color="blue", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
            Plots.plot!(plts[2], simulation.t, [point[2] for point in simulation.u], lw=3, label=false, color="red", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
        end
    end
    plt = Plots.plot(plts[1], plts[2], layout=(2, 1), size=(1000, 500), xlabel="", ylabel="Population", legend=:bottomright)
    Plots.savefig(plt, "damped_ensemble_"*extended_name*"_traj_2.png")

    plts = [Plots.plot(title = "First variable traj 3"), Plots.plot(title = "Second variable traj 3")]
    #print the experimental data 
    Plots.scatter!(plts[1], experimental_data_3.t, experimental_data_3.x1, label="Prey", color="green")
    Plots.scatter!(plts[2], experimental_data_3.t, experimental_data_3.x2, label="Predator", color="green")

    indexes_to_keep = []
    for i in 1:length(simulations_3)

        if likelihood_to_keep[i]
            simulation = simulations_3[i]
            likelihood = likelihoods[i]
            index = indexes[i]

            push!(indexes_to_keep, index)

            name_ensamble_ind = string(index) *", loglikelihood "*string(round(likelihood))


            #if likelihood_to_keep[i] 
            Plots.plot!(plts[1], simulation.t, [point[1] for point in simulation.u], lw=3, label=false, color="blue", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
            Plots.plot!(plts[2], simulation.t, [point[2] for point in simulation.u], lw=3, label=false, color="red", name=name_ensamble_ind, hoverinfo="name",  hoverlabel=attr(
                namelength=-1,
                padding=attr(t=20, b=20, l=20, r=20)))
        end
    end
    plt = Plots.plot(plts[1], plts[2], layout=(2, 1), size=(1000, 500), xlabel="", ylabel="Population", legend=:bottomright)
    Plots.savefig(plt, "damped_ensemble_"*extended_name*"_traj_3.png")

    return likelihoods_tmp, validation_likelihoods
end

# analyze the results
likelihoods_1, validation_likelihoods_1 = analyze_results(results_1, "gain_1", 0.01)
likelihoods_01, validation_likelihoods_01 = analyze_results(results_01, "gain_01", 0.01)
likelihoods_001, validation_likelihoods_001 = analyze_results(results_001, "gain_001", 0.01)

#compares the likelihoods
min_value = minimum([minimum(log10.(validation_likelihoods_1)), minimum(log10.(validation_likelihoods_01)), minimum(log10.(validation_likelihoods_001))])
max_value = maximum([maximum(log10.(validation_likelihoods_1)), maximum(log10.(validation_likelihoods_01)), maximum(log10.(validation_likelihoods_001))])
bin_width = (max_value - min_value) / 50
bins = min_value:bin_width:max_value

plt = Plots.histogram(log10.(validation_likelihoods_1), bins=bins, label="gain 1.0", xlabel="Validation likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.5)
Plots.histogram!(plt, log10.(validation_likelihoods_01), bins=bins, label="gain 0.1", xlabel="Validation likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.5)
Plots.histogram!(plt, log10.(validation_likelihoods_001), bins=bins, label="gain 0.01", xlabel="Validation likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.5)
Plots.savefig(plt, "validation_likelihood_comparison.png")

min_value = minimum([minimum(log10.(likelihoods_1[likelihoods_1.>0])), minimum(log10.(likelihoods_01[likelihoods_01.>0])), minimum(log10.(likelihoods_001[likelihoods_001.>0]))])
max_value = maximum([maximum(log10.(likelihoods_1[likelihoods_1.>0])), maximum(log10.(likelihoods_01[likelihoods_01.>0])), maximum(log10.(likelihoods_001[likelihoods_001.>0]))])
bin_width = (max_value - min_value) / 50
bins = min_value:bin_width:max_value

plt = Plots.histogram(log10.(likelihoods_1[likelihoods_1.>0]), bins=bins, label="gain 1", xlabel="Likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.8)
Plots.histogram!(plt, log10.(likelihoods_01[likelihoods_01.>0]), bins=bins, label="gain 0.1", xlabel="Likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.8)
Plots.histogram!(plt, log10.(likelihoods_001[likelihoods_001.>0]), bins=bins, label="gain 0.01", xlabel="Likelihood", ylabel="Frequency", title="Validation likelihood distribution", alpha=0.8)
Plots.savefig(plt, "likelihood_comparison.png")