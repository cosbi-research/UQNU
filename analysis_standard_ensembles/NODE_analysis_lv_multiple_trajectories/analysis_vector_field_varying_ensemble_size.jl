cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

result_image_folder = "ensemble_diagnostic_varying_vector_field"
if !isdir(result_image_folder)
    mkdir(result_image_folder)
end

include("../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/lotka_volterra_in_silico_data_bounding_box.jld")

experimental_data_traj_1 = deserialize("../data_generator_traj_1/lotka_volterra_in_silico_data_no_noise.jld")
experimental_data_traj_2 = deserialize("../data_generator_traj_2/lotka_volterra_in_silico_data_no_noise.jld")
experimental_data_traj_3 = deserialize("../data_generator_traj_3/lotka_volterra_in_silico_data_no_noise.jld")

#concatenate the three dataframes 
experimental_data = vcat(experimental_data_traj_1, experimental_data_traj_2, experimental_data_traj_3)

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
#function to generate the Data
function lotka_volterra_gound_truth(u)
    α, β, γ, δ = original_parameters
    du = similar(u)
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
    return du
end

ensemble_1 = deserialize("ensemble_results_model_1_gain_1.0_reg_0_0.0_to_keep_traj_1.jld")
ensemble_2 = deserialize("ensemble_results_model_1_gain_1.0_reg_0_0.0_to_keep_traj_2.jld")
ensemble_3 = deserialize("ensemble_results_model_1_gain_1.0_reg_0_0.0_to_keep_traj_3.jld")

ensemble_1 = [member for member in ensemble_1 if member.status == "success"]
ensemble_2 = [member for member in ensemble_2 if member.status == "success"]
ensemble_3 = [member for member in ensemble_3 if member.status == "success"]

# extract uniformely 1000 points from the bounding box
n_points = 50000

function sample_uniform_2d(a, b, c, d)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    return (x, y)
end

min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]
ground_truth_vect_field = [lotka_volterra_gound_truth([p[1], p[2]]) for p in points]

# plots the points with the experimental data and the bounding box
plt = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points")
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="experimental data")

# compute for each point the confidence interval

function get_analysis_ci(i, ensemble, neural_network_dimension)

    approximating_neural_network = Lux.Chain(
        Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
        Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
        Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform),
    )
    get_uode_model_function = function (appr_neural_network, state)
        #generates the function with the parameters
        f(u, p) =
            let appr_neural_network = appr_neural_network, st = state
                û = appr_neural_network(u, p, st)[1]
                du = similar(u)
                du[1] = û[1]
                du[2] = û[2]
                return du
            end
    end
    local_rng = StableRNG(seed)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)
    model_function = get_uode_model_function(approximating_neural_network, st)

    point = points[i]
    gound_truth_vect = ground_truth_vect_field[i]

    ensemble_predictions = [model_function([point[1], point[2]], member.training_res.p) for member in ensemble if member.status == "success"]
    ensemble_predictions_dy1 = [member[1] for member in ensemble_predictions]
    ensmeble_predictions_dy2 = [member[2] for member in ensemble_predictions]

    mean_dy1 = mean(ensemble_predictions_dy1)
    mean_dy2 = mean(ensmeble_predictions_dy2)

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat(ensemble_predictions_dy1, ensmeble_predictions_dy2), 0.95)

    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(gound_truth_vect, confidence_ellipse)

    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean=[mean_dy1, mean_dy2], confidence_ellipse=confidence_ellipse, gound_truth_in_Ci=gound_truth_in_ellipse, area=confidece_ellipse_area, gound_truth=gound_truth_vect)
end

function analyze_method(index, total_ensemble, nn_dimension, threshold, points)

    println("**********************Analyzing method*********************")

    ensemble = [member for member in total_ensemble[index]]
    results = [get_analysis_ci(i, ensemble, nn_dimension, points) for i in 1:n_points]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean=[r.mean for r in results], ensemble_predictions=[r.ensemble_predictions for r in results], ground_truth_in_Ci=[r.gound_truth_in_Ci for r in results], mean_area=mean_area, ellipse_areas=[r.area for r in results], ground_truths=[r.gound_truth for r in results])
end

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)

dimensions_ensemble = [5, 10, 20, 30]

cicps_distributions = Dict()
mean_cicps = Dict()

for dimension_ensemble in dimensions_ensemble
    ensemble_1_tmp = ensemble_1[1:(dimension_ensemble*5)]
    ensemble_2_tmp = ensemble_2[1:(dimension_ensemble*5)]
    ensemble_3_tmp = ensemble_3[1:(dimension_ensemble*5)]

    tmp_ensemble = vcat(ensemble_1_tmp, ensemble_2_tmp, ensemble_3_tmp)

    indexes = [(k*dimension_ensemble+1):(k*dimension_ensemble+dimension_ensemble) for k in 0:14]
    results_ensembles = [analyze_method(index, tmp_ensemble, 32, threshold, points) for index in indexes]

    results = (points=points, results_ensemble=results_ensembles)
    serialize(result_image_folder * "/results_analysis" * string(dimension_ensemble) * ".jld", results)

    results_deserialized = deserialize(result_image_folder * "/results_analysis" * string(dimension_ensemble) * ".jld")
    results_ensemble = results_deserialized.results_ensemble
    points = results_deserialized.points

    mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])
    mean_cicp = round(mean_cicp, digits=2)

    # print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
    y_1_coordinate = [p[1] for p in points]
    y_2_coordinate = [p[2] for p in points]
    ground_truth_in_ensemble = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]

    #add two fake points to force the visualization between 0 and 1
    points_to_plot = deepcopy(points)
    push!(points_to_plot, points[1])
    push!(points_to_plot, points[1])
    ground_truth_in_ensemble_to_plot = deepcopy(ground_truth_in_ensemble)
    push!(ground_truth_in_ensemble_to_plot, 0)
    push!(ground_truth_in_ensemble_to_plot, 1)

    points_to_plot = reverse(points_to_plot)
    ground_truth_in_ensemble_to_plot = reverse(ground_truth_in_ensemble_to_plot)

    min_x = minimum([minimum(p[1]) for p in points_to_plot])
    max_x = maximum([maximum(p[1]) for p in points_to_plot])
    min_y = minimum([minimum(p[2]) for p in points_to_plot])
    max_y = maximum([maximum(p[2]) for p in points_to_plot])


    plt = Plots.scatter([p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="x", ylabel="y", zcolor=ground_truth_in_ensemble_to_plot, color=:viridis, markerstrokewidth=0, markersize=4, dpi = 600)
    Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="training data", legend=false)
    Plots.plot!(plt, xlims=(min_x, max_x), ylims=(min_y, max_y))
    Plots.plot!(plt, legend=:topleft)
    Plots.plot!(plt, title="Ensemble size $dimension_ensemble, mean CP: $mean_cicp", legend=:topleft)
    Plots.savefig(plt, result_image_folder * "/analysis_vector_field_ensemble_" * string(dimension_ensemble) * ".png")


    cicps_distributions[string(dimension_ensemble)] = [results_ensemble[j].cicp for j in axes(results_ensembles, 1)]

    mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])
    mean_cicps[string(dimension_ensemble)] = mean_cicp
end

box_plot = Plots.plot(title="", ylabel="MPICP", xlabel="Ensemble size", legend=nothing)
for ensemble_dimension in dimensions_ensemble
    box_plot = Plots.boxplot!(box_plot, cicps_distributions[string(ensemble_dimension)], label=ensemble_dimension, color=:lightblue)
end
Plots.ylims!(box_plot, 0, 1)
#set the xticks to be the ensemble dimensions
Plots.plot!(box_plot, xticks=(1:length(dimensions_ensemble), dimensions_ensemble))

#save the picture 
Plots.savefig(box_plot, result_image_folder * "/box_plot_cicp_with_different_sizes.png")
#save the mean cicps
CSV.write(result_image_folder * "/mean_cicps.csv", DataFrame(ensemble_size=dimensions_ensemble, mean_cicp=[mean_cicps[string(ensemble_dimension)] for ensemble_dimension in dimensions_ensemble]))
serialize(result_image_folder * "/cicps_distributions.jld", cicps_distributions)    