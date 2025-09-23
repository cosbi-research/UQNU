cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

result_image_folder = "ensemble_5_diagnostic_enlarged_vector_field"
if !isdir(result_image_folder)
    mkpath(result_image_folder)
end

gr()

include("../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("bounding_box_vec_field.jld")
experimental_data_traj_1 = deserialize("../data_generator_traj_1/damped_oscillator_in_silico_data_no_noise.jld")
experimental_data_traj_2 = deserialize("../data_generator_traj_2/damped_oscillator_in_silico_data_no_noise.jld")
experimental_data_traj_3 = deserialize("../data_generator_traj_3/damped_oscillator_in_silico_data_no_noise.jld")

#concatenate the three dataframes 
experimental_data = vcat(experimental_data_traj_1, experimental_data_traj_2, experimental_data_traj_3)

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[0.1, 2]
function damped_oscillator_ground_truth(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
    return du
end

ensemble_1 = deserialize("ensemble_results_model_3_gain_1.0_reg_0_0.0_to_keep_traj_1.jld")
ensemble_2 = deserialize("ensemble_results_model_3_gain_1.0_reg_0_0.0_to_keep_traj_2.jld")
ensemble_3 = deserialize("ensemble_results_model_3_gain_1.0_reg_0_0.0_to_keep_traj_3.jld")

ensemble_1 = [member for member in ensemble_1 if member.status == "success"]
ensemble_2 = [member for member in ensemble_2 if member.status == "success"]
ensemble_3 = [member for member in ensemble_3 if member.status == "success"]

ensemble_1 = ensemble_1[1:50]
ensemble_2 = ensemble_2[1:50]
ensemble_3 = ensemble_3[1:50]

ensemble = vcat(ensemble_1, ensemble_2, ensemble_3)

min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

width = max_y_1 - min_y1
height = max_y2 - min_y2

#enlarge the bounding box by 30%
min_y1 = min_y1 - 1.0 * width
max_y_1 = max_y_1 + 1.0 * width
min_y2 = min_y2 - 1.0 * height
max_y2 = max_y2 + 1.0 * height


#build a 100 * 100 grid
points = [(x, y) for x in range(min_y1, max_y_1, length=100), y in range(min_y2, max_y2, length=100)]
points = vec(points)
n_points = length(points)

ground_truth_vect_field = [damped_oscillator_ground_truth([p[1], p[2]]) for p in points]

# compute for each point the confidence interval
function get_analysis_ci(i, ensemble, neural_network_dimension, points)

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

    return (mean = [mean_dy1, mean_dy2], ensemble_predictions = [ensemble_predictions_dy1, ensmeble_predictions_dy2], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = gound_truth_vect) 
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
indexes = [(k*5+1):(k*5+5) for k in 0:(10*3-1)]
results_ensembles = [analyze_method(index, ensemble, 32, threshold, points) for index in indexes]

results = (points = points, results_ensemble = results_ensembles)
serialize("results_analysis.jld", results)

results_deserialized = deserialize("results_analysis.jld")
results_ensemble = results_deserialized.results_ensemble
points = results_deserialized.points

# print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
y_1_coordinate = [p[1] for p in points]
y_2_coordinate = [p[2] for p in points]
ground_truth_in_ensemble = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]

#add two fake points to force the visualization between 0 and 1
points_to_plot = deepcopy(points)
insert!(points_to_plot, 1, points[1])
insert!(points_to_plot, 1, points[1])
ground_truth_in_ensemble_to_plot = deepcopy(ground_truth_in_ensemble)
insert!(ground_truth_in_ensemble_to_plot, 1, 0)
insert!(ground_truth_in_ensemble_to_plot, 1, 1)

plt = Plots.scatter([p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="sampled points", xlabel="y1", ylabel="y2", zcolor=ground_truth_in_ensemble_to_plot, color=:viridis, markerstrokewidth = 0, markersize = 3, dpi=600)
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="experimental data", legend = false)
Plots.savefig(plt, result_image_folder * "/analysis_vector_field_ensemble_5.png")

