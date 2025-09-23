cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions, HypothesisTests

result_image_folder = "ensemble_5_diagnostic"
if !isdir(result_image_folder)
    mkpath(result_image_folder)
end

gr()

include("../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/lorenz_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/lorenz_in_silico_data_no_noise.jld")

in_dim = 3
out_dim = 3

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
#function to generate the Data
function lorenz_ground_truth(u)
    σ, r, b = original_parameters
    du = similar(u)
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
    return du
end

ensemble = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep.jld")

# extract uniformely 1000 points from the bounding box
n_points = 50000
function sample_uniform_3d(a, b, c, d, e, f)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    z = rand(rng) * (f - e) + e  # Sample z-coordinate
    return (x, y, z)
end

min_y1, max_y1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

points = [sample_uniform_3d(min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3) for i in 1:n_points]
ground_truth_vect_field = [lorenz_ground_truth([p[1], p[2], p[3]]) for p in points]

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
                du[3] = û[3]
                return du
            end
    end
    local_rng = StableRNG(seed)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)
    model_function = get_uode_model_function(approximating_neural_network, st)

    point = points[i]
    gound_truth_vect = ground_truth_vect_field[i]

    ensemble_predictions = [model_function([point[1], point[2], point[3]], member.training_res.p) for member in ensemble if member.status == "success"]
    ensemble_predictions_dy1 = [member[1] for member in ensemble_predictions] 
    ensmeble_predictions_dy2 = [member[2] for member in ensemble_predictions]
    ensemble_predictions_dy3 = [member[3] for member in ensemble_predictions]

    mean_dy1 = mean(ensemble_predictions_dy1)
    mean_dy2 = mean(ensmeble_predictions_dy2)
    mean_dy3 = mean(ensemble_predictions_dy3)

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat(ensemble_predictions_dy1, ensmeble_predictions_dy2, ensemble_predictions_dy3), 0.95)

    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(gound_truth_vect, confidence_ellipse)

    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean = [mean_dy1, mean_dy2, mean_dy3], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = gound_truth_vect) 
end

function analyze_method(member_seeds, total_ensemble, nn_dimension)

    println("**********************Analyzing method*********************")

    ensemble = [member for member in total_ensemble if member.status == "success" && member.seed in member_seeds]
    results = [get_analysis_ci(i, ensemble, nn_dimension) for i in 1:n_points]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

seeds = [member.seed for member in ensemble if member.status == "success"]

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)
sampled_seeds = [seeds[(k*5 + 1):(k*5+5)] for k in 0:9]

results_ensembles = [analyze_method(sampled_seed, ensemble, 32) for sampled_seed in sampled_seeds] 

results = (points = points, results_ensemble = results_ensembles)
serialize(result_image_folder*"/results_analysis.jld", results)

results_deserialized = deserialize(result_image_folder * "/results_analysis.jld")
results_ensemble = results_deserialized.results_ensemble
points = results_deserialized.points

# print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
y_1_coordinate = [p[1] for p in points]
y_2_coordinate = [p[2] for p in points]
y_3_coordinate = [p[3] for p in points]
ground_truth_in_ensemble = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]

mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])

#assess the statistical significance against the threshold
mcicps = [results_ensemble[j].cicp for j in axes(results_ensembles, 1)]
# Compute the differences from the fixed value
differences = mcicps .- 0.95
# Perform the Wilcoxon signed-rank test (two-tailed)
test = HypothesisTests.SignedRankTest(differences)
pvalue_test = pvalue(test; tail = :left)

#write the p-value on a text file
open(result_image_folder * "/p_value_vect_field.txt", "w") do file
    write(file, "P-value: $pvalue_test\n")
end

#boxplot of the ground truth in the confidence interval
plt = Plots.plot()
Plots.boxplot!(plt, [1], [mean(results_ensemble[j].ground_truth_in_Ci) for j in axes(results_ensemble, 1)], label="", ylabel="% ground truth in prediction interval", xlabel="", title="Ground truth in prediction interval")
Plots.plot!(plt, ylims=(0,1))
Plots.savefig(plt, result_image_folder * "/ground_truth_in_Ci_boxplot.png")

#for each point, compute the distance from the training trajectory (not significatively correlated)
distances_from_experimental_data = []
exp_data_as_list = [[experimental_data[i, "x1"], experimental_data[i, "x2"], experimental_data[i, "x3"]] for i in 1:size(experimental_data, 1)]

for i in 1:n_points
    point = points[i]
    # compute the distance from the experimental data
    distances = [norm([point[1], point[2], point[3]] .- exp) for exp in exp_data_as_list]
    push!(distances_from_experimental_data, minimum(distances))
end

#for each point, copute the average cicp
average_cicp = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]
Plots.scatter(distances_from_experimental_data, average_cicp, xlabel="Distance from experimental data", ylabel="Average CICP", title="Average CICP vs Distance from experimental data", label="", color=:blue, markersize=2)

#plot the distribution of the 