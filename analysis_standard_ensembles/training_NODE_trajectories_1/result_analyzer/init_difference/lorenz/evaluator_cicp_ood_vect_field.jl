cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

include("../../../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 10^(-2)

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../../../data_generator/lorenz_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../../../data_generator/lorenz_in_silico_data_no_noise.jld")
experimental_points = [(p[1], p[2], p[3]) for p in eachrow(experimental_data)]

in_dim = 3
out_dim = 3
#approximating neural network
neural_network_dimension = 32
activation_function_fun = gelu

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)
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

results_1 = deserialize("ensemble_results_model_2_gain_1.0_reg_0.jld")
results_01 = deserialize("ensemble_results_model_2_gain_0.1_reg_0.jld")
results_001 = deserialize("ensemble_results_model_2_gain_0.01_reg_0.jld")

# extract uniformely 1000 points from the bounding box
n_points = 5000

function sample_uniform_3d(a, b, c, d, e, f)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c
    z = rand() * (f - e) + e  # Sample y-coordinate
    return (x, y, z)
end

min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

points = [sample_uniform_3d(min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3) for i in 1:n_points]
ground_truth_vect_field = [lorenz_ground_truth([p[1], p[2], p[3]]) for p in points]

# plots the points with the experimental data and the bounding box
plt = Plots.scatter([p[1] for p in points], [p[2] for p in points],  [p[3] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points")
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], experimental_data[:, :x3], label="experimental data")

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
    ensmeble_predictions_dy3 = [member[3] for member in ensemble_predictions]

    mean_dy1 = mean(ensemble_predictions_dy1)
    mean_dy2 = mean(ensmeble_predictions_dy2)
    mean_dy3 = mean(ensmeble_predictions_dy3)

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat(ensemble_predictions_dy1, ensmeble_predictions_dy2, ensmeble_predictions_dy3), 0.95)

    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(gound_truth_vect, confidence_ellipse)

    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean = [mean_dy1, mean_dy2, mean_dy3], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = gound_truth_vect) 
end

function analyze_method(member_seeds, total_ensemble, nn_width, threshold)

    println("**********************Analyzing method*********************")

    ensemble = [member for member in total_ensemble if member.status == "success" && member.seed in member_seeds && member.validation_likelihood < threshold]
    results = [get_analysis_ci(i, ensemble, nn_width) for i in 1:n_points]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

seeds_1 = [member.seed for member in results_1 if member.status == "success"]
seeds_01 = [member.seed for member in results_01 if member.status == "success"]
seeds_001 = [member.seed for member in results_001 if member.status == "success"]

seeds = unique(vcat(seeds_1, seeds_01, seeds_001))

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)
sampled_seeds = [sample(seeds, 20, replace=false) for i in 1:10]

results_1_analyzed = [analyze_method(sampled_seeds[i], results_1, 32, threshold) for i in 1:10]
results_01_analyzed = [analyze_method(sampled_seeds[i], results_01, 32, threshold) for i in 1:10]
results_001_analyzed = [analyze_method(sampled_seeds[i], results_001, 32, threshold) for i in 1:10]

results = (results_1_analyzed = results_1_analyzed, results_01_analyzed = results_01_analyzed, results_001_analyzed = results_001_analyzed, points = points)
serialize("results_analysis_vect_field.jld", results)

results_deserialized = deserialize("results_analysis_vect_field.jld")
results_1_analyzed = results_deserialized.results_1_analyzed
results_01_analyzed = results_deserialized.results_01_analyzed
results_001_analyzed = results_deserialized.results_001_analyzed
points = results_deserialized.points

#plot the distribution of the cicp_d1
mini = minimum([minimum([r.cicp for r in results_1_analyzed]), minimum([r.cicp for r in results_01_analyzed]), minimum([r.cicp for r in results_001_analyzed])])
maxi = maximum([maximum([r.cicp for r in results_1_analyzed]), maximum([r.cicp for r in results_01_analyzed]), maximum([r.cicp for r in results_001_analyzed])])
bin_width = (maxi-mini)/100
bins = mini:bin_width:maxi

plt = Plots.histogram([r.cicp for r in results_1_analyzed], bins=bins, label="cicp 1.0", xlabel="CICP", ylabel="Frequency", title="CICP distribution", alpha=0.5)
Plots.histogram!([r.cicp for r in results_01_analyzed], bins=bins, label="cicp 0.1", alpha=0.5)
Plots.histogram!([r.cicp for r in results_001_analyzed], bins=bins, label="cicp L2 0.1", alpha=0.5)

Plots.savefig(plt, "cicp_distribution_both_variables.png")

mini = minimum([minimum([r.mean_area for r in results_1_analyzed]), minimum([r.mean_area for r in results_01_analyzed]), minimum([r.mean_area for r in results_001_analyzed])])
maxi = maximum([maximum([r.mean_area for r in results_1_analyzed]), maximum([r.mean_area for r in results_01_analyzed]), maximum([r.mean_area for r in results_001_analyzed])])
bin_width = (maxi-mini)/100
bins = mini:bin_width:maxi

#mean area distribution
plt = Plots.histogram([r.mean_area for r in results_1_analyzed], bins=bins, label="area 1.0", xlabel="Mean area", ylabel="Frequency", title="Mean area distribution", alpha=0.5)
Plots.histogram!([r.mean_area for r in results_01_analyzed], bins=bins, label="area L2 0.1", alpha=0.5)
Plots.histogram!([r.mean_area for r in results_001_analyzed], bins=bins, label="area L2 0.01", alpha=0.5)

Plots.savefig(plt, "mean_area_distribution_both_variables.png")

#viol plot
# Extracting data for violin plots
cicp_1 = [r.cicp for r in results_1_analyzed]
cicp_01 = [r.cicp for r in results_01_analyzed]
cicp_001 = [r.cicp for r in results_001_analyzed]

data = [cicp_1; cicp_01; cicp_001]
labels = vcat(
    fill("gain 1", length(cicp_1)),
    fill("gain 0.1", length(cicp_01)),
    fill("gain 0.01", length(cicp_001))
)

# Group data for statistics
grouped_data = [cicp_1, cicp_01, cicp_001]
group_labels = ["gain 1", "gain 0.1", "gain 0.01"]

# Calculate means and 95% CI (1.96 * std / sqrt(n))
means = [mean(g) for g in grouped_data]
std_devs = [std(g) for g in grouped_data]
cis = [1.96 * std(g) for g in grouped_data]

# Create the violin plot
StatsPlots.violin(
    labels,
    data,
    legend=false,
    alpha=0.5,
    ylims=(0, 1),
    linewidth=0.5
)

# Overlay mean and 95% CI with scatter and error bars
Plots.scatter!(
    group_labels,
    means,
    yerr=cis,
    color=:black,
    label="Mean ± 95% CI",
    markershape=:circle,
    markersize=6
)

# Save the plot
Plots.savefig("cicp_violin_plot_with_errorbars.png")
