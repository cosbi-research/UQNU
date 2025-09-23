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

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/damped_oscillator_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/damped_oscillator_in_silico_data_no_noise.jld")

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

ensemble = deserialize("ensemble_results_model_3_with_seed.jld")
min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

#build a 100 * 100 grid
#points = [(x, y) for x in range(min_y1, max_y_1, length=100), y in range(min_y2, max_y2, length=100)]

n_points = 50000
function sample_uniform_2d(a, b, c, d)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    return (x, y)
end
points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]

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

function analyze_method(member_seeds, total_ensemble, nn_dimension, threshold, points)

    println("**********************Analyzing method*********************")

    ensemble = [member for member in total_ensemble if member.status == "success" && member.seed in member_seeds && member.validation_likelihood < threshold]
    results = [get_analysis_ci(i, ensemble, nn_dimension, points) for i in 1:n_points]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean = [r.mean for r in results], ensemble_predictions = [r.ensemble_predictions for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

seeds = [member.seed for member in ensemble if member.status == "success"]

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
sampled_seeds = [seeds[(k*5 + 1):(k*5+5)] for k in 0:9]
results_ensembles = [analyze_method(sampled_seed, ensemble, 32, threshold, points) for sampled_seed in sampled_seeds] 

results = (points = points, results_ensemble = results_ensembles)
serialize(result_image_folder * "/results_analysis_vect_field.jld", results)

results_deserialized = deserialize(result_image_folder * "/results_analysis_vect_field.jld")
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

min_x = minimum([minimum(p[1]) for p in points_to_plot])
max_x = maximum([maximum(p[1]) for p in points_to_plot])
min_y = minimum([minimum(p[2]) for p in points_to_plot])
max_y = maximum([maximum(p[2]) for p in points_to_plot])

mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])

#mean cicp of lotka_volterra_gound_truth
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

mean_cicp = round(mean_cicp, digits=2)

plt = Plots.scatter([p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="x", ylabel="y", zcolor=ground_truth_in_ensemble_to_plot, color=:viridis, markerstrokewidth=0, markersize=4, dpi=600)
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="training data", legend=false)
Plots.plot!(plt, xlims=(min_x, max_x), ylims=(min_y, max_y))
Plots.plot!(plt, title="mean CP: $mean_cicp", legend=:topleft)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)
Plots.savefig(plt, result_image_folder * "/analysis_vector_field_ensemble_5.svg")
Plots.savefig(plt, result_image_folder * "/analysis_vector_field_ensemble_5.png")

#print the vector field of the distance between the ground truth and the mean of the ensemble predictions
# print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
points_to_plot = deepcopy(points)
y_1_coordinate = [p[1] for p in points]
y_2_coordinate = [p[2] for p in points]
magnitude_confidence_region = [mean([results_ensemble[j].ellipse_areas[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]
magnitude_confidence_region = log10.(magnitude_confidence_region)

plt = Plots.scatter([p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="sampled points", xlabel="y1", ylabel="y2", zcolor=magnitude_confidence_region, color=:viridis, markerstrokewidth = 0, markersize = 3)
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="experimental data", legend = false)
Plots.savefig(plt, result_image_folder * "/analysis_magnitude_vector_field_ensemble_5.png")

points = [(x, y) for x in range(min_y1, max_y_1, length=21), y in range(min_y2, max_y2, length=21)]
points = vec(points)
n_points = length(points)

ground_truth_vect_field = [damped_oscillator_ground_truth([p[1], p[2]]) for p in points]
results_ensembles = [analyze_method(sampled_seed, ensemble, 32, threshold, points) for sampled_seed in sampled_seeds] 
#for result in results_ensembles

for result_iterator in 1:length(results_ensembles)

    plt = Plots.plot(size=(1800, 1600), aspect_ratio=:equal, legend = false, xlabel = "y1", ylabel = "y2", title = "Mean Vector Field (Ensemble 1)", xlims=(min_y1, max_y_1), ylims=(min_y2, max_y2))

    result = results_ensembles[result_iterator]

    magnitudes = norm.(ground_truth_vect_field)

    ground_truth_vect_field_tmp = ground_truth_vect_field ./ (10 * magnitudes)
    ground_truth_x = [g[1] for g in ground_truth_vect_field_tmp]
    ground_truth_y = [g[2] for g in ground_truth_vect_field_tmp]


    #try and depict the vector field 
    for i in 1:5

        resulting_vect_field = [[result.ensemble_predictions[p][1][i], result.ensemble_predictions[p][2][i]] for p in 1:n_points]
        points_quiver = deepcopy(points)
        x_points = [p[1] for p in points_quiver]
        y_points = [p[2] for p in points_quiver]

        #magnitudes = norm.(resulting_vect_field)

        vector_field = resulting_vect_field ./ (10 .* magnitudes)
        x_vector_field = [r[1] for r in vector_field]
        y_vector_field = [r[2] for r in vector_field]

        quiver!(plt,
            x_points, y_points,
            quiver = (x_vector_field, y_vector_field),
            xlabel = "y1", ylabel = "y2", 
            title = "Mean Vector Field (Ensemble 1)",
            legend = false,
            color = :red
        )
    end

    quiver!(plt, 
        x_points, y_points,
        quiver = (ground_truth_x, ground_truth_y),
        color = :black, label = "Ground Truth Vector Field"
    )

    #save the plot for the esnemble 
    Plots.savefig(plt, result_image_folder * "/ensemble_vector_field_5_$result_iterator.png")

end

#plot the ground truth vector field in black
ground_truth_vect_field = ground_truth_vect_field ./ (10 .* norm.(ground_truth_vect_field))
ground_truth_x = [g[1] for g in ground_truth_vect_field]
ground_truth_y = [g[2] for g in ground_truth_vect_field]

plt = Plots.plot(size=(1800, 1600), aspect_ratio=:equal, legend = false, xlabel = "y1", ylabel = "y2", title = "Mean Vector Field (Ensemble 1)", xlims=(min_y1, max_y_1), ylims=(min_y2, max_y2))

points_quiver = deepcopy(points)
x_points = [p[1] for p in points_quiver]
y_points = [p[2] for p in points_quiver]
quiver!(plt, 
    x_points, y_points,
    quiver = (ground_truth_x, ground_truth_y),
    color = :black, label = "Ground Truth Vector Field"
)

#save the plot for the ground truth vector field
Plots.savefig(plt, result_image_folder * "/ground_truth_vector_field_5.png")


