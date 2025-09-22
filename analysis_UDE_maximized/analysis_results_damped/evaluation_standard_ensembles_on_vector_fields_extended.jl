cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

include("../ConfidenceEllipse.jl")
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
#function to generate the Data
function damped_ground_truth_function(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α * u[1]^3 - β * u[2]^3
    du[2] = β * u[1]^3 - α * u[2]^3
    return du
end


n_points = 50000
min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

# enlarge to 50/%
new_min_y1 = min_y1 - 0.2 * (max_y_1 - min_y1)
new_max_y_1 = max_y_1 + 0.2 * (max_y_1 - min_y1)
new_min_y2 = min_y2 - 0.2 * (max_y2 - min_y2)
new_max_y2 = max_y2 + 0.2 * (max_y2 - min_y2)

min_y1, max_y_1, min_y2, max_y2 = new_min_y1, new_max_y_1, new_min_y2, new_max_y2

function sample_uniform_2d(a, b, c, d)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    return (x, y)
end
points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]

ground_truth_vect_field = [damped_ground_truth_function([p[1], p[2]]) for p in points]

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
                û = appr_neural_network(u, p.p_net, st)[1]
                du = similar(u)
                du[1] = -p.α*u[1]^3 + û[1]
                du[2] = û[2] - p.α*u[2]^3
                return du
            end
    end
    local_rng = StableRNG(seed)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)
    model_function = get_uode_model_function(approximating_neural_network, st)

    point = points[i]
    gound_truth_vect = ground_truth_vect_field[i]

    ensemble_predictions = [model_function([point[1], point[2]], member) for member in ensemble]
    ensemble_predictions_dy1 = [member[1] for member in ensemble_predictions] 
    ensmeble_predictions_dy2 = [member[2] for member in ensemble_predictions]

    mean_dy1 = mean(ensemble_predictions_dy1)
    mean_dy2 = mean(ensmeble_predictions_dy2)

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat(ensemble_predictions_dy1, ensmeble_predictions_dy2), 0.95)

    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(gound_truth_vect, confidence_ellipse)

    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean = [mean_dy1, mean_dy2], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = gound_truth_vect) 
end

function analyze_method(ensemble, nn_dimension)

    println("**********************Analyzing method*********************")

    results = [get_analysis_ci(i, ensemble, nn_dimension) for i in 1:(n_points)]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / (n_points)
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end


# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)

naive_ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/damped/result_damped$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(naive_ensembles, tmp_ensemble)
end

results_ensembles = [analyze_method(naive_ensemble, 32) for naive_ensemble in naive_ensembles] 

results = (points = points, results_ensemble = results_ensembles)
serialize("results_analysis_naive_extended.jld", results)

results_deserialized = deserialize("results_analysis_naive_extended.jld")
results_ensemble = results_deserialized.results_ensemble
points = results_deserialized.points

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

#reshape the ground_truth with the original shape
mean_cicp = mean(ground_truth_in_ensemble_to_plot)
mean_cicp = round(mean_cicp; digits=2)

min_x = minimum([minimum(p[1]) for p in points_to_plot])
max_x = maximum([maximum(p[1]) for p in points_to_plot])
min_y = minimum([minimum(p[2]) for p in points_to_plot])
max_y = maximum([maximum(p[2]) for p in points_to_plot])

plt = Plots.scatter([p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="x", ylabel="y", zcolor=ground_truth_in_ensemble_to_plot, color=:viridis, markerstrokewidth=0, markersize=4, dpi=600)
Plots.scatter!(plt, experimental_data[:, :x1], experimental_data[:, :x2], label="training data", legend=false)
Plots.plot!(plt, xlims=(min_x, max_x), ylims=(min_y, max_y))
Plots.plot!(plt, title="Standard (mean CP: $mean_cicp)", legend=:topleft)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)

Plots.savefig(plt, "ground_truth_in_ensemble_naive_extended.png")
Plots.savefig(plt, "ground_truth_in_ensemble_naive_extended.svg")

single_cicps = [mean([results_ensemble[j].ground_truth_in_Ci[i] for i in axes(points, 1)]) for j in axes(results_ensemble, 1)] 
Plots.histogram(single_cicps, label="CICP", xlabel="CICP", ylabel="Frequency", title="CICP distribution")
serialize("cicps_distribution_naive_extended.jld", single_cicps)

single_cicps_maximized = deserialize("cicps_distribution_extended.jld")

using HypothesisTests
single_cicps_maximized_copied = abs.(single_cicps_maximized .- 0.95)
single_cicps_copied = abs.(single_cicps .- 0.95)
test = HypothesisTests.ApproximateSignedRankTest(single_cicps_maximized_copied, single_cicps_copied)
pval = pvalue(test, tail = :left)

#plot the densities comparing the two
plt = Plots.boxplot(["Standard" for _ in single_cicps], single_cicps, label="Standard", xlabel="", ylabel="CP", alpha=0.5, color=:lightblue)
Plots.boxplot!(plt, ["Maximized OOD disagreement" for _ in single_cicps_maximized], single_cicps_maximized, label="Maximized OOD disagreement", alpha=0.5, color=:lightblue)
Plots.plot!(plt, title="p-value: "*string(round(pval, sigdigits=2)))
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    legend=false,
    rightmargin=30px
)
#save it 
Plots.yaxis!(plt, (0, 1.1))
Plots.savefig(plt, "cicps_distribution_comparison_extended.png")
