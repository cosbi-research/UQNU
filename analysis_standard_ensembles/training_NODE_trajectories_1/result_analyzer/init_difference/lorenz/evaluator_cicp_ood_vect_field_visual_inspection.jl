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
experimental_points = [(p[2], p[3], p[4]) for p in eachrow(experimental_data)]

#plots in a 3d scatter the experimental data colored by time
plt = Plots.scatter(experimental_data[:, :x1], experimental_data[:, :x2], experimental_data[:, :x3], label="experimental data", xlabel="y1", ylabel="y2", title="Experimental data", zcolor=experimental_data[:, :t], color=:viridis)

#trying to find the regression plane for the data
experimental_data_for_regression = experimental_data[experimental_data[:, :t] .> 0.75, :]

using DataFrames, GLM, StatsBase, LinearAlgebra

ols = lm(@formula(x3 ~ x1 + x2), experimental_data_for_regression)
round(r2(ols); digits=5)

intercept = coef(ols)[1]
coeff_x = coef(ols)[2]
coeff_y = coef(ols)[3]

#plot the scatterplot with the regression plane
x = LinRange(minimum(experimental_data_for_regression[:, :x1]), maximum(experimental_data_for_regression[:, :x1]), 100)
y = LinRange(minimum(experimental_data_for_regression[:, :x2]), maximum(experimental_data_for_regression[:, :x2]), 100)
z = [intercept + coeff_x * x_i + coeff_y * y_i for x_i in x, y_i in y]

plotly()
x_to_plot = repeat(x, 1, length(y))
y_to_plot = repeat(y', length(x), 1)
Plots.scatter(x_to_plot, y_to_plot, z, label="regression plane", color=:red)
Plots.scatter!(experimental_data[:, :x1], experimental_data[:, :x2], experimental_data[:, :x3], label="experimental data", xlabel="y1", ylabel="y2", title="Experimental data", zcolor=experimental_data[:, :t], color=:viridis)

#ok the regression plan is ok 

# get the generators of the plane
normal = [coeff_x, coeff_y, -1]
point = [0, 0, intercept]

# first generators
v1 = [coeff_y, -coeff_x, 0]
v2 = LinearAlgebra.cross(normal, v1)
#normalize it with respect to the norm
v1 = v1 / norm(v1)
v2 = v2 / norm(v2)
v3 = normal / norm(normal)

center = (mean(experimental_data_for_regression[:, :x1]), mean(experimental_data_for_regression[:, :x2]), mean(experimental_data_for_regression[:, :x3]))

#express the experimental point in the new basis with respect to the center
experimental_points_centered = [p .- center for p in experimental_points]
#express the experimental point in the new basis
experimental_points_centered_basis = [vcat([dot(p, v1), dot(p, v2), dot(p, v3)]) for p in experimental_points_centered]

#plot the experimental points in the new basis
plt = Plots.scatter([p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], [p[3] for p in experimental_points_centered_basis], label="experimental data", xlabel="v1", ylabel="v2", title="Experimental data in the new basis", zcolor=experimental_data[:, :t], color=:viridis)

#write a function to convert from canonical basis to the new basis
function convert_to_new_basis(p)
    p_centered = p .- center
    p_basis = vcat([dot(p_centered, v1), dot(p_centered, v2), dot(p_centered, v3)])
    return p_basis
end

#write a function to convert from the new basis to the canonical basis
function convert_to_canonical_basis(p_basis)
    p_centered = vcat([v1[1] * p_basis[1] + v2[1] * p_basis[2] + v3[1] * p_basis[3], v1[2] * p_basis[1] + v2[2] * p_basis[2] + v3[2] * p_basis[3], v1[3] * p_basis[1] + v2[3] * p_basis[2] + v3[3] * p_basis[3]])
    p = p_centered .+ center
    return p
end

#get the bounding box in the new basis
min_y1 = minimum([p[1] for p in experimental_points_centered_basis])
max_y_1 = maximum([p[1] for p in experimental_points_centered_basis])
min_y2 = minimum([p[2] for p in experimental_points_centered_basis])
max_y2 = maximum([p[2] for p in experimental_points_centered_basis])
min_y3 = minimum([p[3] for p in experimental_points_centered_basis])
max_y3 = maximum([p[3] for p in experimental_points_centered_basis])

#enlarge the bounding box of the 30%
width = max_y_1 - min_y1
height = max_y2 - min_y2
depth = max_y3 - min_y3

min_y1 = min_y1 - 0.3 * abs(width)
max_y_1 = max_y_1 + 0.3 * abs(width)
min_y2 = min_y2 - 0.3 * abs(height)
max_y2 = max_y2 + 0.3 * abs(height)
min_y3 = min_y3 - 0.3 * abs(depth)
max_y3 = max_y3 + 0.3 * abs(depth)

plt = Plots.scatter([p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", xlabel="v1", ylabel="v2", title="Experimental data in the new basis", zcolor=experimental_data[:, :t], color=:viridis)
Plots.plot!(plt, xlims=(min_y1, max_y_1), ylims=(min_y2, max_y2))


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

function sample_uniform_2d(a, b, c, d)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c
    return (x, y)
end

points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]
ground_truth_vect_field = [lorenz_ground_truth(convert_to_canonical_basis([p[1], p[2], 0.0])) for p in points]

# plots the points with the experimental data and the bounding box
plt = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points")
Plots.scatter!(plt, [s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="experimental data")

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
    point_in_old_coordinate = convert_to_canonical_basis([point[1], point[2], 0.0])
    gound_truth_vect = ground_truth_vect_field[i]

    ensemble_predictions = [model_function([point_in_old_coordinate[1], point_in_old_coordinate[2], point_in_old_coordinate[3]], member.training_res.p) for member in ensemble if member.status == "success"]
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

function analyze_method(member_seeds, total_ensemble, nn__width, threshold)

    println("**********************Analyzing method*********************")

    ensemble = [member for member in total_ensemble if member.status == "success" && member.seed in member_seeds && member.validation_likelihood < threshold]
    results = [get_analysis_ci(i, ensemble, nn__width) for i in 1:n_points]

    cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
    mean_area = mean([r.area for r in results])

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

#possible seeds 
seeds_1 = [member.seed for member in results_1 if member.status == "success"]
seeds_01 = [member.seed for member in results_01 if member.status == "success"]
seeds_001 = [member.seed for member in results_001 if member.status == "success"]

seeds = unique(vcat(seeds_1, seeds_01, seeds_001))

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)
sampled_seeds = [sample(seeds, 200, replace=true) for i in 1:20]

results_1_analyzed = [analyze_method(sampled_seeds[i], results_1, 32, threshold) for i in 1:20]
results_01_analyzed = [analyze_method(sampled_seeds[i], results_01, 32, threshold) for i in 1:20]
results_001_analyzed = [analyze_method(sampled_seeds[i], results_001, 32, threshold) for i in 1:20]

#plot the distribution of the cicp_d1
# print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
y_1_coordinate = [p[1] for p in points]
y_2_coordinate = [p[2] for p in points]
cicp_dy1dy2_1 = [mean([r.ground_truth_in_Ci[i] for r in results_1_analyzed]) for i in 1:n_points]
cicp_dy1dy2_01 = [mean([r.ground_truth_in_Ci[i] for r in results_01_analyzed]) for i in 1:n_points]
cicp_dy1dy2_001 = [mean([r.ground_truth_in_Ci[i] for r in results_001_analyzed]) for i in 1:n_points]

gr()

plt = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=cicp_dy1dy2_1, color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "cicp_dy1dy2_1.png")

plt = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=cicp_dy1dy2_01, color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "cicp_dy1dy2_01.png")

plt = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=cicp_dy1dy2_001, color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "cicp_dy1dy2_001.png")

brewer_score_1 = mean([(x-0.95)^2 for x in cicp_dy1dy2_1])
brewer_score_01 = mean([(x-0.95)^2 for x in cicp_dy1dy2_01])
brewer_score_001 = mean([(x-0.95)^2 for x in cicp_dy1dy2_001])

#do the same for the area of the ellipse
cicp_dy1dy2_1 = [mean([r.ellipse_areas[i] for r in results_1_analyzed]) for i in 1:n_points]
cicp_dy1dy2_01 = [mean([r.ellipse_areas[i] for r in results_01_analyzed]) for i in 1:n_points]
cicp_dy1dy2_001 = [mean([r.ellipse_areas[i] for r in results_001_analyzed]) for i in 1:n_points]

minimum_heat = minimum([minimum(cicp_dy1dy2_1), minimum(cicp_dy1dy2_01), minimum(cicp_dy1dy2_001)])
maximum_heat = maximum([maximum(cicp_dy1dy2_1), maximum(cicp_dy1dy2_01), maximum(cicp_dy1dy2_001)])
augmented_points = vcat([points[1]], [points[1]], points)

cicp_dy1dy2_1 = vcat([minimum_heat], [maximum_heat], cicp_dy1dy2_1)
cicp_dy1dy2_01 = vcat([minimum_heat], [maximum_heat], cicp_dy1dy2_01)
cicp_dy1dy2_001 = vcat([minimum_heat], [maximum_heat], cicp_dy1dy2_001)

min_manual = -7.0
cicp_dy1dy2_1[cicp_dy1dy2_1.< min_manual] .= min_manual
cicp_dy1dy2_01[cicp_dy1dy2_01.< min_manual] .= min_manual
cicp_dy1dy2_001[cicp_dy1dy2_001.< min_manual] .= min_manual

plt = Plots.scatter([p[1] for p in augmented_points], [p[2] for p in augmented_points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=log10.(cicp_dy1dy2_1),  color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "area_dy1dy2_1.png")

plt = Plots.scatter([p[1] for p in augmented_points], [p[2] for p in augmented_points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=log10.(cicp_dy1dy2_01),  color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "area_dy1dy2_01.png")

plt = Plots.scatter([p[1] for p in augmented_points], [p[2] for p in augmented_points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=log10.(cicp_dy1dy2_001), color=:viridis)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "area_dy1dy2_001.png")


area_dy1dy2_001_diff_1 = [mean([log10(results_001_analyzed[k].ellipse_areas[i]) - log10(results_1_analyzed[k].ellipse_areas[i]) for k in axes(results_1_analyzed, 1)]) for i in 1:n_points]
area_dy1dy2_001_diff_01= [mean([log10(results_001_analyzed[k].ellipse_areas[i]) - log10(results_01_analyzed[k].ellipse_areas[i]) for k in axes(results_1_analyzed, 1)]) for i in 1:n_points]

min_value = -1
max_value = 1

area_dy1dy2_001_diff_1[area_dy1dy2_001_diff_1 .< min_value] .= min_value
area_dy1dy2_001_diff_1[area_dy1dy2_001_diff_1 .> max_value] .= max_value

area_dy1dy2_001_diff_01[area_dy1dy2_001_diff_01 .< min_value] .= min_value
area_dy1dy2_001_diff_01[area_dy1dy2_001_diff_01 .> max_value] .= max_value

augmented_points = vcat([points[1]], [points[1]], points)

area_dy1dy2_001_diff_1 = vcat([min_value], [max_value], area_dy1dy2_001_diff_1)
area_dy1dy2_001_diff_01 = vcat([min_value], [max_value], area_dy1dy2_001_diff_01)

col_grad = cgrad([:blue, :white, :orange], [-1.0, 0.0,  1.0])

plt = Plots.scatter([p[1] for p in augmented_points], [p[2] for p in augmented_points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=area_dy1dy2_001_diff_1,  color=col_grad)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "area_dy1dy2_001_vs_1.png")

plt = Plots.scatter([p[1] for p in augmented_points], [p[2] for p in augmented_points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=area_dy1dy2_001_diff_01, color=col_grad)
Plots.scatter!(plt, [p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], label="experimental data", legend = false)
Plots.savefig(plt, "area_dy1dy2_001_vs_01.png")