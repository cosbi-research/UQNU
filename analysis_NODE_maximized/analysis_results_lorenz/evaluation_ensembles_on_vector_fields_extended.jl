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

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/lorenz/result_lorenz$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.ensemble_reprojected
    push!(ensembles, tmp_ensemble)
end

# extract uniformely 100 points from the bounding box
# build a grid of 100 * 100 points
# extract uniformely 1000 points from the bounding box
n_points = 50000
function sample_uniform_3d(a, b, c, d, e, f)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    z = rand(rng) * (f - e) + e  # Sample z-coordinate
    return (x, y, z)
end

min_y1, max_y1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

# enlarge to 50/%
new_min_y1 = min_y1 - 0.2 * (max_y1 - min_y1)
new_max_y1 = max_y1 + 0.2 * (max_y1 - min_y1)
new_min_y2 = min_y2 - 0.2 * (max_y2 - min_y2)
new_max_y2 = max_y2 + 0.2 * (max_y2 - min_y2)
new_max_y3 = max_y3 + 0.2 * (max_y3 - min_y3)
new_min_y3 = min_y3 - 0.2 * (max_y3 - min_y3)

min_y1, max_y1, min_y2, max_y2, min_y3, max_y_3 = new_min_y1, new_max_y1, new_min_y2, new_max_y2, new_min_y3, new_max_y3

points = [sample_uniform_3d(min_y1, max_y1, min_y2, max_y2, min_y3, max_y3) for i in 1:n_points]
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

    ensemble_predictions = [model_function([point[1], point[2], point[3]], member) for member in ensemble]
    ensemble_predictions_dy1 = [member[1] for member in ensemble_predictions]
    ensmeble_predictions_dy2 = [member[2] for member in ensemble_predictions]
    ensemble_predictions_dy3 = [member[3] for member in ensemble_predictions]

    mean_dy1 = mean(ensemble_predictions_dy1)
    mean_dy2 = mean(ensmeble_predictions_dy2)
    mean_dy3 = mean(ensemble_predictions_dy3)

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat(ensemble_predictions_dy1, ensmeble_predictions_dy2, ensemble_predictions_dy3), 0.95)

    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(gound_truth_vect, confidence_ellipse)

    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean=[mean_dy1, mean_dy2, mean_dy3], confidence_ellipse=confidence_ellipse, gound_truth_in_Ci=gound_truth_in_ellipse, area=confidece_ellipse_area, gound_truth=gound_truth_vect)
end

function analyze_method(ensemble, nn_dimension)
    try
        results = [get_analysis_ci(i, ensemble, nn_dimension) for i in 1:(n_points)]

        cicp = sum([r.gound_truth_in_Ci for r in results]) / (n_points)
        mean_area = mean([r.area for r in results])

        return (cicp=cicp, mean=[r.mean for r in results], ground_truth_in_Ci=[r.gound_truth_in_Ci for r in results], mean_area=mean_area, ellipse_areas=[r.area for r in results], ground_truths=[r.gound_truth for r in results])
    catch
        return nothing
    end

end

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)

results_ensembles = [analyze_method(ensemble, 32) for ensemble in ensembles]
results_ensembles = [r for r in results_ensembles if r != nothing]

results = (points=points, results_ensemble=results_ensembles)
serialize("results_analysis_extended.jld", results)

results_deserialized = deserialize("results_analysis_extended.jld")
results_ensemble = results_deserialized.results_ensemble
points = results_deserialized.points

mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])

# print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
y_1_coordinate = [p[1] for p in points]
y_2_coordinate = [p[2] for p in points]
y_3_coordinate = [p[3] for p in points]
ground_truth_in_ensemble = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]

single_cicps = [mean([results_ensemble[j].ground_truth_in_Ci[i] for i in axes(points, 1)]) for j in axes(results_ensemble, 1)] 
serialize("cicps_distribution_extended.jld", single_cicps)