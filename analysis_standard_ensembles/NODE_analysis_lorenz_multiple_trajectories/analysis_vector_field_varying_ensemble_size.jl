cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

result_image_folder = "ensemble_diagnostic_varying_vector_field"
if !isdir(result_image_folder)
    mkdir(result_image_folder)
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
experimental_data_traj_1 = deserialize("../data_generator_traj_1/lorenz_in_silico_data_no_noise.jld")
experimental_data_traj_2 = deserialize("../data_generator_traj_2/lorenz_in_silico_data_no_noise.jld")
experimental_data_traj_3 = deserialize("../data_generator_traj_3/lorenz_in_silico_data_no_noise.jld")

#concatenate the three dataframes 
experimental_data = vcat(experimental_data_traj_1, experimental_data_traj_2, experimental_data_traj_3)

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

ensemble_1 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_1.jld")
ensemble_2 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_2.jld")
ensemble_3 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_3.jld")

ensemble_1 = [member for member in ensemble_1 if member.status == "success"]
ensemble_2 = [member for member in ensemble_2 if member.status == "success"]
ensemble_3 = [member for member in ensemble_3 if member.status == "success"]

# extract uniformely 1000 points from the bounding box
n_points = 50000

function sample_uniform_3d(a, b, c, d, e, f)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    z = rand() * (f - e) + e  # Sample z-coordinate
    return (x, y, z)
end

min_y1, max_y1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

points = [sample_uniform_3d(min_y1, max_y1, min_y2, max_y2, min_y3, max_y3) for i in 1:n_points]
ground_truth_vect_field = [lorenz_ground_truth([p[1], p[2], p[3]]) for p in points]

# compute for each point the confidence interval

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

    return (mean = [mean_dy1, mean_dy2, mean_dy3], ensemble_predictions=[ensemble_predictions_dy1, ensmeble_predictions_dy2, ensemble_predictions_dy3], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = gound_truth_vect) 
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

ensemble_dimensions = [5, 10, 20, 30]

cicps_distributions = Dict()

mean_cicps = Dict()

for ensemble_dimension in ensemble_dimensions
    
    global points 

    ensemble_1_tmp = ensemble_1[1:min(ensemble_dimension*5, length(ensemble_1))]
    ensemble_2_tmp = ensemble_2[1:min(ensemble_dimension*5, length(ensemble_2))]
    ensemble_3_tmp = ensemble_3[1:min(ensemble_dimension*5, length(ensemble_3))]

    tmp_ensemble = vcat(ensemble_1_tmp, ensemble_2_tmp, ensemble_3_tmp)

    indexes = [(k*ensemble_dimension+1):(k*ensemble_dimension+ensemble_dimension) for k in 0:min(14, Int(floor(length(tmp_ensemble)/ensemble_dimension)) - 1)]
    results_ensembles = [analyze_method(index, tmp_ensemble, 32, threshold, points) for index in indexes]

    results = (points = points, results_ensemble = results_ensembles)
    serialize("results_analysis_"*string(ensemble_dimension)*".jld", results)

    results_deserialized = deserialize("results_analysis_"*string(ensemble_dimension)*".jld")
    results_ensemble = results_deserialized.results_ensemble
    points = results_deserialized.points

    # print in the scatter plot the variance (colored from yellow to blue) for each point (for now just dy1)
    y_1_coordinate = [p[1] for p in points]
    y_2_coordinate = [p[2] for p in points]
    y_3_coordinate = [p[3] for p in points]
    ground_truth_in_ensemble = [mean([results_ensemble[j].ground_truth_in_Ci[i] for j in axes(results_ensemble, 1)]) for i in axes(points, 1)]

    mean_cicp = mean([results_ensemble[j].cicp for j in axes(results_ensembles, 1)])
    mean_cicps[string(ensemble_dimension)] = mean_cicp

    cicps_distributions[string(ensemble_dimension)] = [results_ensemble[j].cicp for j in axes(results_ensembles, 1)]
end


box_plot = Plots.plot(title="", ylabel="MPICP", xlabel="Ensemble size", legend=nothing)
for ensemble_dimension in ensemble_dimensions
    global box_plot
    box_plot = Plots.boxplot!(box_plot, cicps_distributions[string(ensemble_dimension)], label=ensemble_dimension, color=:lightblue)
end
Plots.ylims!(box_plot, 0, 1)
#set the xticks to be the ensemble dimensions
Plots.plot!(box_plot, xticks=(1:length(ensemble_dimensions), ensemble_dimensions))
box_plot

#save the picture 
Plots.savefig(box_plot, result_image_folder * "/box_plot_cicp_with_different_sizes.png")

#save the mean cicps
CSV.write(result_image_folder * "/mean_cicps.csv", DataFrame(ensemble_size = ensemble_dimensions, mean_cicp = [mean_cicps[string(ensemble_dimension)] for ensemble_dimension in ensemble_dimensions]))
serialize(result_image_folder * "/cicps_distributions.jld", cicps_distributions)