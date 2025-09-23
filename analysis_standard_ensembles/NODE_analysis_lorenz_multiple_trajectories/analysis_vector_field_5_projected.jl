cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

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
boundig_box_vect_field = deserialize("bounding_box_vec_field.jld")
experimental_data_traj_1 = deserialize("../data_generator_traj_1/lorenz_in_silico_data_no_noise.jld")
experimental_data_traj_2 = deserialize("../data_generator_traj_2/lorenz_in_silico_data_no_noise.jld")
experimental_data_traj_3 = deserialize("../data_generator_traj_3/lorenz_in_silico_data_no_noise.jld")

#concatenate the three dataframes 
experimental_data = vcat(experimental_data_traj_1, experimental_data_traj_2, experimental_data_traj_3)

experimental_points = [(p[2], p[3], p[4]) for p in eachrow(experimental_data)]
experimental_points_for_regression = [(p[2], p[3], p[4]) for p in eachrow(experimental_data_traj_1)]

#trying to find the regression plane for the data
experimental_data_for_regression = experimental_data[experimental_data[:, :t].>0.75, :]

using DataFrames, GLM, StatsBase, LinearAlgebra

ols = lm(@formula(x3 ~ x1 + x2), experimental_data_for_regression)
round(r2(ols); digits=5)

intercept = coef(ols)[1]
coeff_x = coef(ols)[2]
coeff_y = coef(ols)[3]

#plot the scatterplot with the regression plane
x = LinRange(minimum(experimental_data_for_regression[:, :x1]), maximum(experimental_data_for_regression[:, :x1]), 100)
y = LinRange(minimum(experimental_data_for_regression[:, :x2]), maximum(experimental_data_for_regression[:, :x2]), 100)
z = ([intercept + coeff_x * x_i + coeff_y * y_i for y_i in y, x_i in x])
#get the x correspongin to z
#x = repeat(x, outer=length(x))
#y = repeat(y, inner=length(y))
#z = vec(z)

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
experimental_points_distances = abs.([point[3] for point in experimental_points_centered_basis])

# Convert tuples to vectors for plotting
xs = [p[1] for p in experimental_points]
ys = [p[2] for p in experimental_points]
zs = [p[3] for p in experimental_points]


gr()

# Plot the training points
plt1 = Plots.plot(
    xs, ys, zs,
    seriestype = :scatter,
    label = "Training Points",
    legend = :topleft,
    markerstrokewidth = 0,
    markersize = 3,
    xlabel = "x",
    ylabel = "y",
    zlabel = "z",
    color= "red"
)

# Plot the regression plane
Plots.plot!(
    plt1,
    x, y, z,             # meshgrid
    st = :surface,       # or `seriestype = :surface`
    alpha = 0.4,         # transparency
    color = :blue,
    label = "",
    colorbar = false
)

#plot the experimental points in the new basis
gr()
plt = Plots.scatter([p[1] for p in experimental_points_centered_basis], [p[2] for p in experimental_points_centered_basis], [p[3] for p in experimental_points_centered_basis], label="", xlabel="v1", ylabel="v2", title="", zcolor=experimental_data[:, :t], color=:viridis)
Plots.plot!(plt, legend =false) 
#save the plot
Plots.savefig(plt, result_image_folder * "/experimental_data_in_new_basis.svg")


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

#to sample in the projected rectangle, sample randomly a little bit further and then exclude the points not included
min_y1 = min_y1 - 1.5 * abs(width)
max_y_1 = max_y_1 + 1.5 * abs(width)
min_y2 = min_y2 - 1.5 * abs(height)
max_y2 = max_y2 + 1.5 * abs(height)
min_y3 = min_y3 - 1.5 * abs(depth)
max_y3 = max_y3 + 1.5 * abs(depth)

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
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (r - u[3]) - u[2]
    du[3] = u[1] * u[2] - b * u[3]
    return du
end

# extract uniformely 100 points from the bounding box
# build a grid of 100 * 100 points
n_sampled_points = 1000000
function sample_uniform_2d(a, b, c, d)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    return (x, y)
end

points_sampled = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_sampled_points]

original_min_y1 = boundig_box_vect_field[1]
original_max_y1 = boundig_box_vect_field[2]
original_min_y2 = boundig_box_vect_field[3]
original_max_y2 = boundig_box_vect_field[4]
original_min_y3 = boundig_box_vect_field[5]
original_max_y3 = boundig_box_vect_field[6]

n_points= 50000
points = []
for point in points_sampled
    # convert the point to the new basis
    point_original_basis = convert_to_canonical_basis([point[1], point[2], 0.0])
    # check if the point is inside the bounding box
    if (point_original_basis[1] >= original_min_y1 && point_original_basis[1] <= original_max_y1) &&
       (point_original_basis[2] >= original_min_y2 && point_original_basis[2] <= original_max_y2) &&
       (point_original_basis[3] >= original_min_y3 && point_original_basis[3] <= original_max_y3)
        push!(points, point)
    end

    if length(points) >= n_points
        break
    end
end


# convert the points to the canonical basis
ground_truth_vect_field = [lorenz_ground_truth(convert_to_canonical_basis([p[1], p[2], 0.0])) for p in points]

# compute for each point the confidence interval
ensemble_1 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_1.jld")
ensemble_2 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_2.jld")
ensemble_3 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep_traj_3.jld")

ensemble_1 = [member for member in ensemble_1 if member.status == "success"]
ensemble_2 = [member for member in ensemble_2 if member.status == "success"]
ensemble_3 = [member for member in ensemble_3 if member.status == "success"]

ensemble_1 = ensemble_1[1:50]
ensemble_2 = ensemble_2[1:50]
ensemble_3 = ensemble_3[1:50]

ensemble = vcat(ensemble_1, ensemble_2, ensemble_3)

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
                u_reprojected = convert_to_canonical_basis([u[1], u[2], 0.0])
                û = appr_neural_network(u_reprojected, p, st)[1]
                du = similar(u_reprojected)
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

    ensemble_predictions = [model_function([point[1], point[2]], member.training_res.p) for member in ensemble]
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
        results = [get_analysis_ci(i, ensemble, nn_dimension) for i in 1:n_points]

        cicp = sum([r.gound_truth_in_Ci for r in results]) / n_points
        mean_area = mean([r.area for r in results])

        return (cicp=cicp, mean=[r.mean for r in results], ground_truth_in_Ci=[r.gound_truth_in_Ci for r in results], mean_area=mean_area, ellipse_areas=[r.area for r in results], ground_truths=[r.gound_truth for r in results])
    catch e
        @warn "Error in analyzing method: $(e)"
        return nothing
    end

end

ensemble_dimension = 5
ensembles = []

seed = 0
rng = StableRNG(seed)
ensembles = [ensemble[(k*ensemble_dimension+1):(k*ensemble_dimension+ensemble_dimension)] for k in 0:(10*3-1)]

# extract without replacement 100 seed form the ensemble (for 100 times)
#set the seed
seed = 0
rng = StableRNG(seed)

results_ensembles = [analyze_method(ensemble, 32) for ensemble in ensembles]
results_ensembles = [r for r in results_ensembles if r != nothing]

results = (points=points, results_ensemble=results_ensembles)
serialize(result_image_folder * "/results_analysis_" * string(ensemble_dimension) * ".jld", results)

results_deserialized = deserialize(result_image_folder * "/results_analysis_" * string(ensemble_dimension) * ".jld")
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

#revert order not to see the first point
points_to_plot = reverse(points_to_plot)
ground_truth_in_ensemble_to_plot = reverse(ground_truth_in_ensemble_to_plot)

alphas = (log10.(experimental_points_distances) .- minimum(log10.(experimental_points_distances))) ./ (maximum(log10.(experimental_points_distances)) - minimum(log10.(experimental_points_distances)))
alphas = 1 .- alphas

min_x = minimum([minimum(p[1]) for p in points_to_plot])
max_x = maximum([maximum(p[1]) for p in points_to_plot])
min_y = minimum([minimum(p[2]) for p in points_to_plot])
max_y = maximum([maximum(p[2]) for p in points_to_plot])

mean_cicp = 0.48

gr()
plt = Plots.plot()
Plots.scatter!(plt, [s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="training points", color=:orange)
Plots.scatter!(plt, [p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="v1", ylabel="v2", title="", zcolor=ground_truth_in_ensemble_to_plot, color=:viridis, markerstrokewidth=0, markersize=4)
Plots.scatter!(plt, [s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="", alpha=alphas, color=:orange)
#plot with the legend outside
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
Plots.savefig(plt, result_image_folder * "/ground_truth_in_ensemble_" * string(ensemble_dimension) * "_projected.svg")
Plots.savefig(plt, result_image_folder * "/ground_truth_in_ensemble_" * string(ensemble_dimension) * "_projected.png")



