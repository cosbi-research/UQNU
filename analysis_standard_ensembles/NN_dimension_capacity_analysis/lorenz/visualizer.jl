# Script to visualize the error on the vector field and the trajectories of the Lorenz system
cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations, Plots, StableRNGs, DataFrames, CSV, Distributions

rng = MersenneTwister()
Random.seed!(rng, 0)

#boundig_box_vect_field = deserialize("../../data_generator/lorenz_in_silico_data_bounding_box.jld")
boundig_box_vect_field = deserialize("bounding_box_to_train.jld")


#boundig_box_vect_field[5] = boundig_box_vect_field_old[5]
#boundig_box_vect_field[6] = boundig_box_vect_field_old[6]

#boundig_box_vect_field = [0, 20, 0, 20, 0, 20]
width= boundig_box_vect_field[2] - boundig_box_vect_field[1]
height = boundig_box_vect_field[4] - boundig_box_vect_field[3]
depth = boundig_box_vect_field[6] - boundig_box_vect_field[5]

min_y1, max_y1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

#enlarge the dimensions for visualization because the network has been trained on
# a larger domain
width = width * 1.2
height = height * 1.2
depth = depth * 1.2

boundig_box_vect_field = deserialize("bounding_box_vec_field.jld")

# Define the Lorenz vector field
original_parameters = Float64[10, 28, 8/3]
function lorenz_ground_truth(u)
    σ, r, b = original_parameters
    du = similar(u)
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
    return du
end

# Load the trained model (reassign opt only )
tstate = deserialize("trained_model.jld")

# Define the grid for the vector field to be plotted
grid_sampling = 100
# create a grid (tensor in 3d) of points in which each point is on the bounding box
x = collect(range(min_y1, max_y1, length=grid_sampling))
y = collect(range(min_y2, max_y2, length=grid_sampling))
z = collect(range(min_y3, max_y3, length=grid_sampling))
grid = zeros(Float32, 3, grid_sampling, grid_sampling, grid_sampling)
for i in 1:grid_sampling
    for j in 1:grid_sampling
        for k in 1:grid_sampling
            grid[:, i, j, k] = [x[i], y[j], z[k]]
        end
    end
end

# populate the frist grid with the ground truth vector field
ground_truth_vect_field_y1 = [lorenz_ground_truth([grid[:, i, j, k][1], grid[:, i, j, k][2], grid[:, i, j, k][3]])[1] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]
ground_truth_vect_field_y2 = [lorenz_ground_truth([grid[:, i, j, k][1], grid[:, i, j, k][2], grid[:, i, j, k][3]])[2] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]
ground_truth_vect_field_y3 = [lorenz_ground_truth([grid[:, i, j, k][1], grid[:, i, j, k][2], grid[:, i, j, k][3]])[3] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]

neural_vector_field_y1 = [Lux.apply(tstate.model, vec(grid[:, i, j, k]) ./ [width, height, depth], tstate.parameters, tstate.states)[1][1] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]
neural_vector_field_y2 = [Lux.apply(tstate.model, vec(grid[:, i, j, k]) ./ [width, height, depth], tstate.parameters, tstate.states)[1][2] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]
neural_vector_field_y3 = [Lux.apply(tstate.model, vec(grid[:, i, j, k]) ./ [width, height, depth], tstate.parameters, tstate.states)[1][3] for i in 1:grid_sampling, j in 1:grid_sampling, k in 1:grid_sampling]

#populate the second grid with the neural ODE vector field
neural_vector_field_absolute_error = zeros(Float32, grid_sampling, grid_sampling, grid_sampling)
for i in 1:grid_sampling
    for j in 1:grid_sampling
        for k in 1:grid_sampling
            #neural_vector_field_absolute_error[i, j, k] = sum(abs2.(neural_vector_field_y1[i, j, k] .- ground_truth_vect_field_y1[i, j, k]) .+ abs2.(neural_vector_field_y2[i, j, k] .- ground_truth_vect_field_y2[i, j, k]) .+ abs2.(neural_vector_field_y3[i, j, k] .- ground_truth_vect_field_y3[i, j, k]) ./ sum(abs2.([ground_truth_vect_field_y1[i, j, k], ground_truth_vect_field_y2[i, j, k], ground_truth_vect_field_y3[i, j, k]]))) 
            neural_vector_field_absolute_error[i, j, k] = sum(abs2.(neural_vector_field_y1[i, j, k] .- ground_truth_vect_field_y1[i, j, k]) .+ abs2.(neural_vector_field_y2[i, j, k] .- ground_truth_vect_field_y2[i, j, k]) .+ abs2.(neural_vector_field_y3[i, j, k] .- ground_truth_vect_field_y3[i, j, k])) / sum(abs2.([ground_truth_vect_field_y1[i, j, k], ground_truth_vect_field_y2[i, j, k], ground_truth_vect_field_y3[i, j, k]]))
        end
    end
end

mean_error_on_vector_field = mean(neural_vector_field_absolute_error)
println("Mean error on vector field: ", mean_error_on_vector_field)
# Save the mean error on vector field
CSV.write("mean_error_on_vector_field.csv", DataFrame(mean_error=mean_error_on_vector_field))

#################### Plots the max relative error collapsed on x y ################################
max_error = maximum(neural_vector_field_absolute_error, dims=3)
points_to_plot = zeros(Float32, 2, grid_sampling, grid_sampling)
for i in 1:grid_sampling
    for j in 1:grid_sampling
        points_to_plot[:, i, j] = [x[i], y[j]]
    end
end

# Flatten points and errors
x_coords = reshape(points_to_plot[1, :, :], :)
y_coords = reshape(points_to_plot[2, :, :], :)
errors = reshape(max_error, :)  # Assuming max_error is (grid_sampling, grid_sampling)

#add fake point 
first_x = x_coords[1]
first_y = y_coords[1]

x_coords = vcat(first_x, x_coords)
y_coords = vcat(first_y, y_coords)
x_coords = vcat(first_x, x_coords)
y_coords = vcat(first_y, y_coords)

errors = vcat(1.0, errors)
errors = vcat(10.0^(-10), errors)

min_x = minimum(x_coords)
min_y = minimum(y_coords)
max_x = maximum(x_coords)
max_y = maximum(y_coords)

# Scatter plot
plt = Plots.scatter(
    x_coords, y_coords,
    zcolor=log10.(errors),  # Use error values for color intensity
    c=:viridis,      # Colormap
    xlabel="x",
    ylabel="y",
    label = "",
    title="",
    markerstrokewidth = 0
)
Plots.xlims!(plt, min_x, max_x)
Plots.ylims!(plt, min_y, max_y)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)

Plots.savefig(plt, "error_on_vector_field_x_y.png")

#= #################### Plots the max relative error collapsed on x z ################################
max_error = maximum(neural_vector_field_absolute_error, dims=2)
points_to_plot = zeros(Float32, 2, grid_sampling, grid_sampling)
for i in 1:grid_sampling
    for j in 1:grid_sampling
        points_to_plot[:, i, j] = [x[i], z[j]]
    end
end

# Flatten points and errors
x_coords = reshape(points_to_plot[1, :, :], :)
y_coords = reshape(points_to_plot[2, :, :], :)

errors = reshape(max_error, :) 

# Scatter plot
Plots.scatter(
    x_coords, y_coords,
    zcolor=log10.(errors),  # Use error values for color intensity
    c=:viridis,      # Colormap
    xlabel="X",
    ylabel="Z",
    label = "",
    title="Max absolute error (collapsed on X and Z)",
    markerstrokewidth = 0
)

Plots.savefig("error_on_vector_field_x_z.png")

#################### Plots the max relative error collapsed on y z ################################
max_error = maximum(neural_vector_field_absolute_error, dims=1)

points_to_plot = zeros(Float32, 2, grid_sampling, grid_sampling)
for i in 1:grid_sampling
    for j in 1:grid_sampling
        points_to_plot[:, i, j] = [y[i], z[j]]
    end
end

# Flatten points and errors
x_coords = reshape(points_to_plot[1, :, :], :)
y_coords = reshape(points_to_plot[2, :, :], :)

errors = reshape(max_error, :)  # Assuming max_error is (grid_sampling, grid_sampling)

# Scatter plot
Plots.scatter(
    x_coords, y_coords,
    zcolor=log10.(errors),  # Use error values for color intensity
    c=:viridis,      # Colormap
    xlabel="Y",
    ylabel="Z",
    label = "",
    title="Max absolute error (collapsed on Y and Z)",
    markerstrokewidth = 0
)

Plots.savefig("error_on_vector_field_y_z.png")
 =#
######################################### trajectory error ############################
trajectory_number = 100
threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

tspan = (0.0, 0.2)

original_parameters = Float64[10, 28, 8/3]
function lorenz!(du, u, p, t)
    σ, r, b = original_parameters
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

#bounding box of interest 
experimental_data = deserialize("../../data_generator/lorenz_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:trajectory_number
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 3)
    push!(initial_points, [point.x1 * perturbation_1[1], point.x2 * perturbation_1[2], point.x3 * perturbation_1[3]])
end

starting_points = initial_points

simulations = []
#filter the simulations inside the vector field
for i in 1:length(starting_points)
    starting_point = starting_points[i]
    prob = ODEProblem(lorenz!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.002, abstol=1e-8, reltol=1e-8)

    push!(simulations, sol)
end

function lorenz_neural_ode!(du, u, p, t)
    u_rescaled = u ./ [width, height, depth]
    du .= Lux.apply(tstate.model, u_rescaled , tstate.parameters, tstate.states)[1] 
end

simulations_neural_ode = []
for i in 1:length(starting_points) 
    starting_point = starting_points[i]

    simulation_deterministic = simulations[i]

    prob = ODEProblem(lorenz_neural_ode!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.002, abstol=1e-7, reltol=1e-6)

    push!(simulations_neural_ode, sol)
end

#plot the deterministic and neural ODE solution
# plot the deterministic and the node solution for the first simulation
plt = Plots.plot(simulations[1], vars=(1, 2, 3), label="Deterministic", xlabel="y1", ylabel="y2", zlabel="y3", title="Deterministic vs Neural ODE")
Plots.plot!(simulations_neural_ode[1], vars=(1, 2, 3), label="Neural ODE")

plt = Plots.plot(simulations[2], vars=(1, 2, 3), label="Deterministic", xlabel="y1", ylabel="y2", zlabel="y3", title="Deterministic vs Neural ODE")
Plots.plot!(simulations_neural_ode[2], vars=(1, 2, 3), label="Neural ODE")

# average the squared error over the simulations
squared_errors = []
for i in 1:length(simulations)
    sol = simulations[i]
    sol_neural_ode = simulations_neural_ode[i]
    squared_error = mean(abs2.(Array(sol) .- Array(sol_neural_ode)) ./ max.(abs2.(Array(sol)), 0.01))
    push!(squared_errors, squared_error)
end

#plot the distribution of the squared error over the trajectories
println("Average squared error: ", mean(squared_errors))
mean_squared_error_rounded = @sprintf("%.3e", mean(squared_errors))
#plot an histogram with the squared errors
plt = Plots.histogram(log10.(squared_errors), bins=50, xlabel="Squared error (log10)", ylabel="Frequency", title="Squared errors on trajectories (mean: $mean_squared_error_rounded)", label="")
Plots.savefig(plt, "error_on_trajectories.png")

#save the average squared error
CSV.write("average_squared_error.csv", DataFrame(mean_squared_error=mean(squared_errors)))


############################ plots the trajectories #########################################
trajectory_number = 100
threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

tspan = (0.0, 0.2)

original_parameters = Float64[10, 28, 8/3]
function lorenz!(du, u, p, t)
    σ, r, b = original_parameters
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

#bounding box of interest 
experimental_data = deserialize("../../data_generator/lorenz_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:trajectory_number
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 3)
    push!(initial_points, [point.x1 * perturbation_1[1], point.x2 * perturbation_1[2], point.x3 * perturbation_1[3]])
end

starting_points = initial_points

simulations = []
#filter the simulations inside the vector field
for i in 1:length(starting_points)
    starting_point = starting_points[i]
    prob = ODEProblem(lorenz!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.002, abstol=1e-8, reltol=1e-8)

    push!(simulations, sol)
end

function lorenz_neural_ode!(du, u, p, t)
    u_rescaled = u ./ [width, height, depth]
    du .= Lux.apply(tstate.model, u_rescaled , tstate.parameters, tstate.states)[1] 
end

simulations_neural_ode = []
for i in 1:length(starting_points) 
    starting_point = starting_points[i]

    simulation_deterministic = simulations[i]

    prob = ODEProblem(lorenz_neural_ode!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.002, abstol=1e-7, reltol=1e-6)

    push!(simulations_neural_ode, sol)
end

#plot the deterministic and neural ODE solution
# plot the deterministic and the node solution for the first simulation
plt = Plots.plot(simulations[1], vars=(1, 2, 3), label="Ground truth", xlabel="y1", ylabel="y2", zlabel="y3", title="", dpi=600)
Plots.plot!(simulations_neural_ode[1], vars=(1, 2, 3), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    zguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    ztickfont=font(12),    # Increase y-axis label font size
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_1.svg")

plt = Plots.plot(simulations[2], vars=(1, 2, 3), label="Ground truth", xlabel="y1", ylabel="y2", zlabel="y3", title="", dpi=600)
Plots.plot!(simulations_neural_ode[2], vars=(1, 2, 3), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    zguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    ztickfont=font(12),    # Increase y-axis label font size
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_2.svg")

plt = Plots.plot(simulations[3], vars=(1, 2, 3), label="Ground truth", xlabel="y1", ylabel="y2", zlabel="y3", title="", dpi=600)
Plots.plot!(simulations_neural_ode[3], vars=(1, 2, 3), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    zguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    ztickfont=font(12),    # Increase y-axis label font size
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_3.svg")

plt = Plots.plot(simulations[4], vars=(1, 2, 3), label="Ground truth", xlabel="y1", ylabel="y2", zlabel="y3", title="", dpi=600)
Plots.plot!(simulations_neural_ode[4], vars=(1, 2, 3), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    zguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    ztickfont=font(12),    # Increase y-axis label font size
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_4.svg")

plt = Plots.plot(simulations[5], vars=(1, 2, 3), label="Ground truth", xlabel="y1", ylabel="y2", zlabel="y3", title="", dpi=600)
Plots.plot!(simulations_neural_ode[5], vars=(1, 2, 3), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    zguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    ztickfont=font(12),    # Increase y-axis label font size
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_5.svg")


