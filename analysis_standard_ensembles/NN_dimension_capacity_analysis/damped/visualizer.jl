# Script to visualize the error on the vector field and the trajectories of the Lorenz system
cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations, Plots

rng = MersenneTwister()
Random.seed!(rng, 0)

#gets the bounding box of the vector field where I want to approximate the vector field
boundig_box_vect_field = deserialize("../../data_generator/damped_oscillator_in_silico_data_bounding_box.jld")
min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

# Define the Lotka Volterra vector field
original_parameters = Float64[0.1, 2]#function to generate the Data
function damped_oscillator_ground_truth(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
    return du
end

tstate = deserialize("trained_model.jld")

function sample_uniform_2d(a, b, c, d)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    return [x, y]
end

n_points = 100000
points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]
ground_truth_vect_field = [damped_oscillator_ground_truth([p[1], p[2]]) for p in points]

points_as_matrix = hcat(points...)
ground_truth_vect_field_as_matrix = hcat(ground_truth_vect_field...)

y_pred = Lux.apply(tstate.model, points_as_matrix, tstate.parameters, tstate.states)[1]
abs_error_on_vec_field = log10.(vec(sum(abs2.(y_pred .- ground_truth_vect_field_as_matrix), dims=1) ./ max.(sum(abs2.(ground_truth_vect_field_as_matrix), dims=1), 0.01)))

x_to_plot = points_as_matrix[1, :]
y_to_plot = points_as_matrix[2, :]
err_to_plot = abs_error_on_vec_field

avg_relative_error = mean(abs_error_on_vec_field)
#print it in a csv file
CSV.write("average_relative_error_vector_field.csv", DataFrame(average_relative_error=avg_relative_error))

#insert a fake point at the beginning (to visualize the error from -10 to 0)
first_x = x_to_plot[1]
first_y = y_to_plot[1]

x_to_plot = vcat(first_x, x_to_plot)
y_to_plot = vcat(first_y, y_to_plot)
err_to_plot = vcat(-10.0, err_to_plot)

x_to_plot = vcat(first_x, x_to_plot)
y_to_plot = vcat(first_y, y_to_plot)
err_to_plot = vcat(0.0, err_to_plot)

err_to_plot = max.(err_to_plot, -10.0)  # Ensure no values are below -10 for visualization

plt = Plots.scatter(x_to_plot, y_to_plot, zcolor=err_to_plot, xlabel="x", ylabel="y", title="", markerstrokewidth=0, label="", color=:viridis, dpi=600)
Plots.xlims!(min_y1, max_y_1)
Plots.ylims!(min_y2, max_y2)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)

#save the plot
Plots.savefig(plt, "error_on_vector_field.png")


######################################################### ERROR ON TRAJECTORIES #########################################################
#generate 1000 starting point in the vector field
#set the seed
seed = 0
rng = StableRNG(seed)

trajectory_number = 100
attempts = 1000

#bounding box of interest 
experimental_data = deserialize("../../data_generator/damped_oscillator_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:attempts
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 2)

    initial_x1 = point.x1 * perturbation_1[1]
    initial_x2 = point.x2 * perturbation_1[2]

    #ensure they are greater than 0
    if initial_x1 < 0 || initial_x2 < 0
        continue
    end

    push!(initial_points, [initial_x1, initial_x2])
end
starting_points = initial_points[1:trajectory_number]
initial_points = initial_points[1:trajectory_number]

tspan = (0.0, 25.0)
function damped!(du, u, p, t)
    α, β = original_parameters
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

simulations = []
for i in 1:length(starting_points)
    starting_point = starting_points[i]
    prob = ODEProblem(damped!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.25, abstol=1e-7, reltol=1e-6)

    push!(simulations, sol)
end

function damped_neural_node!(du, u, p, t)
    du .= Lux.apply(tstate.model, u, tstate.parameters, tstate.states)[1]
end

simulations_neural_ode = []
for i in 1:length(starting_points) 
    starting_point = starting_points[i]

    simulation_deterministic = simulations[i]
    tspan = (0.0, simulation_deterministic.t[end])

    prob = ODEProblem(damped_neural_node!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.25, abstol=1e-7, reltol=1e-6)

    push!(simulations_neural_ode, sol)
end

# average the squared error over the simulations
squared_errors = []
for i in 1:length(simulations)
    sol = simulations[i]
    sol_neural_ode = simulations_neural_ode[i]
    squared_error = mean(abs2.(Array(sol) .- Array(sol_neural_ode)) ./ max.(abs2.(Array(sol)), 0.0001))
    push!(squared_errors, squared_error)
end

println("Average squared error: ", mean(squared_errors))
mean_squared_error_rounded = @sprintf("%.3e", mean(squared_errors))
#plot an histogram with the squared errors
plt = Plots.histogram(log10.(squared_errors), bins=50, xlabel="Squared error (log10)", ylabel="Frequency", title="Squared errors in trajectories (mean: $mean_squared_error_rounded)", label = "")
Plots.savefig(plt, "error_on_trajectories.png")

#save the average squared error 
CSV.write("average_squared_error.csv", DataFrame(mean_squared_error=mean(squared_errors)))



####################################################### visualizer #########################################################
#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:attempts
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 2)

    initial_x1 = point.x1 * perturbation_1[1]
    initial_x2 = point.x2 * perturbation_1[2]

    #ensure they are greater than 0
    if initial_x1 < 0 || initial_x2 < 0
        continue
    end

    push!(initial_points, [initial_x1, initial_x2])
end
initial_points = initial_points[1:trajectory_number]

tspan = (0.0, 25.0)
function damped!(du, u, p, t)
    α, β = original_parameters
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

simulations = []
for i in 1:length(starting_points)
    starting_point = starting_points[i]
    prob = ODEProblem(damped!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.001, abstol=1e-7, reltol=1e-6)

    push!(simulations, sol)
end

function damped_neural_node!(du, u, p, t)
    du .= Lux.apply(tstate.model, u, tstate.parameters, tstate.states)[1]
end

simulations_neural_ode = []
for i in 1:length(starting_points) 
    starting_point = starting_points[i]

    simulation_deterministic = simulations[i]
    tspan = (0.0, simulation_deterministic.t[end])

    prob = ODEProblem(damped_neural_node!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.001, abstol=1e-7, reltol=1e-6)

    push!(simulations_neural_ode, sol)
end

#plot the deterministic and neural ODE solution
# plot the deterministic and the node solution for the first simulation
plt = Plots.plot(simulations[1], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[1], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_1.svg")

plt = Plots.plot(simulations[2], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[2], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_2.svg")

plt = Plots.plot(simulations[3], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[3], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_3.svg")

plt = Plots.plot(simulations[4], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[4], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_4.svg")  

plt = Plots.plot(simulations[5], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[5], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_5.svg")

plt = Plots.plot(simulations[6], vars=(1, 2), label="Ground truth", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright)
Plots.plot!(simulations_neural_ode[6], vars=(1, 2), label="NODE")
Plots.plot!( 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright
)

Plots.savefig(plt, "trajectory_6.svg")