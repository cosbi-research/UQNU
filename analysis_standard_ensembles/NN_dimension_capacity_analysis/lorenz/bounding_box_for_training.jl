# Script to visualize the error on the vector field and the trajectories of the Lorenz system
cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations, Plots

rng = MersenneTwister()
Random.seed!(rng, 0)

boundig_box_vect_field_traj_1 = deserialize("bounding_box_trajectory_lorenz_1.jld")
boundig_box_vect_field_traj_2 = deserialize("bounding_box_trajectory_lorenz_2.jld")
boundig_box_vect_field_traj_3 = deserialize("bounding_box_trajectory_lorenz_3.jld")

min_y1 = min(boundig_box_vect_field_traj_1[1], boundig_box_vect_field_traj_2[1], boundig_box_vect_field_traj_3[1])
max_y1 = max(boundig_box_vect_field_traj_1[2], boundig_box_vect_field_traj_2[2], boundig_box_vect_field_traj_3[2])
min_y2 = min(boundig_box_vect_field_traj_1[3], boundig_box_vect_field_traj_2[3], boundig_box_vect_field_traj_3[3])
max_y2 = max(boundig_box_vect_field_traj_1[4], boundig_box_vect_field_traj_2[4], boundig_box_vect_field_traj_3[4])
min_y3 = min(boundig_box_vect_field_traj_1[5], boundig_box_vect_field_traj_2[5], boundig_box_vect_field_traj_3[5])
max_y3 = max(boundig_box_vect_field_traj_1[6], boundig_box_vect_field_traj_2[6], boundig_box_vect_field_traj_3[6])

dim_y1 = max_y1 - min_y1
dim_y2 = max_y2 - min_y2
dim_y3 = max_y3 - min_y3

extension_factor = 0.2
min_y1_new = min_y1 - extension_factor * dim_y1
max_y1_new = max_y1 + extension_factor * dim_y1
min_y2_new = min_y2 - extension_factor * dim_y2
max_y2_new = max_y2 + extension_factor * dim_y2
min_y3_new = min_y3 - extension_factor * dim_y3
max_y3_new = max_y3 + extension_factor * dim_y3


min_y1 = min_y1_new 
max_y1 = max_y1_new
min_y2 = min_y2_new
max_y2 = max_y2_new
min_y3 = min_y3_new
max_y3 = max_y3_new

#save the bounding box on vector field
bounding_box_vec_field = (min_y1, max_y1, min_y2, max_y2, min_y3, max_y3)
serialize("bounding_box_vec_field.jld", bounding_box_vec_field)

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

######################################################### ERROR ON TRAJECTORIES #########################################################
#generate 1000 starting point in the vector field
#set the seed
seed = 0
rng = StableRNG(seed)

trajectory_number = 100
attempts = 1000

#bounding box of interest 
experimental_data = deserialize("../../data_generator/lorenz_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)

initial_points = []
for i in 1:attempts
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 3)

    initial_x1 = point.x1 * perturbation_1[1]
    initial_x2 = point.x2 * perturbation_1[2]
    initial_x3 = point.x3 * perturbation_1[3]

    push!(initial_points, [initial_x1, initial_x2, initial_x3])
end
initial_points = initial_points[1:trajectory_number]

tspan = (0.0, 0.2)
function lorenz!(du, u, p, t)
    σ, r, b = original_parameters
    du = similar(u)
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

simulations = []
for i in 1:length(initial_points)
    starting_point = initial_points[i]
    prob = ODEProblem(lorenz!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=1e-7, reltol=1e-6)

    push!(simulations, sol)
end


y1_values = []
y2_values = []
y3_values = []

for simulation in simulations
    simulation_states = simulation.u

    append!(y1_values, [res[1] for res in simulation_states])
    append!(y2_values, [res[2] for res in simulation_states])
    append!(y3_values, [res[3] for res in simulation_states])
end

min_y1_trajectories = minimum(y1_values)
max_y1_trajectories = maximum(y1_values)
min_y2_trajectories = minimum(y2_values)
max_y2_trajectories = maximum(y2_values)
min_y3_trajectories = minimum(y3_values)
max_y3_trajectories = maximum(y3_values)

min_y1 = min(min_y1, min_y1_trajectories)
max_y1 = max(max_y1, max_y1_trajectories)
min_y2 = min(min_y2, min_y2_trajectories)
max_y2 = max(max_y2, max_y2_trajectories)
min_y3 = min(min_y3, min_y3_trajectories)
max_y3 = max(max_y3, max_y3_trajectories)

#write it as jld file
bounding_box = (min_y1, max_y1, min_y2, max_y2, min_y3, max_y3)
serialize("bounding_box_to_train.jld", bounding_box)