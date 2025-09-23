# Script to visualize the error on the vector field and the trajectories of the Lorenz system
cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations, Plots

rng = MersenneTwister()
Random.seed!(rng, 0)

#gets the bounding box of the vector field where I want to approximate the vector field
boundig_box_vect_field_1 = deserialize("bounding_box_trajectory_damped_1.jld")
boundig_box_vect_field_2 = deserialize("bounding_box_trajectory_damped_2.jld")
boundig_box_vect_field_3 = deserialize("bounding_box_trajectory_damped_3.jld")

min_y1 = min(boundig_box_vect_field_1[1], boundig_box_vect_field_2[1], boundig_box_vect_field_3[1])
max_y1 = max(boundig_box_vect_field_1[2], boundig_box_vect_field_2[2], boundig_box_vect_field_3[2])
min_y2 = min(boundig_box_vect_field_1[3], boundig_box_vect_field_2[3], boundig_box_vect_field_3[3])
max_y2 = max(boundig_box_vect_field_1[4], boundig_box_vect_field_2[4], boundig_box_vect_field_3[4])

dim_y1 = max_y1 - min_y1
dim_y2 = max_y2 - min_y2

extension_factor = 0.2
min_y1_new = min_y1 - extension_factor * dim_y1
max_y1_new = max_y1 + extension_factor * dim_y1
min_y2_new = min_y2 - extension_factor * dim_y2
max_y2_new = max_y2 + extension_factor * dim_y2

min_y1 = min_y1_new
max_y1 = max_y1_new
min_y2 = min_y2_new
max_y2 = max_y2_new

#save the bounding box on vector field
bounding_box_vec_field = (min_y1, max_y1, min_y2, max_y2)
serialize("bounding_box_vec_field.jld", bounding_box_vec_field)

# Define the Lotka Volterra vector field
original_parameters = Float64[0.1, 2]#function to generate the Data
function damped_oscillator_ground_truth(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
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
initial_points = initial_points[1:trajectory_number]

tspan = (0.0, 25.0)
function damped!(du, u, p, t)
    α, β = original_parameters
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

simulations = []
for i in 1:length(initial_points)
    starting_point = initial_points[i]
    prob = ODEProblem(damped!, starting_point, tspan, original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.25, abstol=1e-8, reltol=1e-8)

    push!(simulations, sol)
end


y1_values = []
y2_values = []

for simulation in simulations
    simulation_states = simulation.u

    append!(y1_values, [res[1] for res in simulation_states])
    append!(y2_values, [res[2] for res in simulation_states])
end

min_y1_trajectories = minimum(y1_values)
max_y1_trajectories = maximum(y1_values)
min_y2_trajectories = minimum(y2_values)
max_y2_trajectories = maximum(y2_values)

min_y1 = min(min_y1, min_y1_trajectories)
max_y1 = max(max_y1, max_y1_trajectories)
min_y2 = min(min_y2, min_y2_trajectories)
max_y2 = max(max_y2, max_y2_trajectories)

#write it as jld file
bounding_box = (min_y1, max_y1, min_y2, max_y2)
serialize("bounding_box_to_train.jld", bounding_box)