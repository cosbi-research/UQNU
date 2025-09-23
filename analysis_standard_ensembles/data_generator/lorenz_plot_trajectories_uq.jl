cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

attempt_number = 1000
trajectory_number = 100
include("../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
experimental_data = deserialize("../data_generator/lorenz_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:attempt_number
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 3)
    push!(initial_points, [point.x1 * perturbation_1[1], point.x2 * perturbation_1[2], point.x3 * perturbation_1[3]])
end

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
#function to generate the Data
function lorenz!(du, u, p, t)
    σ, r, b = Float64[10, 28, 8/3]
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

simulations = []
for initial_point in initial_points[1:trajectory_number]
    u0 = initial_point
    tspan = (0.0, 0.2)
    #define the problem
    prob = ODEProblem(lorenz!, [u0[1], u0[2], u0[3]], tspan, original_parameters)

    sol = solve(prob, Tsit5(), saveat=0.002)

    df = DataFrame(sol, [:t, :x1, :x2, :x3])

    if size(df, 1) < 100
        @warn "Simulation for initial point $initial_point has less than 100 points, skipping"
        continue
    end

    #restrict the simulation to the bounding box
    push!(simulations, df)
end

plt = Plots.plot3d(legend=false, xlabel="x", ylabel="y", zlabel="z", title="", size=(800, 600))
for simulation in simulations
    Plots.plot!(plt, simulation.x1, simulation.x2, simulation.x3, label="", color=:red, alpha=0.5)
end

for traj in 1:3
    # Plot the bounding box
    x_series = experimental_data[experimental_data.traj .== traj, :x1]
    y_series = experimental_data[experimental_data.traj .== traj, :x2]
    z_series = experimental_data[experimental_data.traj .== traj, :x3]
    Plots.scatter!(plt, x_series, y_series, z_series, label="", color="blue")
end

Plots.savefig(plt, "lorenz_trajectories_test.svg")


