cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

trajectory_number = 100
attempts = 1000


#set the seed
seed = 0
rng = StableRNG(seed)

experimental_data = deserialize("lotka_volterra_in_silico_data_no_noise.jld")

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

# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]

function lokta_volterra!(du, u, p, t)
    α, β, γ, δ = original_parameters
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

simulations = []
for initial_point in initial_points[1:trajectory_number]
    u0 = initial_point
    tspan = (0.0, 10.0)
    #define the problem
    prob = ODEProblem(lokta_volterra!, [u0[1], u0[2]], tspan, original_parameters)

    sol = solve(prob, Tsit5(), saveat=0.01)

    if sol.retcode != :Success
        @warn "ODE solver did not converge for initial point $initial_point"
        continue
    end

    df = DataFrame(sol, [:t, :x1, :x2])

    if size(df, 1) < 100
        @warn "Simulation for initial point $initial_point has less than 100 points, skipping"
        continue
    end

    #restrict the simulation to the bounding box
    push!(simulations, df)
end

plt = Plots.plot(legend=false, xlabel="x", ylabel="y", title="", size=(800, 600))
for simulation in simulations
    Plots.plot!(plt, simulation.x1, simulation.x2, label="", color=:red, alpha=0.5)
end

for traj in 1:3
    # Plot the bounding box
    x_series = experimental_data[experimental_data.traj .== traj, :x1]
    y_series = experimental_data[experimental_data.traj .== traj, :x2]
    Plots.scatter!(plt, x_series, y_series, label="", color="blue")
end

Plots.savefig(plt, "lotka_volterra_trajectories_test.svg")
