cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

rng = Random.default_rng()
Random.seed!(rng, 10)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
original_u0_1 = [3, 1.5]

rel_tol = 1e-10
abs_tol = 1e-10

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
#select only the trajectories with positive initial state 
pertubed_initial_conditions = pertubed_initial_conditions[:, (pertubed_initial_conditions[1, :] .> 0) .& (pertubed_initial_conditions[2, :] .> 0)]

original_u0_2 = pertubed_initial_conditions[:,1]
original_u0_3 = pertubed_initial_conditions[:,2]

initial_time_training = 0.0f0
end_time_training = 10.0f0
times = range(initial_time_training, end_time_training, step=0.1)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, reltol=rel_tol, abstol=abs_tol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, reltol=rel_tol, abstol=abs_tol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, reltol=rel_tol, abstol=abs_tol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

plt = Plots.scatter(sol_as_array_1[1,:], sol_as_array_1[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.plot!(plt, legend=false)


Plots.savefig(plt, "lotka_volterra_in_silico_data.png")

y_1_total_simulations = vcat(sol_as_array_1[1,:], sol_as_array_2[1,:], sol_as_array_3[1,:])
y_2_total_simulations = vcat(sol_as_array_1[2,:], sol_as_array_2[2,:], sol_as_array_3[2,:])

min_y1 = minimum(y_1_total_simulations)
max_y1 = maximum(y_1_total_simulations)
min_y2 = minimum(y_2_total_simulations)
max_y2 = maximum(y_2_total_simulations)

width = max_y1 - min_y1
height = max_y2 - min_y2

expanding_factor = 0.2
bounding_box_epanded = [min_y1 - expanding_factor*width, max_y1 + expanding_factor*width, min_y2 - expanding_factor*height, max_y2 + expanding_factor*height]

#plot over the y_1_total_simulations
Plots.plot!(plt, [bounding_box_epanded[1], bounding_box_epanded[2]], [bounding_box_epanded[3], bounding_box_epanded[3]], label=false, color="black")
Plots.plot!(plt, [bounding_box_epanded[1], bounding_box_epanded[2]], [bounding_box_epanded[4], bounding_box_epanded[4]], label=false, color="black")
Plots.plot!(plt, [bounding_box_epanded[1], bounding_box_epanded[1]], [bounding_box_epanded[3], bounding_box_epanded[4]], label=false, color="black")
Plots.plot!(plt, [bounding_box_epanded[2], bounding_box_epanded[2]], [bounding_box_epanded[3], bounding_box_epanded[4]], label=false, color="black")

Plots.savefig(plt, "lotka_volterra_in_silico_data_with_bounding_box.png")

#save the bounding bounding_box_epanded
serialize("lotka_volterra_in_silico_data_bounding_box.jld", bounding_box_epanded)

expanding_factor = 0.5
bounding_box_epanded = [min_y1 - expanding_factor*width, max_y1 + expanding_factor*width, min_y2 - expanding_factor*height, max_y2 + expanding_factor*height]
serialize("lotka_volterra_in_silico_data_bounding_box_extended.jld", bounding_box_epanded)

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
sol = hcat(sol_as_array_1, sol_as_array_2, sol_as_array_3)
σ = 0.0
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

plt = Plots.scatter(sol_as_array_1_noisy[1,:], sol_as_array_1_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_2_noisy[1,:], sol_as_array_2_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_3_noisy[1,:], sol_as_array_3_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")


#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("lotka_volterra_in_silico_data_no_noise.jld", df)

# add a gaussian noise to the data
σ = 0.01
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

plt = Plots.scatter(sol_as_array_1_noisy[1,:], sol_as_array_1_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_2_noisy[1,:], sol_as_array_2_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")
Plots.scatter!(plt, sol_as_array_3_noisy[1,:], sol_as_array_3_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator")

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("lotka_volterra_in_silico_data_noisy.jld", df)

σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
#select only the trajectories with positive initial state 
pertubed_initial_conditions = pertubed_initial_conditions[:, (pertubed_initial_conditions[1, :] .> 0) .& (pertubed_initial_conditions[2, :] .> 0)]
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]

# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, reltol=rel_tol, abstol=abs_tol)
    # add a gaussian noise to the data
    σ = 0.0
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))

    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:], traj = repeat([i+1], length(sol.t)))
    #save the data
    pure_df = vcat(pure_df, df)

    σ = 0.05
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:], traj = repeat([i+1], length(sol.t)))
    #save the data
    noisy_df = vcat(noisy_df, df)
end

# plots the trajectories (pure)) together with the original one 
plt = Plots.plot()
for traj in unique(pure_df.traj)[1:20]
    traj = Int(traj)
    Plots.scatter!(plt, pure_df[pure_df.traj .== traj, :x1], pure_df[pure_df.traj .== traj, :x2], label=false, color="blue")
end

Plots.scatter!(plt, sol_as_array_1_noisy[1,:], sol_as_array_1_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator", color = "red")
Plots.scatter!(plt, sol_as_array_2_noisy[1,:], sol_as_array_2_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator", color= "red" )
Plots.scatter!(plt, sol_as_array_3_noisy[1,:], sol_as_array_3_noisy[2, :], label="Lotka-Volterra dynamics", xlabel="Prey", ylabel="Predator", color= "red")


serialize("lotka_volterra_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("lotka_volterra_trajectory_with_perturbed_iv_noisy.jld", noisy_df)

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=0.001, reltol=rel_tol, abstol=abs_tol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=0.001, reltol=rel_tol, abstol=abs_tol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=0.001, reltol=rel_tol, abstol=abs_tol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

times = sol_1.t

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1[1,:], x2 = sol_as_array_1[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2[1,:], x2 = sol_as_array_2[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3[1,:], x2 = sol_as_array_3[2,:], traj = repeat([3], length(times))))

#save the data
serialize("lotka_volterra_in_silico_data_no_noise_pure_trajectory.jld", df)



