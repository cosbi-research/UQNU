cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

gr()

rng = Random.default_rng()
Random.seed!(rng, 0)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[0.1, 2]
original_u0_1 = [1, 1]

reltol = 1e-10
abstol = 1e-10

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
#select only the trajectories with positive initial state 
pertubed_initial_conditions = pertubed_initial_conditions[:, (pertubed_initial_conditions[1, :] .> 0) .& (pertubed_initial_conditions[2, :] .> 0)]
original_u0_2 = pertubed_initial_conditions[:,1]
original_u0_3 = pertubed_initial_conditions[:,2]

initial_time_training = 0.0f0
end_time_training = 25.0f0
times = range(initial_time_training, end_time_training, step=0.25)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β = p
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, reltol=reltol, abstol=abstol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, reltol=reltol, abstol=abstol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, reltol=reltol, abstol=abstol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

plt = Plots.scatter(sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", legend=false, color = "blue")
Plots.scatter!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "blue")
Plots.scatter!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "blue")
Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "blue")
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "blue")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "blue")

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_no_noise.jld", df)

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

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]

# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, reltol=reltol, abstol=abstol)
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

serialize("damped_oscillator_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("damped_oscillator_trajectory_with_perturbed_iv_noisy.jld", noisy_df)

######################################### trajectory 2 #####################################################

rng = Random.default_rng()
Random.seed!(rng, 1)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[0.1, 2]
original_u0_1 = [1, 1]

reltol = 1e-10
abstol = 1e-10

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
#select only the trajectories with positive initial state 
pertubed_initial_conditions = pertubed_initial_conditions[:, (pertubed_initial_conditions[1, :] .> 0) .& (pertubed_initial_conditions[2, :] .> 0)]
original_u0_1 = pertubed_initial_conditions[:,1]
original_u0_2 = pertubed_initial_conditions[:,2]
original_u0_3 = pertubed_initial_conditions[:,3]

initial_time_training = 0.0f0
end_time_training = 25.0f0
times = range(initial_time_training, end_time_training, step=0.25)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β = p
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, reltol=reltol, abstol=abstol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, reltol=reltol, abstol=abstol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, reltol=reltol, abstol=abstol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

Plots.scatter!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", legend=false, color = "red")
Plots.scatter!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "red")
Plots.scatter!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "red")
Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "red")
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "red")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "red")

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_no_noise.jld", df)

x_1_values = df.x1
x_2_values = df.x2

min_y1 = minimum(x_1_values)
max_y1 = maximum(x_1_values)
min_y2 = minimum(x_2_values)
max_y2 = maximum(x_2_values)

y_1_dimension = max_y1 - min_y1
y_2_dimension = max_y2 - min_y2

exension_factor = 0.0

min_y1_new = min_y1 - exension_factor * y_1_dimension
max_y1_new = max_y1 + exension_factor * y_1_dimension
min_y2_new = min_y2 - exension_factor * y_2_dimension
max_y2_new = max_y2 + exension_factor * y_2_dimension

#save the bounding box
serialize("bounding_box_trajectory_damped_2.jld", [min_y1_new, max_y1_new, min_y2_new, max_y2_new])


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

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]

# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, reltol=reltol, abstol=abstol)
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

serialize("damped_oscillator_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("damped_oscillator_trajectory_with_perturbed_iv_noisy.jld", noisy_df)


#= ######################################### trajectory 3 #####################################################

rng = Random.default_rng()
Random.seed!(rng, 2)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[0.1, 2]
original_u0_1 = [1, 1]

reltol = 1e-10
abstol = 1e-10

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
#select only the trajectories with positive initial state 
pertubed_initial_conditions = pertubed_initial_conditions[:, (pertubed_initial_conditions[1, :] .> 0) .& (pertubed_initial_conditions[2, :] .> 0)]
original_u0_1 = pertubed_initial_conditions[:,1]
original_u0_2 = pertubed_initial_conditions[:,2]
original_u0_3 = pertubed_initial_conditions[:,3]

initial_time_training = 0.0f0
end_time_training = 25.0f0
times = range(initial_time_training, end_time_training, step=0.25)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β = p
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, reltol=reltol, abstol=abstol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, reltol=reltol, abstol=abstol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, reltol=reltol, abstol=abstol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

Plots.scatter!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", legend=false, color = "green")
Plots.scatter!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "green")
Plots.scatter!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "green")
Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "green")
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "green")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], label="Damped oscillator dynamics", xlabel="y1", ylabel="y2", color = "green")

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_no_noise.jld", df)

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

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], traj = repeat([3], length(times))))

#save the data
serialize("damped_oscillator_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]

# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, reltol=reltol, abstol=abstol)
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

serialize("damped_oscillator_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("damped_oscillator_trajectory_with_perturbed_iv_noisy.jld", noisy_df) =#

Plots.savefig(plt, "damped_oscillator_in_silico_data.png")















