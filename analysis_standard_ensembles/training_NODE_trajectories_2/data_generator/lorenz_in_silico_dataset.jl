cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

gr()

abs_tol = 1e-10
rel_tol = 1e-10

#################################### trajectory 1 ####################################

rng = Random.default_rng()
Random.seed!(rng, 0)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
original_u0_1 = [1.4671326612976134,-4.435365392396897,28.70998129826181]

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)

original_u0_2 = pertubed_initial_conditions[:,1]
original_u0_3 = pertubed_initial_conditions[:,2]


initial_time_training = 0.0f0
end_time_training = 2.0f0
times = range(initial_time_training, end_time_training, step=0.02)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    σ, r, b = p
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, abstol=abs_tol, reltol=rel_tol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

plt = Plots.plot(sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:blue)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color =:blue)
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:blue)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:blue)
#Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_1[1,:]], [sol_as_array_1[2, :]], [sol_as_array_1[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:blue)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_2[1,:]], [sol_as_array_2[2, :]], [sol_as_array_2[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:blue)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_3[1,:]], [sol_as_array_3[2, :]], [sol_as_array_3[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:blue)

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))

#save the data
serialize("lorenz_in_silico_data_no_noise.jld", df)

# add a gaussian noise to the data
σ = 0.01
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))
#save the data
serialize("lorenz_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]
#select only the trajectories with positive initial state 
# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=abs_tol, reltol=rel_tol)
    # add a gaussian noise to the data
    σ = 0.0
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:],  x3 = sol_noisy[3,:],  traj = repeat([i+1], length(sol.t)))
    #save the data
    pure_df = vcat(pure_df, df)

    σ = 0.05
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:], x3 = sol_noisy[3,:], traj = repeat([i+1], length(sol.t)))
    #save the data
    noisy_df = vcat(noisy_df, df)
end

serialize("lorenz_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("lorenz_trajectory_with_perturbed_iv_noisy.jld", noisy_df)

#################################### trajectory 2 ####################################

rng = Random.default_rng()
Random.seed!(rng, 10)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
original_u0_1 = [1.4671326612976134,-4.435365392396897,28.70998129826181]

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)

original_u0_1 = pertubed_initial_conditions[:,1]
original_u0_2 = pertubed_initial_conditions[:,2]
original_u0_3 = pertubed_initial_conditions[:,3]


initial_time_training = 0.0f0
end_time_training = 2.0f0
times = range(initial_time_training, end_time_training, step=0.02)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    σ, r, b = p
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, abstol=abs_tol, reltol=rel_tol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:red)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color =:blue)
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:red)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:red)
#Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_1[1,:]], [sol_as_array_1[2, :]], [sol_as_array_1[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:red)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_2[1,:]], [sol_as_array_2[2, :]], [sol_as_array_2[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:red)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_3[1,:]], [sol_as_array_3[2, :]], [sol_as_array_3[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:red)

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))

#save the data
serialize("lorenz_in_silico_data_no_noise.jld", df)

x_1_values = df.x1
x_2_values = df.x2
x_3_values = df.x3


min_y1 = minimum(x_1_values)
max_y1 = maximum(x_1_values)
min_y2 = minimum(x_2_values)
max_y2 = maximum(x_2_values)
min_y3 = minimum(x_3_values)
max_y3 = maximum(x_3_values)

y_1_dimension = max_y1 - min_y1
y_2_dimension = max_y2 - min_y2
y_3_dimension = max_y3 - min_y3

exension_factor = 0.0   

min_y1_new = min_y1 - exension_factor * y_1_dimension
max_y1_new = max_y1 + exension_factor * y_1_dimension
min_y2_new = min_y2 - exension_factor * y_2_dimension
max_y2_new = max_y2 + exension_factor * y_2_dimension
min_y3_new = min_y3 - exension_factor * y_3_dimension
max_y3_new = max_y3 + exension_factor * y_3_dimension

#save the bounding box
serialize("bounding_box_trajectory_lorenz_2.jld", [min_y1_new, max_y1_new, min_y2_new, max_y2_new, min_y3_new, max_y3_new])


# add a gaussian noise to the data
σ = 0.01
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))
#save the data
serialize("lorenz_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]
#select only the trajectories with positive initial state 
# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=abs_tol, reltol=rel_tol)
    # add a gaussian noise to the data
    σ = 0.0
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:],  x3 = sol_noisy[3,:],  traj = repeat([i+1], length(sol.t)))
    #save the data
    pure_df = vcat(pure_df, df)

    σ = 0.05
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:], x3 = sol_noisy[3,:], traj = repeat([i+1], length(sol.t)))
    #save the data
    noisy_df = vcat(noisy_df, df)
end

serialize("lorenz_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("lorenz_trajectory_with_perturbed_iv_noisy.jld", noisy_df)


#= #################################### trajectory 3 ####################################
rng = Random.default_rng()
Random.seed!(rng, 2)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
original_u0_1 = [1.4671326612976134,-4.435365392396897,28.70998129826181]

σ = 0.1
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)

original_u0_1 = pertubed_initial_conditions[:,1]
original_u0_2 = pertubed_initial_conditions[:,2]
original_u0_3 = pertubed_initial_conditions[:,3]


initial_time_training = 0.0f0
end_time_training = 2.0f0
times = range(initial_time_training, end_time_training, step=0.02)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    σ, r, b = p
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_1, (initial_time_training, end_time_training), original_parameters)
sol_1 = solve(prob, Tsit5(), u0=original_u0_1, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_2 = solve(prob, Tsit5(), u0=original_u0_2, saveat=times, abstol=abs_tol, reltol=rel_tol)
sol_3 = solve(prob, Tsit5(), u0=original_u0_3, saveat=times, abstol=abs_tol, reltol=rel_tol)

sol_as_array_1 = Array(sol_1)
sol_as_array_2 = Array(sol_2)
sol_as_array_3 = Array(sol_3)

Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:green)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color =:blue)
Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:green)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:green)
#Plots.plot!(plt, sol_as_array_3[1,:], sol_as_array_3[2, :], sol_as_array_3[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_1[1,:]], [sol_as_array_1[2, :]], [sol_as_array_1[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", legend=false, color=:green)
#Plots.plot!(plt, sol_as_array_1[1,:], sol_as_array_1[2, :], sol_as_array_1[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_2[1,:]], [sol_as_array_2[2, :]], [sol_as_array_2[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:green)
#Plots.plot!(plt, sol_as_array_2[1,:], sol_as_array_2[2, :], sol_as_array_2[3, :], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3")
Plots.scatter!(plt, [sol_as_array_3[1,:]], [sol_as_array_3[2, :]], [sol_as_array_3[3, :]], label="Lorenz dynamics", xlabel="y1", ylabel="y2", zlabel="y3", color=:green)

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.0
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))

#save the data
serialize("lorenz_in_silico_data_no_noise.jld", df)

# add a gaussian noise to the data
σ = 0.01
#max_oscillations = [maximum(sol[i,30:end]) - minimum(sol[i,30:end]) for i in 1:size(sol, 1)]
max_oscillations = [maximum(sol_as_array_1[i,1:end]) - minimum(sol_as_array_1[i,1:end]) for i in 1:size(sol_as_array_1, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_1, 2))
noise_std = σ * max_oscillations
sol_as_array_1_noisy = sol_as_array_1 .+ noise_std .* randn(size(sol_as_array_1))
max_oscillations = [maximum(sol_as_array_2[i,1:end]) - minimum(sol_as_array_2[i,1:end]) for i in 1:size(sol_as_array_2, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_2, 2))
noise_std = σ * max_oscillations
sol_as_array_2_noisy = sol_as_array_2 .+ noise_std .* randn(size(sol_as_array_2))
max_oscillations = [maximum(sol_as_array_3[i,1:end]) - minimum(sol_as_array_3[i,1:end]) for i in 1:size(sol_as_array_3, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array_3, 2))
noise_std = σ * max_oscillations
sol_as_array_3_noisy = sol_as_array_3 .+ noise_std .* randn(size(sol_as_array_3))

df = DataFrame(t = times, x1 = sol_as_array_1_noisy[1,:], x2 = sol_as_array_1_noisy[2,:], x3 = sol_as_array_1_noisy[3,:], traj = repeat([1], length(times)))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_2_noisy[1,:], x2 = sol_as_array_2_noisy[2,:], x3 = sol_as_array_2_noisy[3,:], traj = repeat([2], length(times))))
df = vcat(df, DataFrame(t = times, x1 = sol_as_array_3_noisy[1,:], x2 = sol_as_array_3_noisy[2,:], x3 = sol_as_array_3_noisy[3,:], traj = repeat([3], length(times))))
#save the data
serialize("lorenz_in_silico_data_noisy.jld", df)

# add a gaussian noise to the initial conditions to generate new trajectories to perturb the data
σ = 0.3
noise_std = σ * original_u0_1
pertubed_initial_conditions = original_u0_1 .+ noise_std .* randn(length(original_u0_1), 5000)
pertubed_initial_conditions = pertubed_initial_conditions[:, 1:100]
#select only the trajectories with positive initial state 
# generate the trajectories with the perturbed initial conditiions
pure_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
noisy_df = DataFrame(t = [], x1 = [], x2 = [], x3 = [], traj = [])
for i in 1:size(pertubed_initial_conditions, 2)
    prob = ODEProblem(ground_truth_function, pertubed_initial_conditions[:, i], (initial_time_training, end_time_training), original_parameters)
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=abs_tol, reltol=rel_tol)
    # add a gaussian noise to the data
    σ = 0.0
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:],  x3 = sol_noisy[3,:],  traj = repeat([i+1], length(sol.t)))
    #save the data
    pure_df = vcat(pure_df, df)

    σ = 0.05
    max_oscillations = [maximum(sol[i,1:end]) - minimum(sol[i,1:end]) for i in 1:size(sol, 1)]
    max_oscillations = repeat(max_oscillations, 1, size(sol, 2))
    noise_std = σ * max_oscillations
    sol_noisy = Array(sol) .+ noise_std .* randn(size(sol))
    
    #save in a dataframe the noisy simulation
    df = DataFrame(t = sol.t, x1 = sol_noisy[1,:], x2 = sol_noisy[2,:], x3 = sol_noisy[3,:], traj = repeat([i+1], length(sol.t)))
    #save the data
    noisy_df = vcat(noisy_df, df)
end

serialize("lorenz_trajectory_with_perturbed_iv_no_noise.jld", pure_df)
serialize("lorenz_trajectory_with_perturbed_iv_noisy.jld", noisy_df)
 =#

savefig(plt, "lorenz_in_silico_data.png")
