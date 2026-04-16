cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Measures   

rng = Random.default_rng()
Random.seed!(rng, 10)

# parameters for Lotka Volterra and initial state
original_parameters = Float64[2.67 * 10^-9 *3600 * 10^5, 1*10^-2*3600, 8* 10^-3*3600, 6.8 * 10^-8*3600 * 10^5, 5*10^-2*3600, 1*10^-3*3600, 7*10^-5*3600 * 10^5, 1.67 * 10^-5*3600, 1.67*10^-4*3600]
original_u0_survival = [1.34 * 10^5 / 10^5, 1.0*10^5 / 10^5, 2.67*10^5 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 2.9*10^3 / 10^5, 0.0 / 10^5]
original_u0_death = [1.34 * 10^5 / 10^5, 1.0*10^5 / 10^5, 2.67*10^5 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 2.9*10^4 / 10^5, 0.0 / 10^5]


initial_time_training = 0.0f0
end_time_training = 16.0f0
times = range(initial_time_training, end_time_training, length=120)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    du[1] = -p[1]*u[4]*u[1] + p[2]*u[5]
    du[2] = p[3]*u[5] - p[4]*u[2]*u[3] + p[5]*u[6] + p[6]*u[6]
    du[3] = -p[4]*u[2]*u[3] + p[5]*u[6]
    du[4] = p[6]*u[6] - p[1]*u[4]*u[1] +p[2]*u[5] - p[7]*u[4]*u[7] + p[8]*u[8] + p[3]*u[5]
    du[5] = -p[3]*u[5] + p[1]*u[4]*u[1] - p[2]*u[5]
    du[6] = -p[6]*u[6] + p[4]*u[2]*u[3] - p[5]*u[6]
    du[7] = -p[7]*u[7]*u[4] + p[8]*u[8] + p[9]*u[8]
    du[8] = p[7]*u[7]*u[4] - p[8]*u[8] - p[9]*u[8]
end

integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6

#generate the data
prob = ODEProblem(ground_truth_function, original_u0_survival, (initial_time_training, end_time_training), original_parameters)
sol_death = solve(prob, integrator, u0=original_u0_death, saveat=times, reltol=reltol, abstol=abstol)
sol_death_continous = solve(prob, integrator, u0=original_u0_death, saveat=range(initial_time_training, end_time_training, length=1000), reltol=reltol, abstol=abstol)

end_time_training = 16.0f0
times_new = range(initial_time_training, end_time_training, length=120)
prob = ODEProblem(ground_truth_function, original_u0_survival, (initial_time_training, end_time_training), original_parameters)
sol_survival = solve(prob, integrator, u0=original_u0_survival, saveat=range(initial_time_training, end_time_training, length=1000), reltol=reltol, abstol=abstol)

#training dataset over the cell death
sol_as_array = Array(sol_death)

rng = Random.default_rng()
Random.seed!(rng, 0)

# add a gaussian noise to the data
σ = 0.05
max_oscillations = [maximum(sol_as_array[i,1:end]) - minimum(sol_as_array[i,1:end]) for i in 1:size(sol_as_array, 1)]

max_oscillations = [mean(sol_as_array[i,1:end]) for i in 1:size(sol_as_array, 1)]
max_oscillations = repeat(max_oscillations, 1, size(sol_as_array, 2))
noise_std = σ * max_oscillations
sol_as_array_noisy = sol_as_array .+ noise_std .* randn(size(sol_as_array))
sol_as_array_noisy = max.(sol_as_array_noisy, 0.0) #to avoid negative concentrations

#save in a dataframe the noisy simulation
df = DataFrame(t = times, x1 = sol_as_array_noisy[1,:],
                     x2 = sol_as_array_noisy[2,:],
                     x3 = sol_as_array_noisy[3,:],
                     x4 = sol_as_array_noisy[4,:],
                     x5 = sol_as_array_noisy[5,:],
                     x6 = sol_as_array_noisy[6,:],
                     x7 = sol_as_array_noisy[7,:],
                     x8 = sol_as_array_noisy[8,:]
                     )

#plot the x_4 variable with the noise
plot_x4_death = Plots.scatter(
    df.t,
    df.x4,
    label="Training data",
    xlabel="Time (hours)",
    ylabel="y4 (10⁵ molecules/cell)",
    title="Cell survival",
    color="red",
    legend = nothing
)
Plots.plot!(plot_x4_death, sol_death_continous.t, sol_death_continous[4,:], label="Ground truth", color="blue")
Plots.plot!(plot_x4_death, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)

plot_x4_survival = Plots.plot(
    sol_survival.t,
    sol_survival[4,:],
    label="Ground truth",
    xlabel="Time (hours)",
    ylabel="y4 (10⁵ molecules/cell)",
    title="Cell death",
    color="blue",
    legend = nothing
)
Plots.plot!(plot_x4_survival, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
)

#put the plots in a single plot

#save the two plots as svg
Plots.savefig(plot_x4_death, "training_data_cell_death.png")
Plots.savefig(plot_x4_survival, "training_data_cell_survival.png")
