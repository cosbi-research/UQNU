# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
original_u0 = [3.1461493970111687, 1.5370475785612603]

initial_time_training = 0.0f0
end_time_training = 5.0f0
times = range(initial_time_training, end_time_training, length=100)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end