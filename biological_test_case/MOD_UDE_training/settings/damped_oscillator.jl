# parameters for Lotka Volterra and initial state
original_parameters = Float64[0.1, 2]
original_u0 = [1, 1]

initial_time_training = 0.0f0
end_time_training = 25.0f0
times = range(initial_time_training, end_time_training, length=100)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    α, β = p
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
end