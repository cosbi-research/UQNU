# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
original_u0 = [10, 10, 10]

initial_time_training = 0.0f0
end_time_training = 30.0f0
times = range(initial_time_training, end_time_training, length=100)

#function to generate the Data
function ground_truth_function(du, u, p, t)
    σ, r, b = p
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end