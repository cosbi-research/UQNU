# configuration file for the integration and the neural network

# numerical integrator
integrator = Vern7()
abstol = 1e-6
reltol = 1e-5
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

get_uode_model_function = function (approximating_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
        let approximating_neural_network = approximating_neural_network, st = state
            û_2 = approximating_neural_network(u, p, st)[1]
            @inbounds du[1] = û_2[1]
            @inbounds du[2] = û_2[2]
        end
end

get_vector_field_function = function (approximating_neural_network, state)
    #generates the function with the parameters
    f(u, p) =
        let approximating_neural_network = approximating_neural_network, st = state
            û_2 = approximating_neural_network(u, p, st)[1]
            return û_2
        end
end

########################################## NEURAL NETWORK STRUCTUREs ######################################################
########################################## NODE neural network ############################################################
in_dim = 2
out_dim = 2

neural_network_dimension = 32
activation_function = 4
activation_function_fun = [tanh, relu, sigmoid, gelu][activation_function]
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=gain)

# different neural networks because I want to train only the last layer of the network

#first block
approximating_neural_network = Lux.Chain(
  Lux.Dense(in_dim, neural_network_dimension, activation_function_fun),
  Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun),
  Lux.Dense(neural_network_dimension, out_dim)
)

######################################### GROUND TRUTH FUNCTION ############################################################
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
#function to generate the Data
function lotka_volterra_gound_truth(u)
    α, β, γ, δ = original_parameters
    du = similar(u)
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
    return du
end