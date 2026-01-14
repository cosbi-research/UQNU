# configuration file for the integration and the neural network

# numerical integrator
integrator = Vern7()
abstol = 1e-6
reltol = 1e-5
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

get_uode_model_function = function (appr_neural_network, state, original_parameters)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters = original_parameters
            #@infiltrate
            û = appr_neural_network(u, p.p_net, st)[1]
            @inbounds du[1] = (p.α * original_parameters[1]) * u[1] + û[1]
            @inbounds du[2] = û[2] - (p.δ * original_parameters[2]) * u[2]
        end
end

get_uode_fixed_model_function = function (appr_neural_network, state, original_parameters)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters = original_parameters
            #@infiltrate
            û = appr_neural_network(u, p.p_net, st)[1]
            @inbounds du[1] = (original_parameters[1]) * u[1] + û[1]
            @inbounds du[2] = û[2] - (original_parameters[2]) * u[2]
        end
end


get_vector_field_function = function (appr_neural_network, state, original_parameters)
    #generates the function with the parameters
    f(u, p) =
        let approximating_neural_network = appr_neural_network, st = state, original_parameters = original_parameters
            #@infiltrate
            û = approximating_neural_network(u, p.p_net, st)[1]
            du_1 = (p.α * original_parameters[1]) * u[1,:] .+ û[1,:]
            du_2 = û[2,:] .- (p.δ * original_parameters[2]) * u[2,:]
            return [du_1'; du_2']
        end
end

########################################## NEURAL NETWORK STRUCTUREs ######################################################
########################################## NODE neural network ############################################################
in_dim = 6
out_dim = 1

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
original_parameters = Float64[2.67 * 10^-9 *3600 * 10^5, 1*10^-2*3600, 8* 10^-3*3600, 6.8 * 10^-8*3600 * 10^5, 5*10^-2*3600, 1*10^-3*3600, 7*10^-5*3600 * 10^5, 1.67 * 10^-5*3600, 1.67*10^-4*3600]

function ca_gound_truth(u)
    p = original_parameters
    du = similar(u)
    du[1] = -p[1]*u[4]*u[1] + p[2]*u[5]
    du[2] = p[3]*u[5] - p[4]*u[2]*u[3] + p[5]*u[6] + p[6]*u[6]
    du[3] = -p[4]*u[2]*u[3] + p[5]*u[6]
    du[4] = p[6]*u[6] - p[1]*u[4]*u[1] +p[2]*u[5] - p[7]*u[4]*u[7] + p[8]*u[8] + p[3]*u[5]
    du[5] = -p[3]*u[5] + p[1]*u[4]*u[1] - p[2]*u[5]
    du[6] = -p[6]*u[6] + p[4]*u[2]*u[3] - p[5]*u[6]
    du[7] = -p[7]*u[7]*u[4] + p[8]*u[8] + p[9]*u[8]
    du[8] = p[7]*u[7]*u[4] - p[8]*u[8] - p[9]*u[8]
end