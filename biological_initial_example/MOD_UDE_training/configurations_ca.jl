# configuration file for the integration and the neural network

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

function get_uode_model_function(appr_neural_network, state, original_parameters)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters = original_parameters
             
            u = max.(min.(u, 10^5), 0.0) # to avoid negative concentrations

            ode_par = p.ode_par.* original_parameters

            û = appr_neural_network(view(u, [1, 4, 5, 6, 7, 8]), p.p_net, st)[1]# Network prediction
            @inbounds du[1] = -ode_par[1]*u[4]*u[1] + ode_par[2]*u[5]
            @inbounds du[2] = ode_par[3]*u[5] - ode_par[4]*u[2]*u[3] + ode_par[5]*u[6] + ode_par[6]*u[6]
            @inbounds du[3] = -ode_par[4]*u[2]*u[3] + ode_par[5]*u[6]
            @inbounds du[4] = û[1]
            @inbounds du[5] = -ode_par[3]*u[5] + ode_par[1]*u[4]*u[1] - ode_par[2]*u[5]
            @inbounds du[6] = -ode_par[6]*u[6] + ode_par[4]*u[2]*u[3] - ode_par[5]*u[6]
            @inbounds du[7] = max(min(-ode_par[7]*u[7]*u[4] + ode_par[8]*u[8] + ode_par[9]*u[8], 10^5), -10^5)
            @inbounds du[8] = max(min(ode_par[7]*u[7]*u[4] - ode_par[8]*u[8] - ode_par[9]*u[8], 10^5), -10^5)
        end
end

get_vector_field_function = function (appr_neural_network, state, original_parameters)
    #generates the function with the parameters
    f(u, p) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters = original_parameters
            #@infiltrate
            ode_par = p.ode_par.* original_parameters

            û = appr_neural_network(view(u, [1, 4, 5, 6, 7, 8]), p.p_net, st)[1]# Network prediction
            du_1 = -ode_par[1]*u[4]*u[1] + ode_par[2]*u[5]
            du_2 = ode_par[3]*u[5] - ode_par[4]*u[2]*u[3] + ode_par[5]*u[6] + ode_par[6]*u[6]
            du_3 = -ode_par[4]*u[2]*u[3] + ode_par[5]*u[6]
            du_4 = û[1]
            du_5 = -ode_par[3]*u[5] + ode_par[1]*u[4]*u[1] - ode_par[2]*u[5]
            du_6 = -ode_par[6]*u[6] + ode_par[4]*u[2]*u[3] - ode_par[5]*u[6]
            du_7 = max(min(-ode_par[7]*u[7]*u[4] + ode_par[8]*u[8] + ode_par[9]*u[8], 10^5), -10^5)
            du_8 = max(min(ode_par[7]*u[7]*u[4] - ode_par[8]*u[8] - ode_par[9]*u[8], 10^5), -10^5)

            return [du_1'; du_2'; du_3'; du_4'; du_5'; du_6'; du_7'; du_8']
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