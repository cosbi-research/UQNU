# This script trains a neural network to approximate the Lotka Volterra vector field. The neural network is trained on a dataset of 50000 points sampled uniformly from the bounding box of the vector field. The dataset is split into training, validation and test sets. The neural network is trained using the Adam optimizer for 1000 epochs. The best model on the validation set is then refined using L-BFGS. The script outputs the test error of the refined model. The trained model is serialized to a file.

cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL

rng = MersenneTwister()
Random.seed!(rng, 0)

#gets the bounding box of the vector field where I want to approximate the vector field
boundig_box_vect_field = deserialize("bounding_box_to_train.jld")
min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

# Define the Lotka Volterra vector field
original_parameters = Float64[0.1, 2]#function to generate the Data
function damped_oscillator_ground_truth(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α*u[1]^3 - β*u[2]^3
    du[2] = β*u[1]^3  - α*u[2]^3
    return du
end

# Extract uniformly 10000 points from the bounding box
n_points = 10000
function sample_uniform_2d(a, b, c, d)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    return [x, y]
end
points = [sample_uniform_2d(min_y1, max_y_1, min_y2, max_y2) for i in 1:n_points]
# computes the ground truth vector field at each point
ground_truth_vect_field = [damped_oscillator_ground_truth([p[1], p[2]]) for p in points]

#split the data in training, validation and test set
training_points = points[1:8000]
validation_points = points[8001:9000]
test_points = points[9001:end]
training_ground_truth = ground_truth_vect_field[1:8000]
validation_ground_truth = ground_truth_vect_field[8001:9000]
test_ground_truth = ground_truth_vect_field[9001:end]

#reshape points
training_points = hcat(training_points...)
validation_points = hcat(validation_points...)
test_points = hcat(test_points...)

#reshape ground truth
training_ground_truth = hcat(training_ground_truth...)
validation_ground_truth = hcat(validation_ground_truth...)
test_ground_truth = hcat(test_ground_truth...)

# builds the dataset
training_data = (training_points, training_ground_truth)
validation_data = (validation_points, validation_ground_truth)
test_data = (test_points, test_ground_truth)

# Define the neural network
neural_network_dimension = 32
model = Lux.Chain(
            Lux.Dense(2, neural_network_dimension, gelu),
            Lux.Dense(neural_network_dimension, neural_network_dimension, gelu),
            Lux.Dense(neural_network_dimension, 2),
        )

# train the neural network
opt = Optimisers.Adam(0.0005f0)
const loss_function = MSELoss()
const dev_cpu = cpu_device()
ps, st = Lux.setup(rng, model)
tstate = Training.TrainState(model, ps, st, opt)
vjp_rule = AutoZygote()
function main(tstate::Training.TrainState, vjp, data, epochs)
    validation_losses = []
    best_on_validation = nothing
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)

        #validation cost
        validation_predictions = Lux.apply(tstate.model, validation_data[1], tstate.parameters, tstate.states)[1]
        validation_loss = loss_function(validation_data[2], validation_predictions)

        push!(validation_losses, validation_loss)
        if validation_loss == minimum(validation_losses)
            best_on_validation = deepcopy(tstate)
        end

        if epoch > 10000 && minimum(validation_losses[1:end]) == minimum(validation_losses[1:end-300])
            @printf "Early stopping at epoch %d\n" epoch
            break
        end

        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g Validation Loss: %.5g\n" epoch loss validation_loss
        end
    end
    return best_on_validation
end
tstate = main(tstate, vjp_rule, training_data, 1000)

# Refine the model with L-BFGS
function loss_for_lbfgs(params)
    predictions, _ = Lux.apply(tstate.model, training_data[1], params, tstate.states)
    loss = mean(abs2.(training_data[2] .- predictions))
    return loss
end

# callback function for L-BFGS
function callback_lbfgs(state, loss, validation_losses, best_on_validations)

    validation_predictions, _ = Lux.apply(tstate.model, validation_data[1], state.u, tstate.states)
    validation_loss = mean(abs2.(validation_data[2] .- validation_predictions))

    # print on the same line
    if length(validation_losses) % 50 == 0
        @printf "Epoch: %3d \t Loss: %.5g \t Validation Loss: %.5g\n" length(validation_losses) loss validation_loss
    end

    push!(validation_losses, validation_loss)
    # Early stopping
    if length(validation_losses) > 2000
        if minimum(validation_losses[1:end-2000]) == minimum(validation_losses)
            println("Early stopping at epoch ", length(validation_losses))
            return true
        end
    end

    if minimum(validation_losses) == validation_loss
        best_on_validations[1] = deepcopy(state.u)
    end

    return false
end

# instruction for training with L-BFGS
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_for_lbfgs(x), Optimization.AutoZygote())
params_vec = ComponentArray(tstate.parameters)
optprob = Optimization.OptimizationProblem(optf, params_vec)
validation_losses = []
best_on_validations = [params_vec]
res = Optimization.solve(optprob, Optim.LBFGS(); callback=(θ, l) -> callback_lbfgs(θ, l, validation_losses, best_on_validations), maxiters=10000)

# get the best solution over validation
resulting_parameters = best_on_validations[1]


# Evaluate the refined model on test data
y_pred = Lux.apply(tstate.model, test_data[1], resulting_parameters, tstate.states)[1]
test_error = mean(abs2.(y_pred .- test_data[2]), dims=1)
@printf "Test Error after LBFGS fine-tuning: %.5g\n" mean(test_error)

#serialize the trained model
tstate = Training.TrainState(tstate.model, resulting_parameters, tstate.states, tstate.optimizer)
serialize("trained_model.jld", tstate)