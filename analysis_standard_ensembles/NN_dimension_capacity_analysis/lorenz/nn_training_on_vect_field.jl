# This script trains a neural network to approximate the Lorenz vector field. The neural network is trained on a dataset of 50000 points sampled uniformly from the bounding box of the vector field. The dataset is split into training, validation and test sets. The neural network is trained using the Adam optimizer for 1000 epochs. The best model on the validation set is then refined using L-BFGS. The script outputs the test error of the refined model. The trained model is serialized to a file.

cd(@__DIR__)

using Lux, ADTypes, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie, Serialization, ComponentArrays, DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL

rng = MersenneTwister()
Random.seed!(rng, 0)

#gets the bounding box of the vector field where I want to approximate the vector field
boundig_box_vect_field = deserialize("bounding_box_to_train.jld")

width= boundig_box_vect_field[2] - boundig_box_vect_field[1]
height = boundig_box_vect_field[4] - boundig_box_vect_field[3]
depth = boundig_box_vect_field[6] - boundig_box_vect_field[5]

# enlarge the bounding box by 10% in each direction, because empirical evidence shows that the vector field is not well approximated at the boundary
boundig_box_vect_field_new = [boundig_box_vect_field[1] - 0.1 * width,
 boundig_box_vect_field[2] + 0.1 * width,
 boundig_box_vect_field[3] - 0.1 * height,
 boundig_box_vect_field[4] + 0.1 * height,
 boundig_box_vect_field[5] - 0.1 * depth,
 boundig_box_vect_field[6] + 0.1 * depth]

 boundig_box_vect_field = boundig_box_vect_field_new

width= boundig_box_vect_field[2] - boundig_box_vect_field[1]
height = boundig_box_vect_field[4] - boundig_box_vect_field[3]
depth = boundig_box_vect_field[6] - boundig_box_vect_field[5]

# Define the Lorenz vector field
original_parameters = Float64[10, 28, 8/3]
function lorenz_ground_truth(u)
    σ, r, b = original_parameters
    du = similar(u)
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
    return du
end

# Extract uniformly 5000 points from the bounding box
n_points = 50000
function sample_uniform_3d(a, b, c, d, e, f)
    x = rand() * (b - a) + a  # Sample x-coordinate
    y = rand() * (d - c) + c  # Sample y-coordinate
    z = rand() * (f - e) + e  # Sample z-coordinate
    return [x, y, z]
end
min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field
points = [sample_uniform_3d(min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3) for i in 1:n_points]
ground_truth_vect_field = [lorenz_ground_truth([p[1], p[2], p[3]]) for p in points]

#split the data in training, validation and test set
training_points = points[1:40000]
validation_points = points[40001:45000]
test_points = points[45001:end]
training_ground_truth = ground_truth_vect_field[1:40000]
validation_ground_truth = ground_truth_vect_field[40001:45000]
test_ground_truth = ground_truth_vect_field[45001:end]

#reshape training points
training_points = hcat(training_points...)
validation_points = hcat(validation_points...)
test_points = hcat(test_points...)

#normalize the points over the dimensions of the bounding box 
#to train the neural network
training_points = training_points ./ [width, height, depth]
validation_points = validation_points ./ [width, height, depth]
test_points = test_points ./ [width, height, depth]

#reshape the ground truths
training_ground_truth = hcat(training_ground_truth...)
validation_ground_truth = hcat(validation_ground_truth...)
test_ground_truth = hcat(test_ground_truth...)

#instantiates the training, validation and test data
training_data = (training_points, training_ground_truth)
validation_data = (validation_points, validation_ground_truth)
test_data = (test_points, test_ground_truth)

# define the neural network
neural_network_dimension = 32
model = Lux.Chain(
            Lux.Dense(3, neural_network_dimension, gelu),
            Lux.Dense(neural_network_dimension, neural_network_dimension, gelu),
            Lux.Dense(neural_network_dimension, 3),
        )

# instructions for training
opt = Optimisers.Adam(0.0005f0)
const loss_function = MSELoss()
const dev_cpu = cpu_device()
ps, st = Lux.setup(rng, model)
tstate = Training.TrainState(model, ps, st, opt)
vjp_rule = AutoZygote()

# Define the training loop
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

        # Early stopping
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
res = Optimization.solve(optprob, Optim.LBFGS(); callback=(θ, l) -> callback_lbfgs(θ, l, validation_losses, best_on_validations), maxiters=40000)

# get the best solution over validation
resulting_parameters = best_on_validations[1]


# Evaluate the refined model on test data
y_pred = Lux.apply(tstate.model, test_data[1], resulting_parameters, tstate.states)[1]
test_error = mean(abs2.(y_pred .- test_data[2]), dims=1)
@printf "Test Error after LBFGS fine-tuning: %.5g\n" mean(test_error)

#serialize the trained model
tstate = Training.TrainState(tstate.model, resulting_parameters, tstate.states, tstate.optimizer)
serialize("trained_model.jld", tstate)