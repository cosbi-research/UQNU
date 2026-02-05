cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

rng = Random.default_rng()
Random.seed!(rng, 10)

#load the dataset 
training_dataset = deserialize("cell_apoptosis_silico_data.jld")

#for each variable, find the min and the max over the dataset
num_variables = size(training_dataset, 2) - 1 #exclude time and traj columns
min_values = zeros(num_variables)
max_values = zeros(num_variables)
for i in 1:num_variables
    min_values[i] = minimum(training_dataset[!, i+1]) #+1 to exclude time column
    max_values[i] = maximum(training_dataset[!, i+1])
end

#create the bounding box as a DataFrame
bounding_box = DataFrame(variable = String[], min_value = Float64[], max_value = Float64[])
for i in 1:num_variables
    push!(bounding_box, (string("u", i), min_values[i], max_values[i]))
end

#extend the bounding box by 50% on each side
for i in 1:num_variables
    range = max_values[i] - min_values[i]
    min_values[i] -= 0.0 * range
    max_values[i] += 0.0 * range
    bounding_box[i, :min_value] = min_values[i]
    bounding_box[i, :max_value] = max_values[i]
end


#save the bounding box
serialize("cell_apoptosis_silico_data_bounding_box.jld", bounding_box)