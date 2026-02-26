cd(@__DIR__)

using Serialization, Random, DataFrames, StableRNGs, Plots

rng = Random.default_rng()

error_levels = [0, 1]

Random.seed!(rng, 0)

#load the data and split into training and validation
datafile = "cell_apoptosis_silico_data.jld"

solutions_dataframe = deserialize(datafile)

size_df = size(solutions_dataframe)[1]
size_validation = round(Int, 0.2 * size_df)

#generates the random mask for the training and the valudation data set
mask = shuffle(2:size_df)
validation_mask = mask[1:size_validation]
training_mask = pushfirst!(mask[size_validation+1:end], 1)

training_dataframe = solutions_dataframe[training_mask, :]
validation_dataframe = solutions_dataframe[validation_mask, :]


max_oscillations = [maximum(training_dataframe[1:end, i]) - minimum(training_dataframe[1:end, i]) for i in 2:(size(training_dataframe, 2))]

solution_dataframes = [solutions_dataframe,]
training_dataframes = [training_dataframe,]
validation_dataframes = [validation_dataframe,]

data_structure = (
    solution_dataframes=solution_dataframes,
    training_dataframes=training_dataframes,
    validation_dataframes=validation_dataframes,
    max_oscillations=max_oscillations
)

serialize("cell_apoptosis_training_data_structure.jld", data_structure)
