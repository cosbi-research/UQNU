cd(@__DIR__)

using Serialization, Random, DataFrames, StableRNGs, Plots

rng = Random.default_rng()

error_levels = [0, 1]

for error_level in error_levels
    Random.seed!(rng, 0)

    #load the data and split into training and validation
    datafile = "damped_oscillator_in_silico_data"
    if error_level == 1
    datafile = datafile * "_no_noise.jld"
    else
    datafile = datafile * "_noisy.jld"
    end

    solutions_dataframe = deserialize(datafile)
    trajectories = unique(solutions_dataframe.traj)

    solution_dataframe_1 = solutions_dataframe[solutions_dataframe.traj.==trajectories[1], :]
    size_df = size(solution_dataframe_1)[1]
    size_validation = round(Int, 0.2 * size_df)

    #generates the random mask for the training and the valudation data set
    mask = shuffle(2:size_df)
    validation_mask = mask[1:size_validation]
    training_mask = pushfirst!(mask[size_validation+1:end], 1)

    training_dataframe_1 = solution_dataframe_1[training_mask, :]
    validation_dataframe_1 = solution_dataframe_1[validation_mask, :]

    training_dataframe_1 = sort(training_dataframe_1, [:t])
    validation_dataframe_1 = sort(validation_dataframe_1, [:t])

    solution_dataframe_2 = solutions_dataframe[solutions_dataframe.traj.==trajectories[2], :]
    size_df = size(solution_dataframe_2)[1]
    size_validation = round(Int, 0.2 * size_df)

    training_dataframe_2 = solution_dataframe_2[training_mask, :]
    validation_dataframe_2 = solution_dataframe_2[validation_mask, :]
    training_dataframe_2 = sort(training_dataframe_2, [:t])
    validation_dataframe_2 = sort(validation_dataframe_2, [:t])

    solution_dataframe_3 = solutions_dataframe[solutions_dataframe.traj.==trajectories[3], :]
    size_df = size(solution_dataframe_3)[1]

    size_validation = round(Int, 0.2 * size_df)

    training_dataframe_3 = solution_dataframe_3[training_mask, :]
    validation_dataframe_3 = solution_dataframe_3[validation_mask, :]
    training_dataframe_3 = sort(training_dataframe_3, [:t])
    validation_dataframe_3 = sort(validation_dataframe_3, [:t])

    max_oscillations_1 = [maximum(training_dataframe_1[1:end, i]) - minimum(training_dataframe_1[1:end, i]) for i in 2:(size(training_dataframe_1, 2)-1)]
    max_oscillations_2 = [maximum(training_dataframe_2[1:end, i]) - minimum(training_dataframe_2[1:end, i]) for i in 2:(size(training_dataframe_2, 2)-1)]
    max_oscillations_3 = [maximum(training_dataframe_3[1:end, i]) - minimum(training_dataframe_3[1:end, i]) for i in 2:(size(training_dataframe_3, 2)-1)]
    max_oscillations = [max_oscillations_1, max_oscillations_2, max_oscillations_3]

    solution_dataframes = [solution_dataframe_1, solution_dataframe_2, solution_dataframe_3]
    training_dataframes = [training_dataframe_1, training_dataframe_2, training_dataframe_3]
    validation_dataframes = [validation_dataframe_1, validation_dataframe_2, validation_dataframe_3]

    data_structure = (
        solution_dataframes = solution_dataframes,
        training_dataframes = training_dataframes,
        validation_dataframes = validation_dataframes,
        max_oscillations = max_oscillations
    )

    plts_traj = []
    for traj in 1:3 
        #Plot the data structure
        plt_v1 = Plots.plot(legend=false)
        plt_v2 = Plots.plot(legend=false)

        tmp_training_dataframe = training_dataframes[traj]
        tmp_validation_dataframe = validation_dataframes[traj]

        Plots.scatter!(plt_v1, tmp_training_dataframe[1:end, :t], tmp_training_dataframe[1:end, :x1], label="x1 training", color="blue")
        Plots.scatter!(plt_v1, tmp_validation_dataframe[1:end, :t], tmp_validation_dataframe[1:end, :x1], label="x1 validation", color="red")
        Plots.scatter!(plt_v2, tmp_training_dataframe[1:end, :t], tmp_training_dataframe[1:end, :x2], label="x2 training", color="blue")
        Plots.scatter!(plt_v2, tmp_validation_dataframe[1:end, :t], tmp_validation_dataframe[1:end, :x2], label="x2 validation", color="red")
        #plot a vertical line for the max oscillation

        # Draw a rectangle with min y min(training_dataframe_1.x1) and height max_oscillations[1] along the whole plot
        min_y = minimum(tmp_training_dataframe.x1)
        height = max_oscillations[traj][1]
        x_min = minimum(tmp_training_dataframe.t)
        x_max = maximum(tmp_training_dataframe.t)
        Plots.plot!(plt_v1, [x_min, x_max, x_max, x_min, x_min], [min_y, min_y, min_y + height, min_y + height, min_y], fillalpha=0.2, fillcolor=:green, label="")
        # do the same for the second variable
        min_y = minimum(tmp_training_dataframe.x2)
        height = max_oscillations[traj][2]
        Plots.plot!(plt_v2, [x_min, x_max, x_max, x_min, x_min], [min_y, min_y, min_y + height, min_y + height, min_y], fillalpha=0.2, fillcolor=:green, label="")

        #put the plots together
        plt = Plots.plot(plt_v1, plt_v2, layout=(1, 2)) 
        push!(plts_traj, plt)
    end

    plt_data_structure = Plots.plot(plts_traj..., layout=(3, 1))
    Plots.savefig("damped_oscillator_training_data_structure_err_"*string(error_level)*".png")

    #save as png
    Plots.savefig("damped_oscillator_training_data_structure_err_"*string(error_level)*".png")

    serialize("damped_oscillator_training_data_structure_err_"*string(error_level)*".jld", data_structure)
end