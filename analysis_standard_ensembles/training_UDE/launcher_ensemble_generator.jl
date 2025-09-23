cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates

using PyCall
optuna = pyimport("optuna")

template_content = file_content = read("templates/launcher_sbatch.sh", String)

models = [1,2,3]
model_names = ["lv", "lorenz", "damped"]
#gains = [0.01, 0.1, 1.0, 10.0]
gains = [1.0]

launcher_files = []
for model in models
  for gain in gains

    hyp_filename="hyper_param_tuning/"
    if model == 1
      hyp_filename = hyp_filename*"/hyperparameters_LV_UDE.jld"
    elseif model == 2
      hyp_filename = hyp_filename*"/hyperparameters_LORENZ_UDE.jld"
    else
      hyp_filename = hyp_filename*"/hyperparameters_DAMPED_UDE.jld"
    end

    hyperparameter_study = deserialize(hyp_filename)

    script_arguments =  collect(hyperparameter_study)
    script_arguments[9] = "200"
    script_arguments[10] = "20"

    #collapse them into a string
    argument_as_string = join(script_arguments, " ")

    #replace the template content with the actual content
    file_content = replace(template_content, "##SCRIPT_ARGUMENTS##" => argument_as_string)
    file_content = replace(file_content, "##MODEL##" => model_names[model])
    job_id=string(model)*"_"*string(gain)
    job_id = replace(job_id, "." => "_")
    job_id = "ensemble_gain_"*job_id
    file_content = replace(file_content, "##JOB_NAME##" => job_id)

    #write a file "launcher_ensemble_gain_ model gain.sh"
    open("launcher_ensemble_"*model_names[model]*".sh", "w") do io
        print(io, file_content)
    end
    
    push!(launcher_files, "sbatch launcher_ensemble_"*model_names[model]*".sh")
  end
end

 #write the launcher_gain files with all the sbatch commands
open("launcher_ensemble.sh", "w") do io
    for launcher_file in launcher_files
        println(io, launcher_file)
    end
end