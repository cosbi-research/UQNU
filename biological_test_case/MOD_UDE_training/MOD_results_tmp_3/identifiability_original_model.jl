#=
This script is used to determine the identifiability of the parameters of the original cell apoptosis model
assuming different levels of observability.
=#

cd(@__DIR__)

using ComponentArrays, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using StableRNGs, Zygote, LinearAlgebra, SciMLSensitivity, Optimization

#includes the specific model functions
#load the data and split into training and validation
datafile = "../../data_generator/cell_apoptosis_silico_data.jld"
training_data_structure = deserialize("../../data_generator/cell_apoptosis_training_data_structure.jld")


include("../../cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../../cell_apoptosis_settings/cell_apop_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-8
reltol = 1e-7

sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = training_data_structure.solution_dataframes[1].t

max_oscillations = [maximum(training_data_structure.solution_dataframes[1][:, i]) - minimum(training_data_structure.solution_dataframes[1][:, i]) for i in 2:size(training_data_structure.solution_dataframes[1], 2)]

#get the parameters estimated
tspan = (initial_time_training, end_time_training)

adtype = Optimization.AutoZygote()
par = original_parameters
prob_uode_pred = ODEProblem{true}(ground_truth_function, original_u0, tspan)

###########################################################################################################################
function model(params, final_time, observables)
  prob_uode_pred = ODEProblem{true}(ground_truth_function, original_u0, (0, final_time))
  solutions = solve(prob_uode_pred, integrator, p=params, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
  return Array(solutions)[observables, end]
end

# get the spectral decomposition of the Hessian of the cost function with respect to the parameters
function get_Fisher_Information_Matrix_Eigen_Decomposition(parameters_to_consider, observables)
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2], observables), parameters_to_consider)[1]
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2], observables), parameters_to_consider)[1]
  sensitivity_y3 = Zygote.jacobian(p -> model(p, tsteps[3], observables), parameters_to_consider)[1]
  sensitivity_y4 = Zygote.jacobian(p -> model(p, tsteps[4], observables), parameters_to_consider)[1]
  sensitivity_y5 = Zygote.jacobian(p -> model(p, tsteps[5], observables), parameters_to_consider)[1]
  sensitivity_y6 = Zygote.jacobian(p -> model(p, tsteps[6], observables), parameters_to_consider)[1]
  sensitivity_y7 = Zygote.jacobian(p -> model(p, tsteps[7], observables), parameters_to_consider)[1]
  sensitivity_y8 = Zygote.jacobian(p -> model(p, tsteps[8], observables), parameters_to_consider)[1]
  sensitivity_y9 = Zygote.jacobian(p -> model(p, tsteps[9], observables), parameters_to_consider)[1]
  sensitivity_y10 = Zygote.jacobian(p -> model(p, tsteps[10], observables), parameters_to_consider)[1]
  sensitivity_y11 = Zygote.jacobian(p -> model(p, tsteps[11], observables), parameters_to_consider)[1]
  sensitivity_y12 = Zygote.jacobian(p -> model(p, tsteps[12], observables), parameters_to_consider)[1]
  sensitivity_y13 = Zygote.jacobian(p -> model(p, tsteps[13], observables), parameters_to_consider)[1]
  sensitivity_y14 = Zygote.jacobian(p -> model(p, tsteps[14], observables), parameters_to_consider)[1]
  sensitivity_y15 = Zygote.jacobian(p -> model(p, tsteps[15], observables), parameters_to_consider)[1]
  sensitivity_y16 = Zygote.jacobian(p -> model(p, tsteps[16], observables), parameters_to_consider)[1]
  sensitivity_y17 = Zygote.jacobian(p -> model(p, tsteps[17], observables), parameters_to_consider)[1]
  sensitivity_y18 = Zygote.jacobian(p -> model(p, tsteps[18], observables), parameters_to_consider)[1]
  sensitivity_y19 = Zygote.jacobian(p -> model(p, tsteps[19], observables), parameters_to_consider)[1]
  sensitivity_y20 = Zygote.jacobian(p -> model(p, tsteps[20], observables), parameters_to_consider)[1]
  sensitivity_y21 = Zygote.jacobian(p -> model(p, tsteps[21], observables), parameters_to_consider)[1]
  sensitivity_y22 = Zygote.jacobian(p -> model(p, tsteps[22], observables), parameters_to_consider)[1]
  sensitivity_y23 = Zygote.jacobian(p -> model(p, tsteps[23], observables), parameters_to_consider)[1]
  sensitivity_y24 = Zygote.jacobian(p -> model(p, tsteps[24], observables), parameters_to_consider)[1]
  sensitivity_y25 = Zygote.jacobian(p -> model(p, tsteps[25], observables), parameters_to_consider)[1]
  sensitivity_y26 = Zygote.jacobian(p -> model(p, tsteps[26], observables), parameters_to_consider)[1]
  sensitivity_y27 = Zygote.jacobian(p -> model(p, tsteps[27], observables), parameters_to_consider)[1]
  sensitivity_y28 = Zygote.jacobian(p -> model(p, tsteps[28], observables), parameters_to_consider)[1]
  sensitivity_y29 = Zygote.jacobian(p -> model(p, tsteps[29], observables), parameters_to_consider)[1]
  sensitivity_y30 = Zygote.jacobian(p -> model(p, tsteps[30], observables), parameters_to_consider)[1]
  sensitivity_y31 = Zygote.jacobian(p -> model(p, tsteps[31], observables), parameters_to_consider)[1]
  sensitivity_y32 = Zygote.jacobian(p -> model(p, tsteps[32], observables), parameters_to_consider)[1]
  sensitivity_y33 = Zygote.jacobian(p -> model(p, tsteps[33], observables), parameters_to_consider)[1]
  sensitivity_y34 = Zygote.jacobian(p -> model(p, tsteps[34], observables), parameters_to_consider)[1]
  sensitivity_y35 = Zygote.jacobian(p -> model(p, tsteps[35], observables), parameters_to_consider)[1]
  sensitivity_y36 = Zygote.jacobian(p -> model(p, tsteps[36],observables), parameters_to_consider)[1]
  sensitivity_y37 = Zygote.jacobian(p -> model(p, tsteps[37], observables), parameters_to_consider)[1]
  sensitivity_y38 = Zygote.jacobian(p -> model(p, tsteps[38], observables), parameters_to_consider)[1]
  sensitivity_y39 = Zygote.jacobian(p -> model(p, tsteps[39], observables), parameters_to_consider)[1]
  sensitivity_y40 = Zygote.jacobian(p -> model(p, tsteps[40], observables), parameters_to_consider)[1]
  sensitivity_y41 = Zygote.jacobian(p -> model(p, tsteps[41], observables), parameters_to_consider)[1]
  sensitivity_y42 = Zygote.jacobian(p -> model(p, tsteps[42], observables), parameters_to_consider)[1]
  #keep till 120
  sensitivity_y43 = Zygote.jacobian(p -> model(p, tsteps[43], observables), parameters_to_consider)[1]
  sensitivity_y44 = Zygote.jacobian(p -> model(p, tsteps[44], observables), parameters_to_consider)[1]
  sensitivity_y45 = Zygote.jacobian(p -> model(p, tsteps[45], observables), parameters_to_consider)[1]
  sensitivity_y46 = Zygote.jacobian(p -> model(p, tsteps[46], observables), parameters_to_consider)[1]
  sensitivity_y47 = Zygote.jacobian(p -> model(p, tsteps[47], observables), parameters_to_consider)[1]
  sensitivity_y48 = Zygote.jacobian(p -> model(p, tsteps[48], observables), parameters_to_consider)[1]
  sensitivity_y49 = Zygote.jacobian(p -> model(p, tsteps[49], observables), parameters_to_consider)[1]
  sensitivity_y50 = Zygote.jacobian(p -> model(p, tsteps[50], observables), parameters_to_consider)[1]

sensitivity_y51  = Zygote.jacobian(p -> model(p, tsteps[51],  observables), parameters_to_consider)[1]
sensitivity_y52  = Zygote.jacobian(p -> model(p, tsteps[52],  observables), parameters_to_consider)[1]
sensitivity_y53  = Zygote.jacobian(p -> model(p, tsteps[53],  observables), parameters_to_consider)[1]
sensitivity_y54  = Zygote.jacobian(p -> model(p, tsteps[54],  observables), parameters_to_consider)[1]
sensitivity_y55  = Zygote.jacobian(p -> model(p, tsteps[55],  observables), parameters_to_consider)[1]
sensitivity_y56  = Zygote.jacobian(p -> model(p, tsteps[56],  observables), parameters_to_consider)[1]
sensitivity_y57  = Zygote.jacobian(p -> model(p, tsteps[57],  observables), parameters_to_consider)[1]
sensitivity_y58  = Zygote.jacobian(p -> model(p, tsteps[58],  observables), parameters_to_consider)[1]
sensitivity_y59  = Zygote.jacobian(p -> model(p, tsteps[59],  observables), parameters_to_consider)[1]
sensitivity_y60  = Zygote.jacobian(p -> model(p, tsteps[60],  observables), parameters_to_consider)[1]

sensitivity_y61  = Zygote.jacobian(p -> model(p, tsteps[61],  observables), parameters_to_consider)[1]
sensitivity_y62  = Zygote.jacobian(p -> model(p, tsteps[62],  observables), parameters_to_consider)[1]
sensitivity_y63  = Zygote.jacobian(p -> model(p, tsteps[63],  observables), parameters_to_consider)[1]
sensitivity_y64  = Zygote.jacobian(p -> model(p, tsteps[64],  observables), parameters_to_consider)[1]
sensitivity_y65  = Zygote.jacobian(p -> model(p, tsteps[65],  observables), parameters_to_consider)[1]
sensitivity_y66  = Zygote.jacobian(p -> model(p, tsteps[66],  observables), parameters_to_consider)[1]
sensitivity_y67  = Zygote.jacobian(p -> model(p, tsteps[67],  observables), parameters_to_consider)[1]
sensitivity_y68  = Zygote.jacobian(p -> model(p, tsteps[68],  observables), parameters_to_consider)[1]
sensitivity_y69  = Zygote.jacobian(p -> model(p, tsteps[69],  observables), parameters_to_consider)[1]
sensitivity_y70  = Zygote.jacobian(p -> model(p, tsteps[70],  observables), parameters_to_consider)[1]

sensitivity_y71  = Zygote.jacobian(p -> model(p, tsteps[71],  observables), parameters_to_consider)[1]
sensitivity_y72  = Zygote.jacobian(p -> model(p, tsteps[72],  observables), parameters_to_consider)[1]
sensitivity_y73  = Zygote.jacobian(p -> model(p, tsteps[73],  observables), parameters_to_consider)[1]
sensitivity_y74  = Zygote.jacobian(p -> model(p, tsteps[74],  observables), parameters_to_consider)[1]
sensitivity_y75  = Zygote.jacobian(p -> model(p, tsteps[75],  observables), parameters_to_consider)[1]
sensitivity_y76  = Zygote.jacobian(p -> model(p, tsteps[76],  observables), parameters_to_consider)[1]
sensitivity_y77  = Zygote.jacobian(p -> model(p, tsteps[77],  observables), parameters_to_consider)[1]
sensitivity_y78  = Zygote.jacobian(p -> model(p, tsteps[78],  observables), parameters_to_consider)[1]
sensitivity_y79  = Zygote.jacobian(p -> model(p, tsteps[79],  observables), parameters_to_consider)[1]
sensitivity_y80  = Zygote.jacobian(p -> model(p, tsteps[80],  observables), parameters_to_consider)[1]

sensitivity_y81  = Zygote.jacobian(p -> model(p, tsteps[81],  observables), parameters_to_consider)[1]
sensitivity_y82  = Zygote.jacobian(p -> model(p, tsteps[82],  observables), parameters_to_consider)[1]
sensitivity_y83  = Zygote.jacobian(p -> model(p, tsteps[83],  observables), parameters_to_consider)[1]
sensitivity_y84  = Zygote.jacobian(p -> model(p, tsteps[84],  observables), parameters_to_consider)[1]
sensitivity_y85  = Zygote.jacobian(p -> model(p, tsteps[85],  observables), parameters_to_consider)[1]
sensitivity_y86  = Zygote.jacobian(p -> model(p, tsteps[86],  observables), parameters_to_consider)[1]
sensitivity_y87  = Zygote.jacobian(p -> model(p, tsteps[87],  observables), parameters_to_consider)[1]
sensitivity_y88  = Zygote.jacobian(p -> model(p, tsteps[88],  observables), parameters_to_consider)[1]
sensitivity_y89  = Zygote.jacobian(p -> model(p, tsteps[89],  observables), parameters_to_consider)[1]
sensitivity_y90  = Zygote.jacobian(p -> model(p, tsteps[90],  observables), parameters_to_consider)[1]

sensitivity_y91  = Zygote.jacobian(p -> model(p, tsteps[91],  observables), parameters_to_consider)[1]
sensitivity_y92  = Zygote.jacobian(p -> model(p, tsteps[92],  observables), parameters_to_consider)[1]
sensitivity_y93  = Zygote.jacobian(p -> model(p, tsteps[93],  observables), parameters_to_consider)[1]
sensitivity_y94  = Zygote.jacobian(p -> model(p, tsteps[94],  observables), parameters_to_consider)[1]
sensitivity_y95  = Zygote.jacobian(p -> model(p, tsteps[95],  observables), parameters_to_consider)[1]
sensitivity_y96  = Zygote.jacobian(p -> model(p, tsteps[96],  observables), parameters_to_consider)[1]
sensitivity_y97  = Zygote.jacobian(p -> model(p, tsteps[97],  observables), parameters_to_consider)[1]
sensitivity_y98  = Zygote.jacobian(p -> model(p, tsteps[98],  observables), parameters_to_consider)[1]
sensitivity_y99  = Zygote.jacobian(p -> model(p, tsteps[99],  observables), parameters_to_consider)[1]
sensitivity_y100 = Zygote.jacobian(p -> model(p, tsteps[100], observables), parameters_to_consider)[1]

sensitivity_y101 = Zygote.jacobian(p -> model(p, tsteps[101], observables), parameters_to_consider)[1]
sensitivity_y102 = Zygote.jacobian(p -> model(p, tsteps[102], observables), parameters_to_consider)[1]
sensitivity_y103 = Zygote.jacobian(p -> model(p, tsteps[103], observables), parameters_to_consider)[1]
sensitivity_y104 = Zygote.jacobian(p -> model(p, tsteps[104], observables), parameters_to_consider)[1]
sensitivity_y105 = Zygote.jacobian(p -> model(p, tsteps[105], observables), parameters_to_consider)[1]
sensitivity_y106 = Zygote.jacobian(p -> model(p, tsteps[106], observables), parameters_to_consider)[1]
sensitivity_y107 = Zygote.jacobian(p -> model(p, tsteps[107], observables), parameters_to_consider)[1]
sensitivity_y108 = Zygote.jacobian(p -> model(p, tsteps[108], observables), parameters_to_consider)[1]
sensitivity_y109 = Zygote.jacobian(p -> model(p, tsteps[109], observables), parameters_to_consider)[1]
sensitivity_y110 = Zygote.jacobian(p -> model(p, tsteps[110], observables), parameters_to_consider)[1]

sensitivity_y111 = Zygote.jacobian(p -> model(p, tsteps[111], observables), parameters_to_consider)[1]
sensitivity_y112 = Zygote.jacobian(p -> model(p, tsteps[112], observables), parameters_to_consider)[1]
sensitivity_y113 = Zygote.jacobian(p -> model(p, tsteps[113], observables), parameters_to_consider)[1]
sensitivity_y114 = Zygote.jacobian(p -> model(p, tsteps[114], observables), parameters_to_consider)[1]
sensitivity_y115 = Zygote.jacobian(p -> model(p, tsteps[115], observables), parameters_to_consider)[1]
sensitivity_y116 = Zygote.jacobian(p -> model(p, tsteps[116], observables), parameters_to_consider)[1]
sensitivity_y117 = Zygote.jacobian(p -> model(p, tsteps[117], observables), parameters_to_consider)[1]
sensitivity_y118 = Zygote.jacobian(p -> model(p, tsteps[118], observables), parameters_to_consider)[1]
sensitivity_y119 = Zygote.jacobian(p -> model(p, tsteps[119], observables), parameters_to_consider)[1]
sensitivity_y120 = Zygote.jacobian(p -> model(p, tsteps[120], observables), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21, sensitivity_y22, sensitivity_y23, sensitivity_y24, sensitivity_y25, sensitivity_y26, sensitivity_y27, sensitivity_y28, sensitivity_y29, sensitivity_y30, sensitivity_y31, sensitivity_y32, sensitivity_y33, sensitivity_y34, sensitivity_y35, sensitivity_y36, sensitivity_y37, sensitivity_y38, sensitivity_y39, sensitivity_y40, sensitivity_y41, sensitivity_y42,
  sensitivity_y43, sensitivity_y44, sensitivity_y45, sensitivity_y46, sensitivity_y47, sensitivity_y48, sensitivity_y49, sensitivity_y50,
  sensitivity_y51, sensitivity_y52, sensitivity_y53, sensitivity_y54, sensitivity_y55, sensitivity_y56, sensitivity_y57, sensitivity_y58, sensitivity_y59, sensitivity_y60,
  sensitivity_y61, sensitivity_y62, sensitivity_y63, sensitivity_y64, sensitivity_y65, sensitivity_y66, sensitivity_y67, sensitivity_y68, sensitivity_y69, sensitivity_y70,
  sensitivity_y71, sensitivity_y72, sensitivity_y73, sensitivity_y74, sensitivity_y75, sensitivity_y76, sensitivity_y77, sensitivity_y78, sensitivity_y79, sensitivity_y80,
  sensitivity_y81, sensitivity_y82, sensitivity_y83, sensitivity_y84, sensitivity_y85, sensitivity_y86, sensitivity_y87, sensitivity_y88, sensitivity_y89, sensitivity_y90,
  sensitivity_y91, sensitivity_y92, sensitivity_y93, sensitivity_y94, sensitivity_y95, sensitivity_y96, sensitivity_y97, sensitivity_y98, sensitivity_y99, sensitivity_y100,
  sensitivity_y101, sensitivity_y102, sensitivity_y103, sensitivity_y104, sensitivity_y105, sensitivity_y106, sensitivity_y107, sensitivity_y108, sensitivity_y109, sensitivity_y110,
  sensitivity_y111, sensitivity_y112, sensitivity_y113, sensitivity_y114, sensitivity_y115, sensitivity_y116, sensitivity_y117, sensitivity_y118, sensitivity_y119, sensitivity_y120)

  normalization_matrix = vec(repeat(max_oscillations[observables],119))
  normalization_matrix = Diagonal(1 ./ (normalization_matrix))
  normalization_matrix = abs2.(normalization_matrix)

  hessian_matrix = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
  hessian_matrix = Symmetric(hessian_matrix)
  eigen_value_decomposition = eigen(hessian_matrix)

  eigen_values = real.(eigen_value_decomposition.values)
  eigen_vectors = real.(eigen_value_decomposition.vectors)'

  eigen_vectors_with_eigen_values = hcat(eigen_vectors, eigen_values)

  return eigen_vectors_with_eigen_values, hessian_matrix
end

  # eigenvectors based identifiability algorithm as in Quaiser, Tom, and Martin Mönnigmann. "Systematic identifiability testing for unambiguous mechanistic modeling–application to JAK-STAT, MAP kinase, and NF-κ B signaling pathway models." BMC systems biology 3 (2009): 1-21.
function get_identifiable_parameters(observables)
  eigen_vectors_normalized_with_eigen_values, hessian_matrix = get_Fisher_Information_Matrix_Eigen_Decomposition(par, observables)
  null_eigenvectors = eigen_vectors_normalized_with_eigen_values[abs.(eigen_vectors_normalized_with_eigen_values[:,end]).<1e-5, :]

  #find the maximum position in each null eigen_vectors
  max_positions = []
  for i in 1:size(null_eigenvectors)[1]
    max_position = argmax(abs.(null_eigenvectors[i, 1:end-1]))
    push!(max_positions, max_position)
  end

  identifiable_parameters = Set([1, 2, 3, 4, 5, 6, 7, 8, 9])
  unidentifiable_parameters = Set([])
  
  for i in 1:size(eigen_vectors_normalized_with_eigen_values)[1]

    #get the matrix 
    tmp_sensitivity_matrix = copy(hessian_matrix)
    tmp_sensitivity_matrix = tmp_sensitivity_matrix[sort(collect(identifiable_parameters)), sort(collect(identifiable_parameters))]

    #get the eigen values
    tmp_sensitivity_matrix = Symmetric(tmp_sensitivity_matrix)
    eigen_value_decomposition = eigen(tmp_sensitivity_matrix)

    eigen_values = real.(eigen_value_decomposition.values)
    eigen_vectors = real.(eigen_value_decomposition.vectors)'

    eigen_vectors_normalized = eigen_vectors
    eigen_vectors_normalized_with_eigen_values = hcat(eigen_vectors_normalized, eigen_values)

    first_eigen_vector = eigen_vectors_normalized_with_eigen_values[1, :]
    eigen_value = first_eigen_vector[end]

    if abs(eigen_value) < 1e-5
      println("The parameter ", i, " is not identifiable")

      components = abs.(first_eigen_vector[1:end-1])
      #get the maximum position
      max_position = argmax(components)
      corresponding_parameter = sort(collect(identifiable_parameters))[max_position]
      push!(unidentifiable_parameters, corresponding_parameter)
      #remove all the unidentifiable_parameters from the identifiable_parameters
      identifiable_parameters = setdiff(identifiable_parameters, unidentifiable_parameters) 

    else
      break
    end
  end

  return identifiable_parameters
end

identifiables_all_observables_variables = get_identifiable_parameters([4])
