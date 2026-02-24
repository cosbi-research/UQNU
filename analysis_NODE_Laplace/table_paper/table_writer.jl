cd(@__DIR__)

using Serialization, Printf

#deserialize the CP results 
results_lv = deserialize("../analysis_results_lv/comparison_results_lv.jld")
results_damped = deserialize("../analysis_results_damped/comparison_results_damped.jld")   
results_lorenz = deserialize("../analysis_results_lorenz/comparison_results_lorenz.jld")

#read the template as a string
template = read("table_template.txt", String)

lv_standard_placeholder = "##LV_standard##"
lv_maximized_placeholder = "##LV_maximized##"
damped_standard_placeholder = "##DAMPED_standard##"
damped_maximized_placeholder = "##DAMPED_maximized##"
lorenz_standard_placeholder = "##LORENZ_standard##"
lorenz_maximized_placeholder = "##LORENZ_maximized##"

function get_text(mean, sem, pval_value, print_pval=false)
    rounded_mean = round(mean, digits=3)
    rounded_sem = round(sem, digits=3)

    #as string
    rounded_mean = @sprintf("%.3f", mean)
    rounded_sem = @sprintf("%.3f", rounded_sem)

    p_val = ""
    if pval_value < 0.001
        p_val = "(***)"
    elseif pval_value < 0.01
        p_val = "(**)"
    elseif pval_value < 0.05
        p_val = "(*)"
    end

   
    res= string(rounded_mean, 
            " ± ", rounded_sem)

    if print_pval && pval_value < 0.05
        res = string("\\textbf{", res, "}"," ", p_val)
    end

    if print_pval==false
        res = string(res, " ", "\\quad \\quad \\quad")
    end

    return res
end

#replace the placeholders in the template with the results
lv_standard_text = get_text(
    results_lv.results_vector_field.mean_cp_standard, 
    results_lv.results_vector_field.sem_cp_standard, 
    results_lv.results_vector_field.comparison_p_val
)
lv_maximized_text = get_text(
    results_lv.results_vector_field.mean_cp_maximized, 
    results_lv.results_vector_field.sem_cp_maximized, 
    results_lv.results_vector_field.comparison_p_val,
    true
)
damped_standard_text = get_text(
    results_damped.results_vector_field.mean_cp_standard, 
    results_damped.results_vector_field.sem_cp_standard, 
    results_damped.results_vector_field.comparison_p_val
)
damped_maximized_text = get_text(
    results_damped.results_vector_field.mean_cp_maximized, 
    results_damped.results_vector_field.sem_cp_maximized, 
    results_damped.results_vector_field.comparison_p_val,
    true
)
lorenz_standard_text = get_text(
    results_lorenz.results_vector_field.mean_cp_standard, 
    results_lorenz.results_vector_field.sem_cp_standard, 
    results_lorenz.results_vector_field.comparison_p_val
)
lorenz_maximized_text = get_text(
    results_lorenz.results_vector_field.mean_cp_maximized, 
    results_lorenz.results_vector_field.sem_cp_maximized, 
    results_lorenz.results_vector_field.comparison_p_val,
    true
)

#replace the placeholders in the template with the results
vector_field_table = replace(template, 
    lv_standard_placeholder => lv_standard_text,
    lv_maximized_placeholder => lv_maximized_text,
    damped_standard_placeholder => damped_standard_text,
    damped_maximized_placeholder => damped_maximized_text,
    lorenz_standard_placeholder => lorenz_standard_text,
    lorenz_maximized_placeholder => lorenz_maximized_text
)

#write the table in a file
table_file = "vector_field_results.txt"
open(table_file, "w") do file
    write(file, vector_field_table)
end

#write the trajectory 

lv_standard_text = get_text(
    results_lv.results_trajectory.mean_cp_trajectory_standard, 
    results_lv.results_trajectory.sem_cp_trajectory_standard, 
    results_lv.results_trajectory.comparison_p_val
)
lv_maximized_text = get_text(
    results_lv.results_trajectory.mean_cp_trajectory_maximized, 
    results_lv.results_trajectory.sem_cp_trajectory_maximized, 
    results_lv.results_trajectory.comparison_p_val,
    true
)
damped_standard_text = get_text(
    results_damped.results_trajectory.mean_cp_trajectory_standard, 
    results_damped.results_trajectory.sem_cp_trajectory_standard, 
    results_damped.results_trajectory.comparison_p_val
)
damped_maximized_text = get_text(
    results_damped.results_trajectory.mean_cp_trajectory_maximized, 
    results_damped.results_trajectory.sem_cp_trajectory_maximized, 
    results_damped.results_trajectory.comparison_p_val,
    true
)
lorenz_standard_text = get_text(
    results_lorenz.results_trajectory.mean_cp_trajectory_standard, 
    results_lorenz.results_trajectory.sem_cp_trajectory_standard, 
    results_lorenz.results_trajectory.comparison_p_val
)
lorenz_maximized_text = get_text(
    results_lorenz.results_trajectory.mean_cp_trajectory_maximized, 
    results_lorenz.results_trajectory.sem_cp_trajectory_maximized, 
    results_lorenz.results_trajectory.comparison_p_val,
    true
)

tarjectory_table = replace(template, 
    lv_standard_placeholder => lv_standard_text,
    lv_maximized_placeholder => lv_maximized_text,
    damped_standard_placeholder => damped_standard_text,
    damped_maximized_placeholder => damped_maximized_text,
    lorenz_standard_placeholder => lorenz_standard_text,
    lorenz_maximized_placeholder => lorenz_maximized_text
)

#write the table in a file
trajectory_file = "trajectory_results.txt"
open(trajectory_file, "w") do file
    write(file, tarjectory_table)
end
