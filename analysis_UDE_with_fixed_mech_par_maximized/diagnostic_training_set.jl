module diagnostic_training_set

using Lux, OrdinaryDiffEq, Logging, Plots

export printValidationPlots, printValidationPlots3d

function get_model_simulations(θ, t, trajectory, initial_states, prob_uode_pred, integrator, reltol, abstol)
    @debug "Computing model simulations"
    if trajectory == 1
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_states[1, :, 1]
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol
        )
    elseif trajectory == 2
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_states[2, :, 1]
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol
        )
    elseif trajectory == 3
        trajectory_sol = solve(
            remake(
                prob_uode_pred;
                p=θ,
                tspan=extrema(t),
                u0=initial_states[3, :, 1]
            ),
            integrator;
            saveat=t,
            reltol=reltol,
            abstol=abstol
        )
    end

    return Array(trajectory_sol)
end

function printValidationPlots(θ, t, trajectory, initial_states, prob_uode_pred, integrator, reltol, abstol, training_data_structure)

    @debug "Computing model simulations on training set and trajectory " trajectory

    trajectory_sol = get_model_simulations(θ, t, trajectory, initial_states, prob_uode_pred, integrator, reltol, abstol)
    #plot the results
    p1 = plot(t, trajectory_sol[1, :, 1], label="Model", color="blue", linewidth=2)
    scatter!(p1, training_data_structure.solution_dataframes[trajectory].t, training_data_structure.solution_dataframes[trajectory].x1, label="Data", color="red", markersize=2)
    p2 = plot(t, trajectory_sol[2, :, 1], label="Model", color="blue", linewidth=2)
    scatter!(p2, training_data_structure.solution_dataframes[trajectory].t, training_data_structure.solution_dataframes[trajectory].x2, label="Data", color="red", markersize=2)

    #place the plots together
    p = plot(p1, p2, layout=(2, 1), size=(800, 800), legend=false)

    return p
end

function printValidationPlots3d(θ, t, trajectory, initial_states, prob_uode_pred, integrator, reltol, abstol, training_data_structure)

    @debug "Computing model simulations on training set and trajectory " trajectory

    trajectory_sol = get_model_simulations(θ, t, trajectory, initial_states, prob_uode_pred, integrator, reltol, abstol)
    #plot the results
    p1 = plot(t, trajectory_sol[1, :, 1], label="Model", color="blue", linewidth=2)
    scatter!(p1, training_data_structure.solution_dataframes[trajectory].t, training_data_structure.solution_dataframes[trajectory].x1, label="Data", color="red", markersize=2)
    p2 = plot(t, trajectory_sol[2, :, 1], label="Model", color="blue", linewidth=2)
    scatter!(p2, training_data_structure.solution_dataframes[trajectory].t, training_data_structure.solution_dataframes[trajectory].x2, label="Data", color="red", markersize=2)
    p3 = plot(t, trajectory_sol[3, :, 1], label="Model", color="blue", linewidth=2)
    scatter!(p3, training_data_structure.solution_dataframes[trajectory].t, training_data_structure.solution_dataframes[trajectory].x3, label="Data", color="red", markersize=2)

    #place the plots together
    p = plot(p1, p2, p3, layout=(3, 1), size=(800, 800), legend=false)

    return p
end

end