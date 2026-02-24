module out_of_domain_variability_3d

include("ConfidenceEllipse.jl")
using Distributions, .ConfidenceEllipse, Logging, Plots

export out_of_domain_var, get_ensemble_predictions, getAverageVariance, getOutOfDomainAnalysisInSinglePoint, getOutOfDomainAnalysis, computeGroundTruth

mutable struct out_of_domain_var_3d
    x_range::AbstractRange
    y_range::AbstractRange
    z_range::AbstractRange
    vector_field_function::Function
    ground_truth_function::Function
    experimental_points::Array{Array{Float64}}
    ground_truth_y1::Array{Float64}
    ground_truth_y2::Array{Float64}
    ground_truth_y3::Array{Float64}
    points::Array{Float64}
end

function computeGroundTruth(out_of_domain_grid)
    #create two matrixes with the same number of elements as x * y
    
    ground_truth_y1 = [out_of_domain_grid.ground_truth_function([x, y, z])[1] for x in out_of_domain_grid.x_range, y in out_of_domain_grid.y_range, z in out_of_domain_grid.z_range]
    ground_truth_y2 = [out_of_domain_grid.ground_truth_function([x, y, z])[2] for x in out_of_domain_grid.x_range, y in out_of_domain_grid.y_range, z in out_of_domain_grid.z_range]
    ground_truth_y3 = [out_of_domain_grid.ground_truth_function([x, y, z])[3] for x in out_of_domain_grid.x_range, y in out_of_domain_grid.y_range, z in out_of_domain_grid.z_range]

    out_of_domain_grid.ground_truth_y1 = ground_truth_y1
    out_of_domain_grid.ground_truth_y2 = ground_truth_y2
    out_of_domain_grid.ground_truth_y3 = ground_truth_y3

    @info "Ground truth vector field computed"

    return
end

function computePoints(out_of_domain_grid)
        #trasform the array of points in a matrix with columns the points
        points = vec([[x, y, z] for x in out_of_domain_grid.x_range, y in out_of_domain_grid.y_range, z in out_of_domain_grid.z_range])
        points = hcat(points...)

        out_of_domain_grid.points = points
        return 
end    

# get the predictions of the ensemble on a matrix of points point 
function get_ensemble_predictions(out_of_domain_grid, ensemble, points)
    predictions = []
    for i in 1:length(ensemble)
        prediction = out_of_domain_grid.vector_field_function(points, ensemble[i])
        push!(predictions, prediction)
    end
    return predictions
end

# get the average variance of the ensemble on the out of domain grid
function getAverageVariance(out_of_domain_grid, ensemble)
    average_variance = 0.0

    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, out_of_domain_grid.points)
    predictions = stack(predictions, dims = 3)
    variances = var(predictions, dims = 3)[:,:, 1]
    variances = sum(variances, dims = 1)
    average_variance = mean(variances)
    
    return average_variance
end

function getVarianceInPoint(out_of_domain_grid, ensemble, point)
    average_variance = 0.0

    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, reshape(point,3,1))
    #for each component in each point, compute the variance
    predictions = stack(predictions, dims = 3)
    variances = var(predictions, dims = 3)[:,:, 1]
    variances = sum(variances, dims = 1)
    average_variance = mean(variances)

    return average_variance
end

function getVarianceInPoints(out_of_domain_grid, ensemble, points)
    average_variance = 0.0

    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, points)
    #for each component in each point, compute the variance
    predictions = stack(predictions, dims = 3)
    variances = var(predictions, dims = 3)[:,:, 1]
    variances = sum(variances, dims = 1)
    average_variance = mean(variances)

    return average_variance
end

#perform analysis of out of domain generalization in a single point
function getOutOfDomainAnalysisInSinglePoint(out_of_domain_grid, ensemble, ground_truth, x_index, y_index, z_index)
    #get the predictions of the ensemble on the out of domain grid
    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, [out_of_domain_grid.x_range[x_index], out_of_domain_grid.y_range[y_index], out_of_domain_grid.z_range[z_index]])

    mean_dy1 = mean([pred[1] for pred in predictions])
    mean_dy2 = mean([pred[2] for pred in predictions])
    mean_dy3 = mean([pred[3] for pred in predictions])

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat([pred[1] for pred in predictions], [pred[2] for pred in predictions], [pred[3] for pred in predictions]), 0.95)
    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(ground_truth, confidence_ellipse)
    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean = [mean_dy1, mean_dy2, mean_dy3], confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = ground_truth) 
end

#perform global analysis of out of domain generalization
function getOutOfDomainAnalysis(out_of_domain_grid, ensemble)

    @debug "computing the out of domain analysis"

    results = []
    for x_index in 1:length(out_of_domain_grid.x_range)
        for y_index in 1:length(out_of_domain_grid.y_range)
            for z_index in 1:length(out_of_domain_grid.z_range)
                ground_truth = [out_of_domain_grid.ground_truth_y1[x_index, y_index, z_index], out_of_domain_grid.ground_truth_y2[x_index, y_index, z_index], out_of_domain_grid.ground_truth_y3[x_index, y_index, z_index]]
                push!(results, getOutOfDomainAnalysisInSinglePoint(out_of_domain_grid, ensemble, ground_truth, x_index, y_index, z_index))
            end
        end
    end

    cicp = sum([r.gound_truth_in_Ci for r in results]) / length(results)
    mean_area = mean([r.area for r in results])

    @debug "out of domain analysis computed"

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

#plot the results 
function plotOutOfDomainAnalysis(out_of_domain_grid, out_of_domain_analysis)
    
    #plot the CICP on the OOD region considered
    points_matrix = [[x, y, z] for x in out_of_domain_grid.x_range, y in out_of_domain_grid.y_range, z in out_of_domain_grid.z_range]
    points = vec(points_matrix)
    
    cicp_on_points = out_of_domain_analysis.ground_truth_in_Ci

    plt_1 = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=cicp_on_points, color=:viridis)
    Plots.scatter!(plt_1, [p[1] for p in out_of_domain_grid.experimental_points], [p[2] for p in out_of_domain_grid.experimental_points], label="experimental data", legend = false)
    
    areas = out_of_domain_analysis.ellipse_areas

    plt_2 = Plots.scatter([p[1] for p in points], [p[2] for p in points], label="sampled points", xlabel="y1", ylabel="y2", title="Sampled points", zcolor=areas, color=:viridis)
    Plots.scatter!(plt_2, [p[1] for p in out_of_domain_grid.experimental_points], [p[2] for p in out_of_domain_grid.experimental_points], label="experimental data", legend = false)

    return (cicp_out_of_domain = plt_1, areas_out_of_domain = plt_2)
end

end # module
