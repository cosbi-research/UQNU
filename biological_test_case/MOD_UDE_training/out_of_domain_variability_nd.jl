module out_of_domain_variability_nd

include("ConfidenceEllipse.jl")
using Distributions, .ConfidenceEllipse, Logging, Plots, Random

export out_of_domain_var, get_ensemble_predictions, getAverageVariance, getOutOfDomainAnalysisInSinglePoint, getOutOfDomainAnalysis, computeGroundTruth

mutable struct out_of_domain_var_nd
    ranges::Array{Tuple{Float64, Float64}}
    num_points::Int
    vector_field_function::Function
    ground_truth_function::Function
    experimental_points::Array{Array{Float64}}
    ground_truth_y::Array{Array{Float64}}
    points::Array{Float64}
    dimension::Int
end

function computeGroundTruth(out_of_domain_grid)
    #create two matrixes with the same number of elements as x * y
    
    out_of_domain_grid.ground_truth_y = [out_of_domain_grid.ground_truth_function(out_of_domain_grid.points[:, i]) for i in axes(out_of_domain_grid.points, 2)]
    
    @info "Ground truth vector field computed"

    return
end

function computePoints(out_of_domain_grid)
        #trasform the array of points in a matrix with columns the points
        seed = 0
        tmp_rng = MersenneTwister(seed)
        
        points = [rand(tmp_rng, out_of_domain_grid.dimension) for i in 1:out_of_domain_grid.num_points]

        for point in points 
            for d in 1:out_of_domain_grid.dimension
                point[d] = out_of_domain_grid.ranges[d][1] + (out_of_domain_grid.ranges[d][2] - out_of_domain_grid.ranges[d][1]) * point[d]
            end
        end
    
        #each column is a point
        points = hcat(points...) # each row is a point
        
        out_of_domain_grid.points = points
        return 
end    

# get the predictions of the ensemble on a matrix of points point 
function get_ensemble_predictions(out_of_domain_grid, ensemble, points)
    predictions = []
    for i in axes(ensemble, 1)
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

    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, reshape(point,out_of_domain_grid.dimension,1))
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
function getOutOfDomainAnalysisInSinglePoint(out_of_domain_grid, ensemble, ground_truth, point_index)
    #get the predictions of the ensemble on the out of domain grid
    predictions = get_ensemble_predictions(out_of_domain_grid, ensemble, out_of_domain_grid.points[:, point_index])

    mean_dy = [mean([pred[i] for pred in predictions]) for i in 1:out_of_domain_grid.dimension]

    confidence_ellipse = ConfidenceEllipse.compute_confidence_ellipse(hcat([[pred[i] for pred in predictions] for i in 1:out_of_domain_grid.dimension]...), 0.95)
    gound_truth_in_ellipse = ConfidenceEllipse.is_point_inside_ellipse(ground_truth, confidence_ellipse)
    confidece_ellipse_area = ConfidenceEllipse.get_ellipse_volume(confidence_ellipse)

    return (mean = mean_dy, confidence_ellipse = confidence_ellipse, gound_truth_in_Ci = gound_truth_in_ellipse, area = confidece_ellipse_area, gound_truth = ground_truth) 
end

#perform global analysis of out of domain generalization
function getOutOfDomainAnalysis(out_of_domain_grid, ensemble)

    @debug "computing the out of domain analysis"

    results = []
    for point_index in 1:out_of_domain_grid.num_points
        ground_truth = out_of_domain_grid.ground_truth_y[point_index]
        push!(results, getOutOfDomainAnalysisInSinglePoint(out_of_domain_grid, ensemble, ground_truth, point_index))
    end

    cicp = sum([r.gound_truth_in_Ci for r in results]) / length(results)
    mean_area = mean([r.area for r in results])

    @debug "out of domain analysis computed"

    return (cicp=cicp, mean = [r.mean for r in results], ground_truth_in_Ci = [r.gound_truth_in_Ci for r in results], mean_area = mean_area, ellipse_areas = [r.area for r in results], ground_truths = [r.gound_truth for r in results])
end

#plot the results 
function plotOutOfDomainAnalysis(out_of_domain_grid, out_of_domain_analysis)
    
    #print the result of the analysis 
    @info "CICP: $(out_of_domain_analysis.cicp)"
    @info "Mean Ellipse Area: $(out_of_domain_analysis.mean_area)"
    
    return
end

function getDistance(first_point, second_point)
    return sqrt(sum((first_point - second_point).^2))
end

#get the minimum distance between each out of domain point and the experimental points
function getOutOfDomainDistance(out_of_domain_grid)
    points = out_of_domain_grid.points
    distances = []
    for j in axes(points, 2)
        point = points[:, j]
        minDistance = Inf
        for experimental_point in out_of_domain_grid.experimental_points
            tmp = getDistance(point, experimental_point)
            if tmp < minDistance
                minDistance = tmp
            end
        end
        push!(distances, minDistance)
    end
    return distances
end

end # module
