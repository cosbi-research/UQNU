module ConfidenceEllipse

# Import necessary packages
using LinearAlgebra
using Distributions

# Export the main function
export compute_confidence_ellipse
export is_point_inside_ellipse
export get_ellipse_volume

function get_Hotelling_critical_value(sample_size, confidence, dimension)

    # Degrees of freedom
    df1 = dimension       # Numerator degrees of freedom
    df2 = sample_size - dimension   # Denominator degrees of freedom

    # Critical value for F-distribution
    F_crit = quantile(FDist(df1, df2), confidence)

    # Convert F critical value to Hotelling's T^2 threshold
    T2_threshold = dimension * (sample_size - 1) / (sample_size - dimension) * F_crit

    return T2_threshold
end

function compute_confidence_ellipse(data::Matrix, confidence_level::Float64=0.95)
    # Step 1: Compute mean and covariance
    mean_vec = mean(data, dims=1)'
    cov_matrix = cov(data)

    # Step 2: Eigenvalues and eigenvectors
    eig_vals, eig_vecs = eigen(cov_matrix)
    sorted_indices = sortperm(eig_vals, rev=true)  # Sort eigenvalues in descending order
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    dimension = size(data, 2)
    # Step 3: Scale axes based on Hotelling distribution
    statistics_value = get_Hotelling_critical_value(size(data, 1), confidence_level, dimension)  # Hotelling's T^2 value for confidence level
    #axes_lengths = sqrt.(eig_vals * statistics_value / size(data, 1))
    # It's not the confidence interval on the mean
    axes_lengths = sqrt.(eig_vals * statistics_value)

    return (center=mean_vec, semiaxis_directions=eig_vecs, semiaxis_lengths=axes_lengths)
end

function is_point_inside_ellipse(point::Vector, ellipse::NamedTuple)
    # Step 1: Translate the point to the origin
    translated_point = point .- ellipse.center

    # Step 2: rotate the point to the ellipse's coordinate system
    projections = transpose(ellipse.semiaxis_directions) * translated_point

    # Step 3: Check if the point is inside the ellipse
    return sum((projections ./ ellipse.semiaxis_lengths) .^ 2) <= 1
end

function get_ellipse_volume(ellipse::NamedTuple)
    dimension = length(ellipse.center)
    if dimension == 2
        return π * ellipse.semiaxis_lengths[1] * ellipse.semiaxis_lengths[2]
    elseif dimension == 3
        return 4/3 * π * ellipse.semiaxis_lengths[1] * ellipse.semiaxis_lengths[2] * ellipse.semiaxis_lengths[3]
    end
end

end # module


