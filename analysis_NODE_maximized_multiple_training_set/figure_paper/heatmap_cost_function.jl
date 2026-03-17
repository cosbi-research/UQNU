cd(@__DIR__)

using Plots; pythonplot()

# Define the function
f(x, y) = 1.0 + (log(20, 1 + abs(y - x^2)))^2 * abs(y - 3x^2)^2 *  abs(y + 0.1 * x^2)^2 *  abs(y +  exp(x))^2 * abs(y^2 - 3x)^2 * abs(y - 6x^2)^2 * log10(1+ abs(sin(x) + cos(y)))
f(x, y) = (3x + y^2) * abs(sin(x) + cos(y))


# Create the range
x = range(0, 5, length=100)
y = range(0, 8, length=50)

# Compute log-scaled values
z = (@. f(x', y))

# Create custom levels: densely spaced near the minimum of z
zmin = minimum(z)
zmax = maximum(z)

# Generate non-linear levels: denser near zmin
dense_levels = vcat(
    range(zmin, zmin + 1.0, length=5),        # dense near min
    range(zmin + 1.0, zmax, length=20)[2:end]  # sparser up to max
)

# Plot the contour
contour(x, y, z, levels=dense_levels, color=:turbo, cbar=false, lw=1)


plt = contourf(x, y, z, levels=20, color=:turbo, cbar=false)
xlabel!(plt, L"\theta_1")
ylabel!(plt, L"\theta_2")

#save the fig as png and svg
Plots.savefig(plt, "cost_function_heatmap.png")
Plots.savefig(plt, "cost_function_heatmap.svg")