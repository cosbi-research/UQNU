cd(@__DIR__)

using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) =  -p1 * u[1] - p2 * u[2] * u[1],
    x2'(t) =  p3 * u[1] * u[2] - p4 * u[2],
    y1(t) = x1(t),
    y2(t) = x2(t)
)

assess_identifiability(ode)