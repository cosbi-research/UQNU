cd(@__DIR__)

using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -p1 * x1(t)^3 + p2,
    x2'(t) =  p2 - p1 * x2(t)^3,
    y1(t) = x1(t), 
    y2(t) = x2(t)
)

assess_identifiability(ode)