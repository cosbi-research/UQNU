cd(@__DIR__)

using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) =  (-p1 + p7) * (x2(t) - x1(t)),
    x2'(t) =  p2 * x1(t) - p3 * x2(t),
    x3'(t) = -p5 * x3(t) + p6 * x1(t) * x3(t),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t)
)

assess_identifiability(ode)