using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -p1 * x4(t) * x1(t) + p2 * x5(t),
    x2'(t) =  p3 * x5(t) - p4 * x2(t) * x3(t) + p5 * x6(t) + p6 * x6(t),
    x3'(t) = -p4 * x2(t) * x3(t) + p5 * x6(t),
    x4'(t) =  p6 * x6(t) - p1 * x4(t) * x1(t) + p2 * x5(t) -
              p7 * x4(t) * x7(t) + p8 * x8(t) + p3 * x5(t),
    x5'(t) = -p3 * x5(t) + p1 * x4(t) * x1(t) - p2 * x5(t),
    x6'(t) = -p6 * x6(t) + p4 * x2(t) * x3(t) - p5 * x6(t),
    x7'(t) = -p7 * x7(t) * x4(t) + p8 * x8(t) + p9 * x8(t),
    x8'(t) =  p7 * x7(t) * x4(t) - p8 * x8(t) - p9 * x8(t),
    y(t)   = x4(t)   # only observable
)

assess_identifiability(ode)

# configuration file for the integration and the neural network

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-4
reltol = 1e-5
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

######################################### GROUND TRUTH FUNCTION ############################################################
original_parameters = Float64[2.67 * 10^-9 *3600 * 10^5, 1*10^-2*3600, 8* 10^-3*3600, 6.8 * 10^-8*3600 * 10^5, 5*10^-2*3600, 1*10^-3*3600, 7*10^-5*3600 * 10^5, 1.67 * 10^-5*3600, 1.67*10^-4*3600]

function ca_gound_truth(u)
    p = original_parameters
    du = similar(u)
    du[1] = -p[1]*u[4]*u[1] + p[2]*u[5]
    du[2] = p[3]*u[5] - p[4]*u[2]*u[3] + p[5]*u[6] + p[6]*u[6]
    du[3] = -p[4]*u[2]*u[3] + p[5]*u[6]
    du[4] = p[6]*u[6] - p[1]*u[4]*u[1] +p[2]*u[5] - p[7]*u[4]*u[7] + p[8]*u[8] + p[3]*u[5]
    du[5] = -p[3]*u[5] + p[1]*u[4]*u[1] - p[2]*u[5]
    du[6] = -p[6]*u[6] + p[4]*u[2]*u[3] - p[5]*u[6]
    du[7] = -p[7]*u[7]*u[4] + p[8]*u[8] + p[9]*u[8]
    du[8] = p[7]*u[7]*u[4] - p[8]*u[8] - p[9]*u[8]

    return du
end

#local identifiability analysis in the point 

#compute the jacobian hypothizing y4 observable and all parameters unknown and observing 

#try with the Hessian based method