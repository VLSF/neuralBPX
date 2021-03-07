include("optimization.jl")

function test_preconditioner(L::Integer, N_sweeps::Integer, k_batch::Integer, equation::Function, name::String)
    h = 1/2^L
    k = sqrt(1/h)
    parameters_ = [k, ]
    BCs = ["D", "D", "D", "D"]
    symmetry = "symmetric"

    params = optimization_parameters(equation, ADAM(0.001), parameters_, 10, k_batch, N_sweeps, BCs, L, name, symmetry, "Spectral")
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, B, parameters_start = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    history, prec, parameters = stopping_optimization(params; check_time=30, tol=1e-3)

    cond_init = measure_prec_condition_number(parameters_start, prec)
    cond_final = measure_condition_number(parameters, prec)
    rho_init = (cond_init-1)/(cond_init+1)
    rho_final = (cond_final-1)/(cond_final+1)
    println("Initial condition number ", round(cond_init, digits=3), " Final condition number ", round(cond_final, digits=3))
    println("Initial ρ number ", round(rho_init, digits=3), " Final ρ number ", round(rho_final, digits=3))
    println("Initial N ", round(-1/log10(rho_init), RoundUp), " Final N ", round(-1/log10(rho_final), RoundUp))
    println("Optimization length ", length(history))

    return history, prec, parameters
end
