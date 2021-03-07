include("preconditioners.jl")
include("optimization_loops.jl")
include("Losses.jl")
include("Iterations.jl")
include("equations.jl")
include("containers.jl")
using Flux.Optimise

function basic_optimization(params::optimization_parameters)
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, B, parameters = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    # initial_condition_number = measure_condition_number(parameters, prec)
    loss_function(x::AbstractArray{<:Real, 1}) = prec_loss_function(x, prec, params)
    history = standard_optimization!(parameters, loss_function, params.epochs, params.optim)
    # final_condition_number = measure_condition_number(parameters, prec)
    return history, prec, parameters # (initial_condition_number, final_condition_number), history, prec, parameters
end

function stopping_optimization(params::optimization_parameters; check_time=10, tol=1e-2)
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, B, parameters = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    pre_opt, post_opt = deepcopy(params.optim), deepcopy(params.optim)

    # θ = parameters[end:end]
    # M = assemble_matrix(parameters, prec)
    # pre_loss_function(x::AbstractArray{<:Real, 1}) = scalar_loss_function(x, M, prec, params)
    # history_pre = early_stopping_optimization!(θ, pre_loss_function, params.epochs, pre_opt; check_time=check_time, tol=tol)
    # parameters[end] = θ[1]

    loss_function(x::AbstractArray{<:Real, 1}) = prec_loss_function(x, prec, params)
    history = early_stopping_optimization!(parameters, loss_function, params.epochs, params.optim; check_time=check_time, tol=tol)

    # θ = parameters[end:end]
    # M = assemble_matrix(parameters, prec)
    # post_loss_function(x::AbstractArray{<:Real, 1}) = scalar_loss_function(x, M, prec, params)
    # history_post = early_stopping_optimization!(θ, post_loss_function, params.epochs, post_opt; check_time=check_time, tol=tol)
    # parameters[end] = θ[1]

    return history, prec, parameters
end

function stopping_optimization_continuation(params::optimization_parameters, parameters::Array{Float64, 1}; check_time=10, tol=1e-2)
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, _, _ = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)

    loss_function(x::AbstractArray{<:Real, 1}) = prec_loss_function(x, prec, params)
    history = early_stopping_optimization!(parameters, loss_function, params.epochs, params.optim; check_time=check_time, tol=tol)

    return history, prec, parameters
end

function modified_optimization(params::optimization_parameters)
    # A = params.equation(params.L, params.BCs, params.eq_params)
    # prec, B, parameters = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    # initial_condition_number = measure_s_condition_number(parameters, prec, params)
    # loss_function(x::AbstractArray{<:Real, 1}) = spectral_loss_function(x, prec, params)
    # history = standard_optimization!(parameters, loss_function, params.epochs, params.optim)
    # final_condition_number = measure_s_condition_number(parameters, prec, params)
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, B, parameters = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    loss_function(x::AbstractArray{<:Real, 1}) = prec_loss_function(x, prec, params)
    ρ(x::AbstractArray{<:Real, 1}) = spectral_radius(x, prec, params)
    history = spectral_optimization!(parameters, loss_function, ρ, params.epochs, params.optim)
    return history, prec, parameters
end

function modified_optimization_continuation(params::optimization_parameters, parameters::Array{Float64, 1})
    # A = params.equation(params.L, params.BCs, params.eq_params)
    # prec, B, parameters = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    # initial_condition_number = measure_s_condition_number(parameters, prec, params)
    # loss_function(x::AbstractArray{<:Real, 1}) = spectral_loss_function(x, prec, params)
    # history = standard_optimization!(parameters, loss_function, params.epochs, params.optim)
    # final_condition_number = measure_s_condition_number(parameters, prec, params)
    A = params.equation(params.L, params.BCs, params.eq_params)
    prec, B, _ = initialise_preconditioner(A, params.L, params.BCs, params.name, params.symmetry)
    loss_function(x::AbstractArray{<:Real, 1}) = prec_loss_function(x, prec, params)
    ρ(x::AbstractArray{<:Real, 1}) = spectral_radius(x, prec, params)
    history = spectral_optimization!(parameters, loss_function, ρ, params.epochs, params.optim)
    return history, prec, parameters
end
