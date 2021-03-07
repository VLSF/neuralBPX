using ForwardDiff, DiffResults
using FiniteDiff
include("preconditioners.jl")

function standard_optimization!(parameters::Array{Float64, 1}, loss_function::Function, epochs::Integer, optim)
    result = DiffResults.GradientResult(parameters)
    # cfg = ForwardDiff.GradientConfig(loss_function, parameters, ForwardDiff.Chunk{length(parameters)}())
    history = zeros(epochs)
    for i = 1:epochs
        ForwardDiff.gradient!(result, loss_function, parameters)
        update!(optim, parameters, DiffResults.gradient(result))
        history[i] = DiffResults.value(result)
    end
    return history
end

function early_stopping_optimization!(parameters::Array{Float64, 1}, loss_function::Function, epochs::Integer, optim; check_time::Integer=10, tol::Real=1e-2)
    result = DiffResults.GradientResult(parameters)
    # cfg = ForwardDiff.GradientConfig(loss_function, parameters, ForwardDiff.Chunk{length(parameters)}())
    history = []
    j = 1
    running_average = []
    while j<=epochs
        ForwardDiff.gradient!(result, loss_function, parameters)
        update!(optim, parameters, DiffResults.gradient(result))
        push!(history, DiffResults.value(result))
        # println(DiffResults.gradient(result))
        if (j%check_time) == 0
            push!(running_average, sum(history[(end-check_time+1):end])/check_time)
            if length(running_average)>=2
                difference = abs(running_average[end] - running_average[end-1])
                if difference<tol
                    break
                end
            end
        end
        j += 1
    end
    return history
end

function tracking_optimization!(parameters::Array{Float64, 1}, prec::preconditioner, params::optimization_parameters, loss_function::Function)
    result = similar(parameters)
    history = zeros(params.epochs)
    condition_number = zeros(params.epochs)
    for i = 1:params.epochs
        ForwardDiff.gradient!(result, loss_function, parameters)
        value = loss_function(parameters)
        update!(params.optim, parameters, result)
        history[i] = value
        # condition_number[i] = measure_s_condition_number(parameters, prec, params)
        condition_number[i] = measure_condition_number(parameters, prec)
    end
    return history, condition_number
end

function tracking_optimization!(parameters::Array{Float64, 1}, prec::T_preconditioner, params::optimization_parameters, loss_function::Function)
    result = similar(parameters)
    history = zeros(params.epochs)
    condition_number = zeros(params.epochs)
    for i = 1:params.epochs
        ForwardDiff.gradient!(result, loss_function, parameters)
        value = loss_function(parameters)
        update!(params.optim, parameters, result)
        history[i] = value
        # condition_number[i] = measure_s_condition_number(parameters, prec, params)
        condition_number[i] = measure_condition_number(parameters, prec)
    end
    return history, condition_number
end

function spectral_optimization!(parameters::Array{Float64, 1}, loss_function::Function, ρ::Function, epochs::Integer, optim)
    result = DiffResults.GradientResult(parameters)
    result_2 = DiffResults.GradientResult(parameters)
    grads = Array{Float64, 1}(undef, length(parameters))
    history = zeros(epochs)
    for i = 1:epochs
        ForwardDiff.gradient!(result_2, ρ, parameters)
        parameters[end] = 1/(DiffResults.value(result_2))
        ForwardDiff.gradient!(result, loss_function, parameters)
        grads .= DiffResults.gradient(result) - parameters[end]^2 * DiffResults.gradient(result)[end] * DiffResults.gradient(result_2)
        update!(optim, parameters, grads)
        history[i] = DiffResults.value(result)
    end
    return history
end

function tracking_spectral_optimization!(parameters::Array{Float64, 1}, prec::T_preconditioner, params::optimization_parameters, loss_function::Function, ρ::Function, epochs::Integer, optim)
    result = DiffResults.GradientResult(parameters)
    result_2 = DiffResults.GradientResult(parameters)
    grads = Array{Float64, 1}(undef, length(parameters))
    condition_number = zeros(epochs)
    history = zeros(epochs)
    for i = 1:epochs
        ForwardDiff.gradient!(result_2, ρ, parameters)
        parameters[end] = 1/(DiffResults.value(result_2))
        ForwardDiff.gradient!(result, loss_function, parameters)
        grads .= DiffResults.gradient(result) - parameters[end]^2 * DiffResults.gradient(result)[end] * DiffResults.gradient(result_2)
        update!(optim, parameters, grads)
        history[i] = DiffResults.value(result)
        # condition_number[i] = measure_s_condition_number(parameters, prec, params)
        condition_number[i] = measure_condition_number(parameters, prec)
    end
    return history, condition_number
end

function tracking_spectral_optimization!(parameters::Array{Float64, 1}, prec::preconditioner, params::optimization_parameters, loss_function::Function, ρ::Function, epochs::Integer, optim)
    result = DiffResults.GradientResult(parameters)
    result_2 = DiffResults.GradientResult(parameters)
    grads = Array{Float64, 1}(undef, length(parameters))
    condition_number = zeros(epochs)
    history = zeros(epochs)
    for i = 1:epochs
        ForwardDiff.gradient!(result_2, ρ, parameters)
        parameters[end] = 1/(DiffResults.value(result_2))
        ForwardDiff.gradient!(result, loss_function, parameters)
        grads .= DiffResults.gradient(result) - parameters[end]^2 * DiffResults.gradient(result)[end] * DiffResults.gradient(result_2)
        update!(optim, parameters, grads)
        history[i] = DiffResults.value(result)
        # condition_number[i] = measure_s_condition_number(parameters, prec, params)
        condition_number[i] = measure_condition_number(parameters, prec)
    end
    return history, condition_number
end
