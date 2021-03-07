using LinearAlgebra
include("FEM1D.jl")
include("containers.jl")
include("Iterations.jl")

function ρ_error(dimension::Integer, k::Integer, iteration::Function, batch_size::Integer)
    x = randn((dimension, batch_size))
    y = x
    for i = 1:k
        y = iteration(y)
    end
    ρ = 0
    n_y, n_x = 0, 0
    for j = 1:batch_size
        ρ += (norm(y[:, j])/norm(x[:, j]))^(1/k)
    end
    ρ = ρ/batch_size
end

function ρ_F(dimension::Integer, k::Integer, iteration::Function, batch_size::Integer)
    x = rand((-1, 1), (dimension, batch_size))
    for l = 1:k
        x = iteration(x)
    end
    ρ = (norm(x)^2/batch_size)^(1/(2*k))
end

function prec_loss_function(parameters::AbstractArray{<:Real, 1}, prec::preconditioner, params::optimization_parameters)
    M = assemble_matrix(parameters, prec)
    iteration(x) = Richardson(x, prec.A, x -> M*x, parameters[end]; prec=prec.symmetry)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function spectral_loss_function(parameters::AbstractArray{<:Real, 1}, prec::preconditioner, params::optimization_parameters)
    M = assemble_matrix(parameters, prec)
    iteration(x) = Richardson(x, prec.A, x -> M*x, 1/spectral_radius(parameters, prec, params); prec=prec.symmetry)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function scalar_loss_function(parameters::AbstractArray{<:Real, 1}, M::SparseMatrixCSC{Float64,Int64}, prec::preconditioner, params::optimization_parameters)
    iteration(x) = Richardson(x, prec.A, x -> M*x, parameters[end]; prec=prec.symmetry)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function ρ_error_T(dimension::Tuple{Integer, Integer}, k::Integer, iteration::Function, batch_size::Integer)
    n_y, n_x = 0, 0
    for j = 1:batch_size
        x = randn(dimension)
        n_x += norm(x)
        y = x
        for i = 1:k
            y = iteration(y)
        end
        n_y += norm(y)
    end
    ρ = (n_y/n_x)^(1/k)
end

function ρ_F_T(dimension::Tuple{Integer, Integer}, k::Integer, iteration::Function, batch_size::Integer)
    n_x = 0
    for j = 1:batch_size
        x = rand((-1, 1), dimension)
        for l = 1:k
            x = iteration(x)
        end
        n_x += norm(x)^2
    end
    ρ = (n_x/batch_size)^(1/(2*k))
end

function prec_loss_function(parameters::AbstractArray{<:Real, 1}, prec::T_preconditioner, params::optimization_parameters)
    N = 2^prec.L+1
    DOF_x, DOF_y = N - 2 + sum(prec.BCs[1:2] .== "N"), N - 2 + sum(prec.BCs[3:4] .== "N")
    dimension = (DOF_y, DOF_x)
    iteration(x) = Richardson_T(x, prec.C, x -> apply_T_preconditioner(x, parameters, prec), parameters[end]; prec=prec.symmetry)
    if params.error == "Frobenius"
        loss = ρ_F_T(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error_T(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function spectral_loss_function(parameters::AbstractArray{<:Real, 1}, prec::T_preconditioner, params::optimization_parameters)
    N = 2^prec.L+1
    DOF_x, DOF_y = N - 2 + sum(prec.BCs[1:2] .== "N"), N - 2 + sum(prec.BCs[3:4] .== "N")
    dimension = (DOF_y, DOF_x)
    iteration(x) = Richardson_T(x, prec.C, x -> apply_T_preconditioner(x, parameters, prec), 1/spectral_radius(parameters, prec, params); prec=prec.symmetry)
    if params.error == "Frobenius"
        loss = ρ_F_T(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error_T(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function Frobenius_loss_function(parameters::AbstractArray{<:Real, 1}, prec::T_preconditioner, params::optimization_parameters)
    N = 2^prec.L+1
    DOF_x, DOF_y = N - 2 + sum(prec.BCs[1:2] .== "N"), N - 2 + sum(prec.BCs[3:4] .== "N")
    dimension = (DOF_y, DOF_x)
    iteration(x) = x
    if prec.symmetry == "symmetric"
        iteration(x) = x - apply_T_preconditioner(reshape(prec.A*reshape(apply_T_preconditioner(x, parameters, prec), (DOF_y*DOF_x, )), dimension), parameters, prec)
    else
        iteration(x) = x - apply_T_preconditioner(reshape(prec.A*reshape(x, (DOF_y*DOF_x, )), dimension), parameters, prec)
    end
    loss = ρ_F_T(dimension, 1, iteration, params.k_batch)
end

function Frobenius_loss_function(parameters::AbstractArray{<:Real, 1}, prec::preconditioner, params::optimization_parameters)
    M = assemble_matrix(parameters, prec)
    iteration(x) = x
    if prec.symmetry == "left"
        iteration(x) = x - M*(prec.A*x)
    else
        iteration(x) = x - M*(prec.A*(M*x))
    end
    loss = ρ_F(size(prec.A)[1], 1, iteration, params.k_batch)
end

function scalar_loss_function(parameters::AbstractArray{<:Real, 1}, M::SparseMatrixCSC{Float64,Int64}, prec::T_preconditioner, params::optimization_parameters)
    iteration(x) = Richardson(x, prec.A, x -> M*x, parameters[end]; prec=prec.symmetry)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, iteration, params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, iteration, params.k_batch)
    end
    return loss
end

function spectral_radius(parameters::AbstractArray{<:Real, 1}, prec::preconditioner, params::optimization_parameters)
    function iteration(x::AbstractArray{<:Real, 2}, M::SparseMatrixCSC{<:Real, <:Integer}, A::SparseMatrixCSC{<:Real, <:Integer}, symmetry::String)
        if symmetry == "symmetric"
            return M*A*M*x
        elseif symmetry == "left"
            return M*A*x
        end
    end
    M = assemble_matrix(parameters, prec)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, x -> iteration(x, M, prec.A, prec.symmetry), params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, x -> iteration(x, M, prec.A, prec.symmetry), params.k_batch)
    end
    return loss
end

function spectral_radius(parameters::AbstractArray{<:Real, 1}, prec::T_preconditioner, params::optimization_parameters)
    function iteration(x::AbstractArray{<:Real, 2}, M::SparseMatrixCSC{<:Real, <:Integer}, A::SparseMatrixCSC{<:Real, <:Integer}, symmetry::String)
        if symmetry == "symmetric"
            return M*A*M*x
        elseif symmetry == "left"
            return M*A*x
        end
    end
    M = assemble_matrix(parameters, prec)
    dimension = size(prec.A)[1]
    if params.error == "Frobenius"
        loss = ρ_F(dimension, params.k_sweeps, x -> iteration(x, M, prec.A, prec.symmetry), params.k_batch)
    elseif params.error == "Spectral"
        loss = ρ_error(dimension, params.k_sweeps, x -> iteration(x, M, prec.A, prec.symmetry), params.k_batch)
    end
    return loss
end

function test_ρ_1(k::Integer, N::Integer, L::Integer)
    A, b, x = BVP_DD(L, x -> one(x), x -> zero(x), 1, 1, x -> x)
    ρ = maximum(abs.(eigvals(Array(A))))
    ρ_err = ρ_error(2^L-1, k, x -> A*x)
    ρ_Frob = ρ_F(2^L-1, k, N, x -> A*x)
    println("Spectral radius ", ρ)
    println("Estimation with error ", ρ_err)
    println("Estimation with Frobenius ", ρ_Frob)
end

function test_ρ_2(k::Integer, L::Integer)
    A, b, x = BVP_DD(L, x -> 1 + x^2, x -> exp(x), 1, 1, x -> x)
    ρ = maximum(abs.(eigvals(Array(A))))
    ρ_estimated = ρ_error(2^L-1, k, x -> A*x)
    println("Spectral radius ", ρ)
    println("Estimation ", ρ_estimated)
end
