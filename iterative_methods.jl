include("FEM1D.jl")

using LinearAlgebra, SparseArrays

function BPX_half_matrix(L::Integer, l::Integer)
    ψ = Array{Float64, 1}(undef, 2^(L-l))
    χ = Array{Float64, 1}(undef, 2^(L-l))
    ψ .= range(1, 2^(L-l), step=1)/2^(L-l)
    χ .= 1 .- ψ
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))[1:(end-1), 1:(end-1)]
    return B
end

function Schwarz_matrix_3D(A::SparseMatrixCSC{Float64, Int64}, L::Integer)
    B = spdiagm(0 => diag(A).^-1)
    for l=1:(L-1)
        P_x = BPX_half_matrix(L, l)
        P = kron(P_x, P_x, P_x)
        if l == 1
            A_P = sparse(inv(Array(transpose(P)*A*P)))
        else
            A_P = spdiagm(0 => diag(transpose(P)*A*P).^-1)
        end
        B += P*A_P*transpose(P)
    end
    return B
end

function Laplace_3D(L::Integer)
    N = 2^L - 1
    A = spdiagm(0 => -2*ones(N), 1 => ones(N-1), -1 => ones(N-1))
    D = spdiagm(0 => ones(N))
    T = [[D, D, A], [A, D, D], [D, A, D]]
    M = sum([kron(t[1], kron(t[2], t[3])) for t = T])
    return M
end

function CG_solver!(A::SparseMatrixCSC{Float64, Int64}, b::Array{<:Real, 1}, x::Array{<:Real, 1}, tol::Real)
    r::Array{Float64, 1} = b - A*x
    p::Array{Float64, 1} = r .+ 0.0
    p_A::Array{Float64, 1} = r .+ 0.0
    β::Float64 = 1.0
    α::Float64 = 1.0
    γ_0::Float64 = norm(r, Inf)
    error::Float64 = 1.0
    γ::Float64 = γ_0
    N_sweeps::Integer = 0
    while error/γ_0 > tol^2
        γ = dot(r, r)
        p_A .= A*p
        α = γ/dot(p, p_A)
        x .+= α*p
        r .-= α*(p_A)
        β = dot(r, r)/γ
        p .= r + β*p
        error = norm(b - A*x, Inf)
        N_sweeps += 1
    end
    return N_sweeps
end

function PCG_solver!(A::SparseMatrixCSC{Float64, Int64}, B::SparseMatrixCSC{Float64, Int64}, b::Array{<:Real, 1}, x::Array{<:Real, 1}, tol::Real)
    r::Array{Float64, 1} = b - A*x
    p::Array{Float64, 1} = r .+ 0.0
    p_A::Array{Float64, 1} = r .+ 0.0
    z::Array{Float64, 1} = r .+ 0.0
    β::Float64 = 1.0
    α::Float64 = 1.0
    γ_0::Float64 = norm(r, Inf)
    error::Float64 = 1.0
    γ::Float64 = γ_0
    N_sweeps::Integer = 0
    while error/γ_0 > tol^2
        γ = dot(r, z)
        p_A .= A*p
        α = γ/dot(p, p_A)
        x .+= α*p
        r .-= α*(p_A)
        z .= B*r
        β = dot(r, z)/γ
        p .= z + β*p
        error = norm(b - A*x, Inf)
        N_sweeps += 1
    end
    return N_sweeps
end

function test_CG(L::Integer)
    A = Laplace_3D(L)
    N = 2^L - 1
    h = 1/2^L
    x = randn(N^3)
    b = randn(N^3)
    N_sweeps = CG_solver!(A, b, x, 1e-6)
    println(N_sweeps)
end

function test_PCG(L::Integer)
    A = Laplace_3D(L)
    B = Schwarz_matrix_3D(A, L)
    N = 2^L - 1
    h = 1/2^L
    x = randn(N^3)
    b = randn(N^3)
    N_sweeps = PCG_solver!(A, B, b, x, 1e-6)
    println(N_sweeps)
end

function test_sparse(L::Integer)
    A = Laplace_3D(L)
    N = 2^L - 1
    b = randn(N^3)
    F = lu(A)
    x = F \ b
    return nothing
end
