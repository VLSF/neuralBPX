include("FEM1D.jl")
include("FEM2D.jl")

function T_Laplace_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, A1], [A2, D2]]
end

function T_Flipped_Laplace_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 2*ones(N)/2, 1 => ones(N-1), -1 => ones(N-1))
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D]]
end

function T_Laplace_2D_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => ones(N)/2, 1 => -ones(N-1)/4, -1 => -ones(N-1)/4)
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D]]
end

function T_Laplace_2D_flipped_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => ones(N)/2, 1 => ones(N-1)/4, -1 => ones(N-1)/4)
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D]]
end

function T_Laplace_source_2D_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    c = parameters[1]
    N = 2^L-1
    A = spdiagm(0 => 2*ones(N), 1 => -ones(N-1), -1 => -ones(N-1))
    D = spdiagm(0 => ones(N))
    [[D, A/(c^2+4)], [A/(c^2+4), D], [c^2*D/(c^2+4), D]]
end

function T_Laplace_Implicit_2D_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    c = parameters[1]
    N = 2^L-1
    A = spdiagm(0 => 2*ones(N), 1 => -ones(N-1), -1 => -ones(N-1))
    D = spdiagm(0 => ones(N))
    [[D, A*c/(4*c+1)], [A*c/(4*c+1), D], [D, D/(4*c+1)]]
end

function T_Laplace_Anisotropic_2D_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    c = parameters[1]
    N = 2^L-1
    A = spdiagm(0 => 2*ones(N), 1 => -ones(N-1), -1 => -ones(N-1))
    D = spdiagm(0 => ones(N))
    [[D, A*c/(2+2*c)], [A/(2+2*c), D]]
end

function T_Biharmonic_2D_norm(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 10*ones(N), 1 => -8*ones(N-1), -1 => -8*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    B = spdiagm(-1 => ones(N-1), 1 => ones(N-1))
    D = spdiagm(0 => ones(N))
    Bd = sparse([1, N], [1, N], [1.0, 1.0])
    [[D, A/20], [A/20, D], [2*B, B/20], [Bd/20, D], [D, Bd/20]]
end

function T_Helmholtz_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    κ = parameters[1]
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, A1], [A2, D2], [-κ^2*D1, D2]]
end

function T_Convection_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    v1 = parameters[1]
    v2 = parameters[2]
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    T2, _, _ = BVP_1D(L, [x -> 0, x -> v1, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    T1, _, _ = BVP_1D(L, [x -> 0, x -> v2, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, A1], [A2, D2], [D1, T2], [T1, D2]]
end

function T_Disc_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    σ = sqrt(parameters[1])
    g(x) = -(x<=0.5)/σ - (x>0.5)*σ
    A1, _, _ = BVP_1D(L, [g, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [g, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> -g(x), x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> -g(x), x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, A1], [A2, D2]]
end

function T_Implicit_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    μ = parameters[1]
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, μ*A1], [μ*A2, D2], [spdiagm(0 => ones(size(A2)[1])), spdiagm(0 => ones(size(A1)[1]))]]
end

function T_Anisotropic_Laplace_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    ϵ = parameters[1]
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[ϵ*D1, A1], [A2, D2]]
end

function T_Mixed_Derivative_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    ϵ = parameters[1]
    A1, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    A2, _, _ = BVP_1D(L, [x -> -1, x -> 0, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    T2, _, _ = BVP_1D(L, [x -> 0, x -> -1, x -> 0, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    T1, _, _ = BVP_1D(L, [x -> 0, x -> -1, x -> 0, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D1, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[3]], [0, BCs[4]]])
    D2, _, _ = BVP_1D(L, [x -> 0, x -> 0, x -> 1, x -> 0], [[0, BCs[1]], [0, BCs[2]]])
    [[D1, A1], [A2, D2], [-2*ϵ*T1, T2]]
end

function T_Biharmonic_equation(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 10*ones(N), 1 => -8*ones(N-1), -1 => -8*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    B = spdiagm(-1 => ones(N-1), 1 => ones(N-1))
    D = spdiagm(0 => ones(N))
    Bd = sparse([1, N], [1, N], [1.0, 1.0])
    [[D, A], [A, D], [2*B, B], [Bd, D], [D, Bd]]
end

function T_Mehrstellen(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 10*ones(N), 1 => -4*ones(N-1), -1 => -4*ones(N-1))
    B = spdiagm(-1 => ones(N-1), 1 => ones(N-1))
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D], [-1*B, B]]
end

function T_fourth_Laplace_2D_2(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 30*ones(N), 1 => -16*ones(N-1), -1 => -16*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    A[1, 1] = 15; A[1, 2] = 4; A[1, 3] = -14; A[1, 4] = 6; A[1, 5] = -1
    A[end, end] = 15; A[end, end-1] = 4; A[end, end-2] = -14; A[end, end-3] = 6; ; A[end, end-4] = -1
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D]]
end

function T_fourth_Laplace_2D_1(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 30*ones(N), 1 => -16*ones(N-1), -1 => -16*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    A[1, 1] = 24; A[1, 2] = -12;
    A[end, end] = 24; A[end, end-1] = -12;
    D = spdiagm(0 => ones(N))
    [[D, A], [A, D]]
end

include("equations.jl")

function test(L::Integer)
    for BCs = Iterators.product(["D", "N"], ["D", "N"], ["D", "N"], ["D", "N"])
        X = T_Biharmonic_equation(L, [BCs...], [2.1, ])
        B = Biharmonic_equation(L, [BCs...], [2.1, ])
        A = sum(kron(C[1], C[2]) for C = X)
        println(BCs, ", ", norm(B - A, Inf))
    end
end
