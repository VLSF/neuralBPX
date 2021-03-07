include("FEM1D.jl")
include("FEM2D.jl")

function Laplace(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    F = [x -> -1, x -> 0, x -> 0, x -> x^2]
    BCs_ = [[1, BCs[1]], [2, BCs[2]]]
    A, _, _ = BVP_1D(L, F, BCs_)
    return A
end

function Laplace_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    F = [x -> -1, x -> 0, x -> -1, x -> 0, x -> 0, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Helmholtz(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    κ = parameters[1]
    F = [x -> -1, x -> 0, x -> -κ^2*1, x -> x^2]
    BCs_ = [[1, BCs[1]], [2, BCs[2]]]
    A, _, _ = BVP_1D(L, F, BCs_)
    return A
end

function Helmholtz_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    κ = parameters[1]
    F = [x -> -1, x -> 0, x -> -1, x -> 0, x -> 0, x -> -κ^2*1, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Convection_Diffusion(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    v = parameters[1]
    F = [x -> -1, x -> -v*1, x -> 0, x -> x^2]
    BCs_ = [[1, BCs[1]], [2, BCs[2]]]
    A, _, _ = BVP_1D(L, F, BCs_)
    return A
end

function Convection_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    v1 = parameters[1]
    v2 = parameters[2]
    F = [x -> -1, x -> 0, x -> -1, x -> v1*1, x -> v2*1, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Disc_Diffusion(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    σ = parameters[1]
    F = [x -> -1-σ*(x>0.5), x -> 0, x -> 0, x -> x^2]
    BCs_ = [[1, BCs[1]], [2, BCs[2]]]
    A, _, _ = BVP_1D(L, F, BCs_)
    return A
end

function Disc_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    σ = parameters[1]
    g(x) = -1*(x[1]<=0.5)*(x[2]>=0.5)-1*(x[1]>=0.5)*(x[2]<=0.5)-σ*(x[1]>0.5)*(x[2]>0.5)-1/σ*(x[1]<0.5)*(x[2]<0.5)
    F = [g, x -> 0, g, x -> 0, x -> 0, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Implicit_Diffusion(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    μ = parameters[1]
    F = [x -> μ*1, x -> 0, x -> 0, x -> x^2]
    BCs_ = [[1, BCs[1]], [2, BCs[2]]]
    A, _, _ = BVP_1D(L, F, BCs_)
    A = I - A
    return A
end

function Implicit_Diffusion_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    μ = parameters[1]
    F = [x -> -1, x -> 0, x -> -1, x -> 0, x -> 0, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    A = I - μ*A
    return A
end

function Anisotropic_Laplace_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    ϵ = parameters[1]
    F = [x -> -1, x -> 0, x -> -ϵ*1, x -> 0, x -> 0, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Mixed_Derivative_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    ϵ = parameters[1]
    F = [x -> -1, x -> -ϵ*1, x -> -1, x -> 0, x -> 0, x -> 0, x -> 0]
    A, _ = BVP_2D(L, F, BCs)
    return A
end

function Biharmonic_equation_2D(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    lex(i, j) = i + N*(j-1)
    for (i, j) = collect(Iterators.product(1:N, 1:N))
        for (p, q) = collect(Iterators.product([+1, -1], [+1, -1]))
            if (1<=i+p<=N) & (1<=j+q<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p, j+q))
                push!(values, +2)
            end
        end
        for p = [+1, -1]
            if (1<=i+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p, j))
                push!(values, -8)
            end
            if (1<=j+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i, j+p))
                push!(values, -8)
            end
        end
        for p = [+2, -2]
            if (1<=i+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p, j))
                push!(values, +1)
            end
            if (1<=j+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i, j+p))
                push!(values, +1)
            end
        end
        if (2<=i<=(N-1)) & (2<=j<=(N-1))
            push!(rows, lex(i, j))
            push!(columns, lex(i, j))
            push!(values, +20)
        else
            push!(rows, lex(i, j))
            push!(columns, lex(i, j))
            if ((i == 1) & (j == 1)) || ((i == N) & (j == N)) || ((i == N) & (j == 1)) || ((i == 1) & (j == N))
                push!(values, +22)
            else
                push!(values, +21)
            end
        end
    end
    A = sparse(rows, columns, values)
end

function Mehrstellen(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    lex(i, j) = i + N*(j-1)
    for (i, j) = collect(Iterators.product(1:N, 1:N))
        for (p, q) = collect(Iterators.product([+1, -1], [+1, -1]))
            if (1<=i+p<=N) & (1<=j+q<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p, j+q))
                push!(values, -1)
            end
        end
        for p = [+1, -1]
            if (1<=i+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p, j))
                push!(values, -4)
            end
            if (1<=j+p<=N)
                push!(rows, lex(i, j))
                push!(columns, lex(i, j+p))
                push!(values, -4)
            end
        end
        push!(rows, lex(i, j))
        push!(columns, lex(i, j))
        push!(values, +20)
    end
    A = sparse(rows, columns, values)
end

function fourth_Laplace_2D_1(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    lex(i, j) = i + N*(j-1)
    for (i, j) = collect(Iterators.product(1:N, 1:N))
        for p = [+2, -2]
            if (i ≠ 1) & (i ≠ N)
                if (1<=i+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i+p, j))
                    push!(values, 1)
                end
            end
            if (j ≠ 1) & (j ≠ N)
                if (1<=j+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i, j+p))
                    push!(values, 1)
                end
            end
        end
        for p = [+1, -1]
            if (i ≠ 1) & (i ≠ N)
                if (1<=i+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i+p, j))
                    push!(values, -16)
                end
            else
                if (1<=i+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i+p, j))
                    push!(values, -12)
                end
            end
            if (j ≠ 1) & (j ≠ N)
                if (1<=j+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i, j+p))
                    push!(values, -16)
                end
            else
                if (1<=j+p<=N)
                    push!(rows, lex(i, j))
                    push!(columns, lex(i, j+p))
                    push!(values, -12)
                end
            end
        end
        if (2<=i<=(N-1)) & (2<=j<=(N-1))
            push!(rows, lex(i, j))
            push!(columns, lex(i, j))
            push!(values, +60)
        else
            push!(rows, lex(i, j))
            push!(columns, lex(i, j))
            if ((i == 1) & (j == 1)) || ((i == N) & (j == N)) || ((i == N) & (j == 1)) || ((i == 1) & (j == N))
                push!(values, +48)
            else
                push!(values, +54)
            end
        end
    end
    A = sparse(rows, columns, values)
end

function fourth_Laplace_2D_2(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    A = spdiagm(0 => 30*ones(N), 1 => -16*ones(N-1), -1 => -16*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    A[1, 1] = 15; A[1, 2] = 4; A[1, 3] = -14; A[1, 4] = 6; A[1, 5] = -1
    A[end, end] = 15; A[end, end-1] = 4; A[end, end-2] = -14; A[end, end-3] = 6; ; A[end, end-4] = -1
    D = spdiagm(0 => ones(N))
    A = kron(A, D) + kron(D, A)
end

function naive_Biharmonic(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    # A = spdiagm(0 => 6*ones(N), -1 => -4*ones(N-1), 1 => -4*ones(N-1), 2 => ones(N-2), -2 => ones(N-2))
    A = spdiagm(0 => (28/3)*ones(N), -1 => (-13/2)*ones(N-1), 1 => (-13/2)*ones(N-1), 2 => 2*ones(N-2), -2 => 2*ones(N-2), 3 => (-1/6)*ones(N-3), -3 => (-1/6)*ones(N-3))
    return A
end

function Biharmonic_equation(L::Integer, BCs::Array{String, 1}, parameters::Array{<:Real, 1})
    N = 2^L-1
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    for i = 1:N
        for p = [+1, -1]
            if (1<=i+p<=N)
                push!(rows, i)
                push!(columns, i+p)
                push!(values, -8)
            end
        end
        for p = [+2, -2]
            if (1<=i+p<=N)
                push!(rows, i)
                push!(columns, i+p)
                push!(values, +1)
            end
        end
        if (2<=i<=(N-1))
            push!(rows, i)
            push!(columns, i)
            push!(values, +10)
        else
            push!(rows, i)
            push!(columns, i)
            push!(values, +11)
        end
    end
    A = sparse(rows, columns, values)
end

function test_Biharmonic()
    errors, H = [], []
    for L = 2:6
        N = 2^L-1
        h = 2.0^(-L)
        A = Biharmonic_equation_2D(L, ["D", "D"], [10.0, ])/h^4
        rhs = zeros(N^2)
        x = LinRange(h, 1-h, N)
        lex(i, j) = i + N*(j-1)
        u_exact = Array{Float64, 1}(undef, N^2)
        for (j, i) = collect(Iterators.product(1:N, 1:N))
            u_exact[lex(i, j)] = (x[i]*x[j]*(1-x[i])*(1-x[j]))^2
            rhs[lex(i, j)] = 8*(6*x[i]^2-6*x[i]+1)*(6*x[j]^2-6*x[j]+1)+24*((x[i]*(1-x[i]))^2+(x[j]*(1-x[j]))^2)
        end
        push!(H, h)
        push!(errors, norm(inv(Array(A))*rhs - u_exact, Inf))
    end
    errors = log10.(errors)
    H = log10.(H)
    return (errors[1] - errors[end])/(H[1] - H[end])
end

function test_Laplace()
    errors, H = [], []
    for L = 3:6
        N = 2^L-1
        h = 2.0^(-L)
        A = fourth_Laplace_2D_2(L, ["D", "D"], [10.0, ])/(12*h^2)
        rhs = zeros(N^2)
        x = LinRange(h, 1-h, N)
        lex(i, j) = i + N*(j-1)
        u_exact = Array{Float64, 1}(undef, N^2)
        for (j, i) = collect(Iterators.product(1:N, 1:N))
            u_exact[lex(i, j)] = sin(π*x[i])*sin(π*x[j])
            rhs[lex(i, j)] = 2*π^2*sin(π*x[i])*sin(π*x[j])
        end
        push!(H, h)
        push!(errors, norm(inv(Array(A))*rhs - u_exact, Inf))
    end
    errors = log10.(errors)
    H = log10.(H)
    return (errors[1] - errors[end])/(H[1] - H[end])
end
