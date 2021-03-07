include("FEM1D.jl")

function B_lL_DD(x::AbstractArray{<:Real, 1}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    N = length(x)+1
    m = length(ψ)
    n = Int64(N/m)
    y = Array{Float64, 1}(undef, n-1)
    for i = 1:(n-2)
        y[i] = dot(x[(1+(i-1)*m):(i*m)], χ) + dot(x[(1+i*m):((i+1)*m)], ψ)
    end
    y[end] = dot(x[(1+(n-2)*m):((n-1)*m)], χ) + dot(x[(1+(n-1)*m):end], ψ[1:(end-1)])
    return y
end

function B_Ll_DD!(x::Array{<:Real, 1}, y::Array{<:Real, 1}, ψ::Array{<:Real, 1}, χ::Array{<:Real, 1})
    n = length(y)
    m = length(ψ)
    x .= 0
    for i = 1:(n-1)
        x[(1+(i-1)*m):(i*m)] += χ*y[i]
        x[(1+i*m):((i+1)*m)] += ψ*y[i]
    end
    x[(1+(n-1)*m):(n*m)] += χ*y[end]
    x[(1+n*m):end] += ψ[1:(end-1)]*y[end]
    return nothing
end

function BPX_DD!(x::Array{<:Real, 1}, ψ::Array{<:Real, 1}, χ::Array{<:Real, 1})
    y = B_lL_DD(x, ψ, χ)
    B_Ll_DD!(x, y, ψ, χ)
    return nothing
end

function B_Ll_DD(y::AbstractArray{<:Real, 1}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    n = length(y)
    m = length(ψ)
    x = Array{Float64, 1}(undef, (n+1)*m-1)
    x .= 0
    for i = 1:(n-1)
        x[(1+(i-1)*m):(i*m)] += χ*y[i]
        x[(1+i*m):((i+1)*m)] += ψ*y[i]
    end
    x[(1+(n-1)*m):(n*m)] += χ*y[end]
    x[(1+n*m):end] += ψ[1:(end-1)]*y[end]
    return x
end

function BPX_DD(x::AbstractArray{<:Real}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    L = Int64(log2(length(x[:, 1])+1))
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))[1:(end-1), 1:(end-1)]
    y = B*(transpose(B)*x)
end

function BPX_DN(x::AbstractArray{<:Real}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    L = Int64(log2(length(x[:, 1])))
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B*(transpose(B)*x)
end

function BPX_ND(x::AbstractArray{<:Real}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    L = Int64(log2(length(x[:, 1])))
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(+1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B*(transpose(B)*x)
end

function BPX_NN(x::AbstractArray{<:Real}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    L = Int64(log2(length(x[:, 1])-1))
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    e1 = zeros(n); e1[1] = 1
    e2 = zeros((1, n+1)); e2[1] = 1
    B = vcat(e2, hcat(kron(e1, χ), (kron(Id, ψ) + kron(S, χ))))
    y = B*(transpose(B)*x)
end

function BPX_layer_operator(x::AbstractArray{<:Real}, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1}, BCs::AbstractArray{String, 1})
    if (BCs[1] == "N") & (BCs[2] == "N")
        return BPX_NN(x, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "N")
        return BPX_DN(x, ψ, χ)
    elseif (BCs[1] == "N") & (BCs[2] == "D")
        return BPX_ND(x, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "D")
        return BPX_DD(x, ψ, χ)
    end
end

function BPX_DD_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))[1:(end-1), 1:(end-1)]
    return B*transpose(B)
end

function BPX_DN_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B*transpose(B)
end

function BPX_ND_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(+1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B*transpose(B)
end

function BPX_NN_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    e1 = zeros(n); e1[1] = 1
    e2 = zeros((1, n+1)); e2[1] = 1
    B = vcat(e2, hcat(kron(e1, χ), (kron(Id, ψ) + kron(S, χ))))
    y = B*transpose(B)
end

function BPX_layer_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1}, BCs::AbstractArray{String, 1})
    if (BCs[1] == "N") & (BCs[2] == "N")
        return BPX_NN_matrix(L, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "N")
        return BPX_DN_matrix(L, ψ, χ)
    elseif (BCs[1] == "N") & (BCs[2] == "D")
        return BPX_ND_matrix(L, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "D")
        return BPX_DD_matrix(L, ψ, χ)
    end
end

function BPX_DD_half_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))[1:(end-1), 1:(end-1)]
    return B
end

function BPX_DN_half_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B
end

function BPX_ND_half_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(+1 => ones(n-1))
    B = (kron(Id, ψ) + kron(S, χ))
    y = B
end

function BPX_NN_half_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1})
    l = L - Int64(log2(length(ψ)))
    n = 2^l
    Id = spdiagm(0 => ones(n))
    S = spdiagm(-1 => ones(n-1))
    e1 = zeros(n); e1[1] = 1
    e2 = zeros((1, n+1)); e2[1] = 1
    B = vcat(e2, hcat(kron(e1, χ), (kron(Id, ψ) + kron(S, χ))))
    y = B
end

function BPX_layer_half_matrix(L::Integer, ψ::AbstractArray{<:Real, 1}, χ::AbstractArray{<:Real, 1}, BCs::AbstractArray{String, 1})
    if (BCs[1] == "N") & (BCs[2] == "N")
        return BPX_NN_half_matrix(L, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "N")
        return BPX_DN_half_matrix(L, ψ, χ)
    elseif (BCs[1] == "N") & (BCs[2] == "D")
        return BPX_ND_half_matrix(L, ψ, χ)
    elseif (BCs[1] == "D") & (BCs[2] == "D")
        return BPX_DD_half_matrix(L, ψ, χ)
    end
end

function BPX_weights(L::Integer, l::Integer)
    ψ = Array{Float64, 1}(undef, 2^(L-l))
    χ = Array{Float64, 1}(undef, 2^(L-l))
    ψ .= range(1, 2^(L-l), step=1)/2^(L-l)
    χ .= 1 .- ψ
    return ψ, χ
end

function BPX_weights(L::Integer, l::Integer, BCs::AbstractArray{String, 1})
    ψ = Array{Float64, 1}(undef, 2^(L-l))
    χ = Array{Float64, 1}(undef, 2^(L-l))
    ψ .= range(1, 2^(L-l), step=1)/2^(L-l)
    if (BCs[1] == "N") & (BCs[2] == "D")
        ψ = reverse(ψ)
    end
    χ .= 1 .- ψ
    return ψ, χ
end

function BPX_DD_full(L::Integer)
    ind = []
    c = 0
    for l=1:(L-1)
        vec_size = 2^(L-l)
        push!(ind, [c+1, c+vec_size])
        push!(ind, [c+vec_size+1, c+2*vec_size])
        push!(ind, [c+2*vec_size+1, c+2*vec_size+1])
        c += 2*vec_size+1
    end
    parameters = Array{Float64, 1}(undef, 2^(L+1)+L-5+1)
    l = 1
    for k=1:3:length(ind)
        ψ, χ = BPX_weights(L, l)
        parameters[ind[k][1]:ind[k][2]] .= ψ
        parameters[ind[k+1][1]:ind[k+1][2]] .= χ
        parameters[ind[k+2][1]:ind[k+2][2]] .= 1
        l += 1
    end
    parameters[end] = 1
    B = parameters[end]*I + sum([parameters[ind[k+2][1]:ind[k+2][2]].*BPX_DD_matrix(L, parameters[ind[k][1]:ind[k][2]], parameters[ind[k+1][1]:ind[k+1][2]]) for k=1:3:length(ind)])
end

function BPX_matrix(L::Integer, BCs::AbstractArray{String, 1})
    ind = []
    c = 0
    for l=1:(L-1)
        vec_size = 2^(L-l)
        push!(ind, [c+1, c+vec_size])
        push!(ind, [c+vec_size+1, c+2*vec_size])
        push!(ind, [c+2*vec_size+1, c+2*vec_size+1])
        c += 2*vec_size+1
    end
    parameters = Array{Float64, 1}(undef, 2^(L+1)+L-5+1)
    l = 1
    for k=1:3:length(ind)
        ψ, χ = BPX_weights(L, l, BCs)
        parameters[ind[k][1]:ind[k][2]] .= ψ
        parameters[ind[k+1][1]:ind[k+1][2]] .= χ
        parameters[ind[k+2][1]:ind[k+2][2]] .= 1
        l += 1
    end
    parameters[end] = 1
    B = parameters[end]*I + sum([parameters[ind[k+2][1]:ind[k+2][2]].*BPX_layer_matrix(L, parameters[ind[k][1]:ind[k][2]], parameters[ind[k+1][1]:ind[k+1][2]], BCs) for k=1:3:length(ind)])
end

function Schwarz_matrix(A::SparseMatrixCSC{Float64, Int64}, L::Integer, BCs::AbstractArray{String, 1})
    B = spdiagm(0 => zeros(size(A)[1]))
    for l=1:(L-1)
        ψ, χ = BPX_weights(L, l, BCs)
        P = BPX_layer_half_matrix(L, ψ, χ, BCs)
        if l == 1
            A_P = sparse(inv(Array(transpose(P)*A*P) + 1e-6*I)) # spdiagm(0 => diag(transpose(P)*A*P).^-1)
        else
            A_P = spdiagm(0 => diag(transpose(P)*A*P).^-1)
        end
        B += P*A_P*transpose(P)
    end
    B += spdiagm(0 => diag(A).^-1)
    return B
end

using Gadfly, Cairo, Fontconfig, DelimitedFiles

function test_Laplace(BCs::AbstractArray{String, 1}, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> -one(x), x -> zero(x), x -> zero(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = Int64((BCs[1] == "N")*(BCs[2] == "N"))
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    # p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    # push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    # push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    # push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    # push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    # results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    # writedlm("precs_comparison/Laplace_"*BCs[1]*BCs[2]*".txt", results)
    # draw(SVG("precs_comparison/Laplace_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function test_Helmholtz(BCs::AbstractArray{String, 1}, k::Real, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> -one(x), x -> zero(x), x -> -k^2*one(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = 0 # Int64((BCs[1] == "N")*(BCs[2] == "N"))
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    writedlm("precs_comparison/Helmholtz_"*string(k)*"_"*BCs[1]*BCs[2]*".txt", results)
    draw(SVG("precs_comparison/Helmholtz_"*string(k)*"_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function test_Convection_Diffusion(BCs::AbstractArray{String, 1}, v::Real, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> -one(x), x -> -v*one(x), x -> zero(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = Int64((BCs[1] == "N")*(BCs[2] == "N"))
        if (BCs[1] == "D")*(BCs[2] == "N")
            shift = 1
        end
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    writedlm("precs_comparison/convection_diffusion_"*string(v)*"_"*BCs[1]*BCs[2]*".txt", results)
    draw(SVG("precs_comparison/convection_diffusion_"*string(v)*"_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function test_Disc_Diffusion(BCs::AbstractArray{String, 1}, σ::Real, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> -one(x)-σ*(x>0.5), x -> zero(x), x -> zero(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = Int64((BCs[1] == "N")*(BCs[2] == "N"))
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    writedlm("precs_comparison/disc_diffusion"*string(σ)*"_"*BCs[1]*BCs[2]*".txt", results)
    draw(SVG("precs_comparison/disc_diffusion"*string(σ)*"_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function test_variable_Convection_Diffusion(BCs::AbstractArray{String, 1}, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> -one(x)-cos(3*pi*x)^2, x -> sin(3*pi*x)^2, x -> zero(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = Int64((BCs[1] == "N")*(BCs[2] == "N"))
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    writedlm("precs_comparison/variable_conv_diffusion_"*BCs[1]*BCs[2]*".txt", results)
    draw(SVG("precs_comparison/variable_conv_diffusion_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function test_implicit_Diffusion(BCs::AbstractArray{String, 1}, μ::Real, verbose=false)
    Ls = [5, 6, 7, 8, 9, 10, 11]
    CondA, CondAS, CondAL, CondSz = [], [], [], []
    for L = Ls
        F = [x -> μ*one(x), x -> zero(x), x -> zero(x), x -> x^2]
        BCs_ = [[1, BCs[1]], [2, BCs[2]]]
        A, _, _ = BVP_1D(L, F, BCs_)
        A = I - A
        B = BPX_matrix(L, BCs)
        Sz = Schwarz_matrix(A, L, BCs)
        A_L, A_S, A_Sz = Array(B*A), Array(B*A*B), Array(Sz*A)
        singular_values_A = svd(Array(A)).S
        values_A_L = sort(abs.(eigvals(A_L)))
        values_A_S = sort(abs.(eigvals(A_S)))
        values_A_Sz = sort(abs.(eigvals(A_Sz)))
        shift = Int64((BCs[1] == "N")*(BCs[2] == "N"))
        push!(CondA, singular_values_A[1]/singular_values_A[end-shift])
        push!(CondAL, values_A_L[end]/values_A_L[1+shift])
        push!(CondAS, values_A_S[end]/values_A_S[1+shift])
        push!(CondSz, values_A_Sz[end]/values_A_Sz[1+shift])
        if verbose
            println("L = ", L)
            println("Condition number of A = ", CondA[end])
            println("Condition number of A_L = ", CondAL[end])
            println("Condition number of A_S = ", CondAS[end])
            println("Condition number of A_Sz = ", CondSz[end])
            println("=======================")
        end
    end
    p = plot(Scale.y_log10, Guide.manual_color_key("", ["A", "left BPX", "symmetric BPX", "Schwarz"], ["red", "black", "blue", "green"]), Guide.xlabel("level"), Guide.ylabel("condition number"))
    push!(p, layer(x=Ls, y=CondA, Geom.line, Geom.point, color=[colorant"red"]))
    push!(p, layer(x=Ls, y=CondAL, Geom.line, Geom.point, color=[colorant"black"]))
    push!(p, layer(x=Ls, y=CondAS, Geom.line, Geom.point, color=[colorant"blue"]))
    push!(p, layer(x=Ls, y=CondSz, Geom.line, Geom.point, color=[colorant"green"]))
    results = hcat(Ls, CondA, CondAS, CondAL, CondSz)
    writedlm("precs_comparison/implicit_diffusion"*string(μ)*"_"*BCs[1]*BCs[2]*".txt", results)
    draw(SVG("precs_comparison/implicit_diffusion"*string(μ)*"_"*BCs[1]*BCs[2]*".svg", 6inch, 4inch), p)
    return nothing
end

function tests()
    BCS = [["D", "D"], ["D", "N"], ["N", "D"], ["N", "N"]]
    functions = [test_Laplace, x -> test_Helmholtz(x, 10), x -> test_Helmholtz(x, 100), x -> test_Convection_Diffusion(x, 100), x -> test_Convection_Diffusion(x, 1000), x -> test_Disc_Diffusion(x, 10), x -> test_Disc_Diffusion(x, 1000), test_variable_Convection_Diffusion, x -> test_implicit_Diffusion(x, 0.01), x -> test_implicit_Diffusion(x, 100)]
    for f=functions
        for BCs = BCS
            f(BCs)
        end
    end
end
