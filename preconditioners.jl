include("BPX1D.jl")
include("BPX2D.jl")
include("containers.jl")
include("Losses.jl")
using SparseArrays

function transform_T(T_prec::T_preconditioner)
    prec = preconditioner(T_prec.A, T_prec.Ind, T_prec.BCs, T_prec.L, T_prec.name, T_prec.symmetry)
end

function assemble_matrix(parameters::AbstractArray{<:Real, 1}, prec::preconditioner)
    B = nothing
    if prec.name == "full BPX"
        B = I + sum([parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]*BPX_layer_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs) for l=1:(prec.L-1)])
    elseif prec.name == "symmetric BPX"
        B = I + sum([parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]][1]*BPX_layer_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs) for l=1:(prec.L-1)])
    elseif prec.name == "reduced BPX"
        B = I + sum([parameters[2^prec.L+l]*BPX_layer_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs) for l=1:(prec.L-1)])
    elseif prec.name == "reduced+symmetric BPX"
        B = I + sum([parameters[2^(prec.L-1)+l]*BPX_layer_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs) for l=1:(prec.L-1)])
    elseif prec.name == "full Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        P = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[1][1][1]:prec.Ind[1][1][2]], parameters[prec.Ind[1][2][1]:prec.Ind[1][2][2]], prec.BCs)
        A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
        B += parameters[prec.Ind[1][3][1]:prec.Ind[1][3][2]][1]*P*A_P*transpose(P)
        for l=2:(prec.L-1)
            P = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs)
            A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            B += parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]*P*A_P*transpose(P)
        end
    elseif prec.name == "symmetric Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        P = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[1][1][1]:prec.Ind[1][1][2]], vcat(reverse(parameters[prec.Ind[1][1][1]:prec.Ind[1][1][2]])[2:end], 0), prec.BCs)
        A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
        B += parameters[prec.Ind[1][2][1]:prec.Ind[1][2][2]][1]*P*A_P*transpose(P)
        for l=2:(prec.L-1)
            P = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs)
            A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            B += parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]][1]*P*A_P*transpose(P)
        end
    elseif prec.name == "reduced Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        P = BPX_layer_half_matrix(prec.L, parameters[1:(2^(prec.L-1))], parameters[(2^(prec.L-1)+1):(2^prec.L)], prec.BCs)
        A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
        B += parameters[2^prec.L+1]*P*A_P*transpose(P)
        for l=2:(prec.L-1)
            P = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs)
            A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            B += parameters[2^prec.L+l]*P*A_P*transpose(P)
        end
    elseif prec.name == "reduced+symmetric Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        P = BPX_layer_half_matrix(prec.L, parameters[1:(2^(prec.L-1))], vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0), prec.BCs)
        A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
        B += parameters[2^(prec.L-1)+1]*P*A_P*transpose(P)
        for l=2:(prec.L-1)
            P = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs)
            A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            B += parameters[2^(prec.L-1)+1]*P*A_P*transpose(P)
        end
    elseif prec.name == "2D full BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]], parameters[prec.Ind[l][4][1]:prec.Ind[l][4][2]], prec.BCs[3:4])
            P = kron(P_y, P_x)
            B += (P*transpose(P))*parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]][1]
        end
    elseif prec.name == "2D symmetric BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], vcat(reverse(parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]])[2:end], 0), prec.BCs[3:4])
            P = kron(P_y, P_x)
            B += (P*transpose(P))*parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]
        end
    elseif prec.name == "2D reduced BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L)+1):(3*2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(3*2^(prec.L-1)+1):(2^(prec.L+1))])[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            B += (P*transpose(P))*parameters[2^(prec.L+1)+l]
        end
    elseif prec.name == "2D reduced+symmetric BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            B += (P*transpose(P))*parameters[2^(prec.L)+l]
        end
    elseif prec.name == "2D full Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]], parameters[prec.Ind[l][4][1]:prec.Ind[l][4][2]], prec.BCs[3:4])
            P = kron(P_y, P_x)
            if l == 1
                A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
            else
                A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            end
            B += P*A_P*transpose(P)*parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]][1]
        end
    elseif prec.name == "2D symmetric Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], vcat(reverse(parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]])[2:end], 0), prec.BCs[3:4])
            P = kron(P_y, P_x)
            if l == 1
                A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
            else
                A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            end
            B += P*A_P*transpose(P)*parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]
        end
    elseif prec.name == "2D reduced Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L)+1):(3*2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(3*2^(prec.L-1)+1):(2^(prec.L+1))])[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            if l == 1
                A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
            else
                A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            end
            B += P*A_P*transpose(P)*parameters[2^(prec.L+1)+l]
        end
    elseif prec.name == "2D reduced+symmetric Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            if l == 1
                A_P = sparse(inv(Array(transpose(P)*prec.A*P) + 1e-6*I))
            else
                A_P = spdiagm(0 => diag(transpose(P)*prec.A*P).^-1)
            end
            B += P*A_P*transpose(P)*parameters[2^(prec.L)+l]
        end
    elseif prec.name == "2D tridiagonal BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            N_x, N_y = size(P_x)[2], size(P_y)[2]
            T_x = spdiagm(0 => parameters[2^(prec.L)+1+6*(l-1)]*ones(N_x), 1 => parameters[2^(prec.L)+2+6*(l-1)]*ones(N_x-1), -1 => parameters[2^(prec.L)+3+6*(l-1)]*ones(N_x-1))
            T_y = spdiagm(0 => parameters[2^(prec.L)+4+6*(l-1)]*ones(N_y), 1 => parameters[2^(prec.L)+5+6*(l-1)]*ones(N_y-1), -1 => parameters[2^(prec.L)+6+6*(l-1)]*ones(N_y-1))
            P = kron(P_y, P_x)*kron(T_y, T_x)
            # P = kron(P_y, P_x)*(kron(spdiagm(0 => ones(N_y)), T_x) + kron(T_y, spdiagm(0 => ones(N_x))))
            B += parameters[6*(prec.L-1)+2^(prec.L)+l]*P*transpose(P)
        end
    elseif prec.name == "2D double Schwarz"
        B = spdiagm(0 => diag(prec.A).^-1)
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P_x_ = BPX_layer_half_matrix(prec.L, (parameters[(2^prec.L+1):(2^prec.L+2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^prec.L+1):(2^prec.L+2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y_ = BPX_layer_half_matrix(prec.L, (parameters[(2^prec.L+2^(prec.L-1)+1):(2^prec.L+2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^prec.L+2^(prec.L-1)+1):(2^prec.L+2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            P_ = kron(P_y_, P_x_)
            if l == 1
                A_P = sparse(inv(Array(transpose(P_)*prec.A*P_) + 1e-6*I))
            else
                A_P = spdiagm(0 => diag(transpose(P_)*prec.A*P_).^-1)
            end
            B += P*A_P*transpose(P)*parameters[2^(prec.L+1)+l]
        end
    elseif prec.name == "2D reduced+symmetric+rescaled BPX"
        B = spdiagm(0 => diag(prec.A).^(-1/2))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P = kron(P_y, P_x)
            ψ, χ = BPX_weights(prec.L, l)
            P_x_ = BPX_layer_half_matrix(prec.L, ψ, χ, prec.BCs[1:2])
            P_y_ = BPX_layer_half_matrix(prec.L, ψ, χ, prec.BCs[3:4])
            P_ = kron(P_y_, P_x_)
            A_P = spdiagm(0 => diag(transpose(P_)*prec.A*P_).^(-1/2))
            B += P*A_P*transpose(P)*parameters[2^(prec.L)+l]
        end
    elseif prec.name == "2D reduced+symmetric+semi BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        shift = 1
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            l_ = min(l + shift, (prec.L - 1))
            if l_ == (prec.L - 1)
                P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l_-1):2^(l_-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l_-1):2^(l_-1):end], prec.BCs[3:4])
                P_y = spdiagm(0 => ones(size(P_y)[1]))
            else
                P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l_-1):2^(l_-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l_-1):2^(l_-1):end], prec.BCs[3:4])
            end
            P = kron(P_y, P_x)
            B += (P*transpose(P))*parameters[2^(prec.L)+l]
        end
    end
    return B
end

function assemble_matrix(parameters::AbstractArray{<:Real, 1}, T_prec::T_preconditioner)
    assemble_matrix(parameters, transform_T(T_prec))
end

function apply_T_preconditioner(X::AbstractArray{<:Real, 2}, parameters::AbstractArray{<:Real, 1}, prec::T_preconditioner)
    Y = X .+ 0.0
    if prec.name == "2D full BPX"
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]], parameters[prec.Ind[l][4][1]:prec.Ind[l][4][2]], prec.BCs[3:4])
            Y += parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]][1]*(((P_y*(transpose(P_y)*X))*P_x)*transpose(P_x))
        end
    elseif prec.name == "2D symmetric BPX"
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], vcat(reverse(parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]])[2:end], 0), prec.BCs[3:4])
            Y += parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]*((P_y*((transpose(P_y)*X)*P_x))*transpose(P_x))
        end
    elseif prec.name == "2D reduced BPX"
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L)+1):(3*2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(3*2^(prec.L-1)+1):(2^(prec.L+1))])[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            Y += parameters[2^(prec.L+1)+l]*((P_y*((transpose(P_y)*X)*P_x))*transpose(P_x))
        end
    elseif prec.name == "2D reduced+symmetric BPX"
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            Y += parameters[2^(prec.L)+l]*((P_y*((transpose(P_y)*X)*P_x))*transpose(P_x))
        end
    elseif prec.name == "2D full Schwarz"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]], parameters[prec.Ind[l][4][1]:prec.Ind[l][4][2]], prec.BCs[3:4])
            if l == 1
                A_inv = inv(Array(sum([kron((transpose(P_y)*c[1])*P_y, (transpose(P_x)*c[2])*P_x) for c = prec.C]) + 1e-6*I))
                y = ((transpose(P_y)*X)*P_x)
                Y += ((P_y*transpose(reshape(A_inv*reshape(transpose(y), (size(A_inv)[1], 1)), (size(y)[2], size(y)[1]))))*transpose(P_x))*parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]][1]
            else
                D_inv = 1 ./ sum([(diag(transpose(P_y)*c[1]*P_y))*transpose(diag(transpose(P_x)*c[2]*P_x)) for c = prec.C])
                Y += ((P_y*(((transpose(P_y)*X)*P_x) .* D_inv))*transpose(P_x))*parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]][1]
            end
        end
    elseif prec.name == "2D symmetric Schwarz"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]], vcat(reverse(parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]])[2:end], 0), prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]], vcat(reverse(parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]])[2:end], 0), prec.BCs[3:4])
            if l == 1
                A_inv = inv(Array(sum([kron((transpose(P_y)*c[1])*P_y, (transpose(P_x)*c[2])*P_x) for c = prec.C]) + 1e-6*I))
                y = ((transpose(P_y)*X)*P_x)
                Y += ((P_y*transpose(reshape(A_inv*reshape(transpose(y), (size(A_inv)[1], 1)), (size(y)[2], size(y)[1]))))*transpose(P_x))*parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]
            else
                D_inv = 1 ./ sum([(diag(transpose(P_y)*c[1]*P_y))*transpose(diag(transpose(P_x)*c[2]*P_x)) for c = prec.C])
                Y += ((P_y*(((transpose(P_y)*X)*P_x) .* D_inv))*transpose(P_x))*parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]][1]
            end
        end
    elseif prec.name == "2D reduced Schwarz"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L)+1):(3*2^(prec.L-1))])[2^(l-1):2^(l-1):end], (parameters[(3*2^(prec.L-1)+1):(2^(prec.L+1))])[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            if l == 1
                A_inv = inv(Array(sum([kron((transpose(P_y)*c[1])*P_y, (transpose(P_x)*c[2])*P_x) for c = prec.C]) + 1e-6*I))
                y = ((transpose(P_y)*X)*P_x)
                Y += ((P_y*transpose(reshape(A_inv*reshape(transpose(y), (size(A_inv)[1], 1)), (size(y)[2], size(y)[1]))))*transpose(P_x))*parameters[2^(prec.L+1)+l]
            else
                D_inv = 1 ./ sum([(diag(transpose(P_y)*c[1]*P_y))*transpose(diag(transpose(P_x)*c[2]*P_x)) for c = prec.C])
                Y += ((P_y*(((transpose(P_y)*X)*P_x) .* D_inv))*transpose(P_x))*parameters[2^(prec.L+1)+l]
            end
        end
    elseif prec.name == "2D reduced+symmetric Schwarz"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            if l == 1
                A_inv = inv(Array(sum([kron((transpose(P_y)*c[1])*P_y, (transpose(P_x)*c[2])*P_x) for c = prec.C]) + 1e-6*I))
                y = ((transpose(P_y)*X)*P_x)
                Y += ((P_y*transpose(reshape(A_inv*reshape(transpose(y), (size(A_inv)[1], 1)), (size(y)[2], size(y)[1]))))*transpose(P_x))*parameters[2^(prec.L)+l]
            else
                D_inv = 1 ./ sum([(diag(transpose(P_y)*c[1]*P_y))*transpose(diag(transpose(P_x)*c[2]*P_x)) for c = prec.C])
                Y += ((P_y*(((transpose(P_y)*X)*P_x) .* D_inv))*transpose(P_x))*parameters[2^(prec.L)+l]
            end
        end
    elseif prec.name == "2D double Schwarz"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            P_x_ = BPX_layer_half_matrix(prec.L, (parameters[(2^prec.L+1):(2^prec.L+2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^prec.L+1):(2^prec.L+2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y_ = BPX_layer_half_matrix(prec.L, (parameters[(2^prec.L+2^(prec.L-1)+1):(2^prec.L+2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^prec.L+2^(prec.L-1)+1):(2^prec.L+2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            if l == 1
                A_inv = inv(Array(sum([kron((transpose(P_y_)*c[1])*P_y_, (transpose(P_x_)*c[2])*P_x_) for c = prec.C]) + 1e-6*I))
                y = ((transpose(P_y)*X)*P_x)
                Y += ((P_y*transpose(reshape(A_inv*reshape(transpose(y), (size(A_inv)[1], 1)), (size(y)[2], size(y)[1]))))*transpose(P_x))*parameters[2^(prec.L)+l]
            else
                D_inv = 1 ./ sum([(diag(transpose(P_y_)*c[1]*P_y_))*transpose(diag(transpose(P_x_)*c[2]*P_x_)) for c = prec.C])
                Y += ((P_y*(((transpose(P_y)*X)*P_x) .* D_inv))*transpose(P_x))*parameters[2^prec.L+2^(prec.L)+l]
            end
        end
    elseif prec.name == "2D tridiagonal BPX"
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            N_x, N_y = size(P_x)[2], size(P_y)[2]
            T_x = spdiagm(0 => parameters[2^(prec.L)+1+6*(l-1)]*ones(N_x), 1 => parameters[2^(prec.L)+2+6*(l-1)]*ones(N_x-1), -1 => parameters[2^(prec.L)+3+6*(l-1)]*ones(N_x-1))
            T_y = spdiagm(0 => parameters[2^(prec.L)+4+6*(l-1)]*ones(N_y), 1 => parameters[2^(prec.L)+5+6*(l-1)]*ones(N_y-1), -1 => parameters[2^(prec.L)+6+6*(l-1)]*ones(N_y-1))
            P = kron(P_y, P_x)*(kron(spdiagm(0 => ones(N_y)), T_x) + kron(T_y, spdiagm(0 => ones(N_x))))
            Y += parameters[6*(prec.L-1)+2^(prec.L)+l]*(P_y*(T_y*(transpose(T_y)*(((transpose(P_y)*(X*P_x))*T_x)*transpose(T_x)))))*transpose(P_x)
            # z = transpose(P_y)*(X*P_x)
            # Y += parameters[6*(prec.L-1)+2^(prec.L)+l]*((P_y*((z*T_x)*transpose(T_x) + T_y*(transpose(T_y)*z) + (transpose(T_y)*z)*transpose(T_x) + (T_y*z)*T_x))*transpose(P_x))
        end
    elseif prec.name == "2D reduced+symmetric+rescaled BPX"
        Y ./= transpose(reshape(diag(prec.A), (size(X)[2], size(X)[1])))
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[3:4])
            ψ, χ = BPX_weights(prec.L, l)
            P_x_ = BPX_layer_half_matrix(prec.L, ψ, χ, prec.BCs[1:2])
            P_y_ = BPX_layer_half_matrix(prec.L, ψ, χ, prec.BCs[3:4])
            D_inv = 1 ./ sum([(diag(transpose(P_y_)*c[1]*P_y_))*transpose(diag(transpose(P_x_)*c[2]*P_x_)) for c = prec.C])
            Y += ((P_y*(((transpose(P_y)*X)*P_x) .* sqrt.(D_inv)))*transpose(P_x))*parameters[2^(prec.L)+l]
        end
    elseif prec.name == "2D reduced+symmetric+semi BPX"
        B = spdiagm(0 => ones(size(prec.A)[1]))
        shift = 1
        for l=1:(prec.L-1)
            P_x = BPX_layer_half_matrix(prec.L, (parameters[1:(2^(prec.L-1))])[2^(l-1):2^(l-1):end], (vcat(reverse(parameters[1:(2^(prec.L-1))])[2:end], 0))[2^(l-1):2^(l-1):end], prec.BCs[1:2])
            l_ = min(l + shift, (prec.L - 1))
            if l_ == (prec.L - 1)
                P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l_-1):2^(l_-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l_-1):2^(l_-1):end], prec.BCs[3:4])
                P_y = spdiagm(0 => ones(size(P_y)[1]))
            else
                P_y = BPX_layer_half_matrix(prec.L, (parameters[(2^(prec.L-1)+1):(2^prec.L)])[2^(l_-1):2^(l_-1):end], (vcat(reverse(parameters[(2^(prec.L-1)+1):(2^prec.L)])[2:end], 0))[2^(l_-1):2^(l_-1):end], prec.BCs[3:4])
            end
            Y += parameters[2^(prec.L)+l]*((P_y*((transpose(P_y)*X)*P_x))*transpose(P_x))
        end
    end
    return Y
end

function indices(BCs::Array{String, 1}, L::Integer, name::String)
    ind = []
    c = 0
    if name[1:4] == "full"
        for l=1:(L-1)
            vec_size = 2^(L-l)
            push!(ind, [[c+1, c+vec_size], [c+vec_size+1, c+2*vec_size], [c+2*vec_size+1, c+2*vec_size+1]])
            c += 2*vec_size+1
        end
    elseif name[1:3] == "sym"
        for l=1:(L-1)
            vec_size = 2^(L-l)
            push!(ind, [[c+1, c+vec_size], [c+vec_size+1, c+vec_size+1]])
            c += vec_size+1
        end
    elseif name[1:7] == "2D full"
        for l=1:(L-1)
            vec_size = 2^(L-l)
            push!(ind, [[c+1, c+vec_size], [c+vec_size+1, c+2*vec_size], [c+2*vec_size+1, c+3*vec_size], [c+3*vec_size+1, c+4*vec_size], [c+4*vec_size+1, c+4*vec_size+1]])
            c += 4*vec_size+1
        end
    elseif name[1:6] == "2D sym"
        for l=1:(L-1)
            vec_size = 2^(L-l)
            push!(ind, [[c+1, c+vec_size], [c+vec_size+1, c+2*vec_size], [c+2*vec_size+1, c+2*vec_size+1]])
            c += 2*vec_size+1
        end
    end
    return ind
end

function default_parameters(prec::preconditioner, ω::Real)
    if prec.name[1:4] == "full"
        parameters = Array{Float64, 1}(undef, 2^(prec.L+1)+prec.L-5+1)
        for l=1:(prec.L-1)
            N_ = 2^l+1
            h_ = 1/(N_-1)
            ψ, χ = BPX_weights(prec.L, l, prec.BCs)
            parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]] .= ψ
            parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]] .= χ
            parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]] .= 1
        end
    elseif prec.name[1:3] == "sym"
        parameters = Array{Float64, 1}(undef, 2^prec.L+prec.L-3+1)
        for l=1:(prec.L-1)
            ψ, _ = BPX_weights(prec.L, l, prec.BCs)
            parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]] .= ψ
            parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]] .= 1
        end
    elseif prec.name[1:8] == "reduced "
        parameters = Array{Float64, 1}(undef, 2*2^(prec.L-1)+prec.L-1+1); parameters .= 1
        ψ, χ  = BPX_weights(prec.L, 1, prec.BCs)
        parameters[1:2^(prec.L-1)] .= ψ
        parameters[(2^(prec.L-1)+1):2^(prec.L)] .= χ
    elseif prec.name[1:min(17, length(prec.name))] == "reduced+symmetric"
        parameters = Array{Float64, 1}(undef, 2^(prec.L-1)+prec.L-1+1); parameters .= 1
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs)
        parameters[1:2^(prec.L-1)] .= ψ
    elseif prec.name[1:7] == "2D full"
        parameters = Array{Float64, 1}(undef, 2^(prec.L+2)+prec.L-8)
        for l=1:(prec.L-1)
            ψ, χ = BPX_weights(prec.L, l, prec.BCs[1:2])
            parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]] .= ψ
            parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]] .= χ
            ψ, χ = BPX_weights(prec.L, l, prec.BCs[3:4])
            parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]] .= ψ
            parameters[prec.Ind[l][4][1]:prec.Ind[l][4][2]] .= χ
            if (prec.name[(end-2):end] == "BPX")
                parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]] .= 1/2^prec.L
            else
                parameters[prec.Ind[l][5][1]:prec.Ind[l][5][2]] .= 1
            end
        end
    elseif prec.name[1:6] == "2D sym"
        parameters = Array{Float64, 1}(undef, 2^(prec.L+1)+prec.L-4)
        for l=1:(prec.L-1)
            ψ, _ = BPX_weights(prec.L, l, prec.BCs[1:2])
            parameters[prec.Ind[l][1][1]:prec.Ind[l][1][2]] .= ψ
            ψ, _ = BPX_weights(prec.L, l, prec.BCs[3:4])
            parameters[prec.Ind[l][2][1]:prec.Ind[l][2][2]] .= ψ
            if (prec.name[(end-2):end] == "BPX")
                parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]] .= 1/2^prec.L
            else
                parameters[prec.Ind[l][3][1]:prec.Ind[l][3][2]] .= 1
            end
        end
    elseif prec.name[1:11] == "2D reduced "
        parameters = Array{Float64, 1}(undef, 2^(prec.L+1)+prec.L-1+1)
        if (prec.name[(end-2):end] == "BPX")
            parameters .=  1/2^prec.L
        else
            parameters .=  1
        end
        ψ, χ  = BPX_weights(prec.L, 1, prec.BCs[1:2])
        parameters[1:2^(prec.L-1)] .= ψ
        parameters[(2^(prec.L-1)+1):2^(prec.L)] .= χ
        ψ, χ  = BPX_weights(prec.L, 1, prec.BCs[3:4])
        parameters[2^(prec.L)+1:(3*2^(prec.L-1))] .= ψ
        parameters[(3*2^(prec.L-1)+1):2^(prec.L+1)] .= χ
    elseif prec.name[1:min(20, length(prec.name))] == "2D reduced+symmetric"
        parameters = Array{Float64, 1}(undef, 2^prec.L+prec.L-1+1)
        if (prec.name[(end-2):end] == "BPX")
            for l = 1:(prec.L-1)
                N = 2^prec.L+1
                h = 1/(N-1)
                N_ = 2^(prec.L-l)+1
                h_ = 1/(N_-1)
                parameters[end-l] = (h/h_)
            end
        else
            parameters .=  1
        end
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[1:2])
        parameters[1:2^(prec.L-1)] .= ψ
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[3:4])
        parameters[(2^(prec.L-1)+1):2^(prec.L)] .= ψ
    elseif prec.name[1:14] == "2D tridiagonal"
        parameters = Array{Float64, 1}(undef, 6*(prec.L-1)+2^(prec.L)+prec.L)
        parameters .=  1/2^prec.L
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[1:2])
        parameters[1:2^(prec.L-1)] .= ψ
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[3:4])
        parameters[(2^(prec.L-1)+1):2^(prec.L)] .= ψ
        parameters[(2^(prec.L)+1):6:(6*(prec.L-1)+2^(prec.L))] .= 1
        parameters[(2^(prec.L)+2):6:(6*(prec.L-1)+2^(prec.L))] .= 1e-4
        parameters[(2^(prec.L)+3):6:(6*(prec.L-1)+2^(prec.L))] .= 1e-4
        parameters[(2^(prec.L)+4):6:(6*(prec.L-1)+2^(prec.L))] .= 1
        parameters[(2^(prec.L)+5):6:(6*(prec.L-1)+2^(prec.L))] .= 1e-4
        parameters[(2^(prec.L)+6):6:(6*(prec.L-1)+2^(prec.L))] .= 1e-4
    elseif prec.name == "2D double Schwarz"
        parameters = Array{Float64, 1}(undef, 2^(prec.L+1)+prec.L)
        parameters .=  1
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[1:2])
        parameters[1:2^(prec.L-1)] .= ψ
        parameters[(1+2^(prec.L)):(2^(prec.L)+2^(prec.L-1))] .= ψ
        ψ, _  = BPX_weights(prec.L, 1, prec.BCs[3:4])
        parameters[(2^(prec.L-1)+1):2^(prec.L)] .= ψ
        parameters[(2^(prec.L)+2^(prec.L-1)+1):2^(prec.L+1)] .= ψ
    end
    parameters[end] = ω
    return parameters
end

function default_parameters(T_prec::T_preconditioner, ω::Real)
    default_parameters(transform_T(T_prec), ω)
end

function matching_preconditioner(prec::preconditioner)
    if (prec.name[1:2] != "2D")
        if (prec.name[(end-2):end] == "BPX")
            B = BPX_matrix(prec.L, prec.BCs)
        elseif prec.name[(end-6):end] == "Schwarz"
            B = Schwarz_matrix(prec.A, prec.L, prec.BCs)
        end
    else
        if (prec.name[(end-2):end] == "BPX")
            B = BPX_matrix_2D(prec.L, prec.BCs)
        elseif prec.name[(end-6):end] == "Schwarz"
            B = Schwarz_matrix_2D(prec.A, prec.L, prec.BCs)
        end
    end
    return B
end

function matching_preconditioner(T_prec::T_preconditioner)
    matching_preconditioner(transform_T(T_prec))
end

function initialise_preconditioner(A::SparseMatrixCSC{<:Real, <:Integer}, L::Integer, BCs::Array{String, 1}, name::String, symmetry::String)
    Ind = indices(BCs, L, name)
    prec = preconditioner(A, Ind, BCs, L, name, symmetry)
    parameters = default_parameters(prec, 1.0)
    B = assemble_matrix(parameters, prec)
    A_prec(x) = B*A*B*x
    if symmetry == "left"
        A_prec(x) = B*A*x
    end
    ω = 1/ρ_error(size(B)[1], 20, A_prec, 10)
    parameters[end] = ω
    return prec, B, parameters
end

function initialise_preconditioner(C::Array{Array{SparseMatrixCSC{Float64,Int64},1},1}, L::Integer, BCs::Array{String, 1}, name::String, symmetry::String)
    Ind = indices(BCs, L, name)
    A = sum(kron(c[1], c[2]) for c = C)
    prec = T_preconditioner(A, C, Ind, BCs, L, name, symmetry)
    B = matching_preconditioner(prec)
    A_prec(x) = B*A*B*x
    if symmetry == "left"
        A_prec(x) = B*A*x
    end
    ω = 1/ρ_error(size(B)[1], 20, A_prec, 10)
    parameters = default_parameters(prec, ω)
    return prec, B, parameters
end

function power_method(f::Function, D::Integer, N_sweeps::Integer)
    x = randn(D)
    α = 1
    for i = 1:N_sweeps
        x = f(x)
        α = maximum(abs.(x))
        x ./= α
    end
    return α
end

function measure_condition_number(parameters::AbstractArray{<:Real, 1}, prec::preconditioner)
    M = assemble_matrix(parameters::AbstractArray{<:Real, 1}, prec::preconditioner)
    if prec.symmetry == "symmetric"
        T = M*prec.A*M
    elseif prec.symmetry == "left"
        T = M*prec.A
    end
    σ = sort(abs.(eigvals(Array(T))))
    return σ[end]/σ[1]
end

function measure_condition_number(parameters::AbstractArray{<:Real, 1}, T_prec::T_preconditioner)
    measure_condition_number(parameters, transform_T(T_prec))
end

function measure_prec_condition_number(parameters::AbstractArray{<:Real, 1}, prec::preconditioner)
    M = matching_preconditioner(prec::preconditioner)
    if prec.symmetry == "symmetric"
        T = M*prec.A*M
    elseif prec.symmetry == "left"
        T = M*prec.A
    end
    σ = sort(abs.(eigvals(Array(T))))
    return σ[end]/σ[1]
end

function measure_prec_condition_number(parameters::AbstractArray{<:Real, 1}, T_prec::T_preconditioner)
    measure_prec_condition_number(parameters, transform_T(T_prec))
end

function measure_s_condition_number(parameters::AbstractArray{<:Real, 1}, prec::preconditioner, params::optimization_parameters)
    ρ = spectral_radius(parameters, prec, params)
    M = assemble_matrix(parameters, prec)
    α = scalar_loss_function([1/ρ, ], M, prec, params)
    return 1/(1-α)
end

function measure_s_condition_number(parameters::AbstractArray{<:Real, 1}, T_prec::T_preconditioner, params::optimization_parameters)
    measure_s_condition_number(parameters, transform_T(T_prec), params)
end

function test_matching_prec(equation::Function, parameters::Array{<:Real, 1}, BCs::Array{String, 1}, L::Integer, name::String, symmetry::String, tol::Real=1e-11)
    A = equation(L, BCs, parameters)
    prec = preconditioner(A, [[[1, 2], [3, 4], [5, 6]]], BCs, L, name, symmetry)
    M = matching_preconditioner(prec)
    if prec.symmetry == "symmetric"
        vals = svdvals(Array(M*prec.A*M))
    elseif prec.symmetry == "left"
        vals = svdvals(Array(M*prec.A))
    end
    prec_condition_number = vals[1]/vals[end-(vals[end]<=tol)]
    vals = svdvals(Array(prec.A))
    condition_number = vals[1]/vals[end-(vals[end]<=tol)]
    return condition_number, prec_condition_number
end

include("equations.jl")
include("tensor_equations.jl")

function test_T_precs(equation::Function, eq_params::Array{<:Real, 1},  BCs::Array{String, 1}, L::Integer, name::String, symmetry::String)
    C = equation(L, BCs, eq_params)
    T_prec, B, parameters = initialise_T_preconditioner(C, L, BCs, name, symmetry)
    N = 2^L+1
    DOF_x, DOF_y = N - 2 + sum(BCs[1:2] .== "N"), N - 2 + sum(BCs[3:4] .== "N")
    x, y = randn(DOF_x*DOF_y, 1), zeros(DOF_x*DOF_y, 1)
    z = transpose(reshape(x, (DOF_x, DOF_y)))
    y = B*x
    z = apply_T_preconditioner(z, parameters, T_prec)
    z = reshape(transpose(z), (DOF_x*DOF_y, 1))
    println(norm(z-y))
end

function test_precs(equation::Function, eq_params::Array{<:Real, 1},  BCs::Array{String, 1}, L::Integer, name::String, symmetry::String)
    A = equation(L, BCs, eq_params)
    prec, B, parameters = initialise_preconditioner(A, L, BCs, name, symmetry)
    B_prec = assemble_matrix(parameters, prec)
    N = 2^L+1
    DOF_x, DOF_y = N - 2 + sum(BCs[1:2] .== "N"), N - 2 + sum(BCs[3:4] .== "N")
    x, y, z = randn(DOF_x*DOF_y, 1), zeros(DOF_x*DOF_y, 1), zeros(DOF_x*DOF_y, 1)
    z = B_prec*x
    y = B*x
    println(norm(z-y))
end

function test()
    # "2D full BPX"
    # "2D reduced BPX"
    # "2D symmetric BPX"
    # "2D reduced+symmetric BPX"
    for BCs = Iterators.product(["D", "N"], ["D", "N"], ["D", "N"], ["D", "N"])
        println(BCs)
        test_T_precs(T_Laplace_2D, [10.0, 10.0, ], [BCs...], 5, "2D full Schwarz", "left")
        #test_precs(Anisotropic_Laplace_2D, [1.0, 1.0, ], [BCs...], 4, "2D full BPX", "left")
        println("=========")
    end
end
