include("BPX1D.jl")

function Schwarz_matrix_2D(A::SparseMatrixCSC{Float64, Int64}, L::Integer, BCs::AbstractArray{String, 1})
    B = spdiagm(0 => diag(A).^-1)
    for l=1:(L-1)
        ψ_x, χ_x = BPX_weights(L, l, BCs[1:2])
        P_x = BPX_layer_half_matrix(L, ψ_x, χ_x, BCs[1:2])
        ψ_y, χ_y = BPX_weights(L, l, BCs[3:4])
        P_y = BPX_layer_half_matrix(L, ψ_y, χ_y, BCs[3:4])
        P = kron(P_y, P_x)
        if l == 1
            A_P = sparse(inv(Array(transpose(P)*A*P + 1e-6*I)))
        else
            A_P = spdiagm(0 => diag(transpose(P)*A*P).^-1)
        end
        B += P*A_P*transpose(P)
    end
    return B
end

function BPX_matrix_2D(L::Integer, BCs::AbstractArray{String, 1})
    N = 2^L+1
    h = 1/(N-1)
    DOF_x, DOF_y = N - 2 + sum(BCs[1:2] .== "N"), N - 2 + sum(BCs[3:4] .== "N")
    B = spdiagm(0 => ones(DOF_x*DOF_y))
    for l=1:(L-1)
        ψ_x, χ_x = BPX_weights(L, l, BCs[1:2])
        P_x = BPX_layer_half_matrix(L, ψ_x, χ_x, BCs[1:2])
        ψ_y, χ_y = BPX_weights(L, l, BCs[3:4])
        P_y = BPX_layer_half_matrix(L, ψ_y, χ_y, BCs[3:4])
        P = kron(P_y, P_x)
        # B += (P*transpose(P))*h
        N_ = 2^l+1
        h_ = 1/(N_-1)
        B += (P*transpose(P))*(h/h_)
    end
    return B
end
