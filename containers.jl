using SparseArrays

struct optimization_parameters
    equation
    optim
    eq_params::Array{<:Real, 1}
    k_sweeps::Integer
    k_batch::Integer
    epochs::Integer
    BCs::Array{String, 1}
    L::Integer
    name::String
    symmetry::String
    error::String
end

struct preconditioner
    A::SparseMatrixCSC{<:Real, <:Integer}
    Ind::Array{Any,1}
    BCs::Array{String, 1}
    L::Integer
    name::String
    symmetry::String
end

struct T_preconditioner
    A::SparseMatrixCSC{<:Real, <:Integer}
    C::Array{Array{SparseMatrixCSC{Float64,Int64},1},1}
    Ind::Array{Any,1}
    BCs::Array{String, 1}
    L::Integer
    name::String
    symmetry::String
end
