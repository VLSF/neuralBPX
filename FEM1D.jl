using SparseArrays, QuadGK, LinearAlgebra

function tent(x::Real, center::Real, h::Real)
    if center<=x<=(center+h)
        return 1 - (x-center)/h
    elseif (center-h)<=x<center
        return 1 + (x-center)/h
    else
        return 0
    end
end

function d_tent(x::Real, center::Real, h::Real)
    if center<=x<=(center+h)
        return -1/h
    elseif (center-h)<=x<center
        return 1/h
    else
        return 0
    end
end

function BVP_DD(l::Integer, a::Function, c::Function, leftBC::Real, rightBC::Real, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, N-2)
    x = zeros(N)
    for i = 1:(N-2)
        xi = h*i
        for j = [-1, 0, +1]
            if 1<=(i+j)<=N-2
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-10)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-10)[1]
    end
    b[1] -= leftBC*quadgk(x -> (-a(x)*d_tent(x, h, h)*d_tent(x, 0, h) + c(x)*tent(x, h, h)*tent(x, 0, h)), 0, h, rtol=1e-10)[1]
    b[N-2] -= rightBC*quadgk(x -> (-a(x)*d_tent(x, 1-h, h)*d_tent(x, 1, h) + c(x)*tent(x, 1-h, h)*tent(x, 1, h)), 1-h, 1, rtol=1e-10)[1]
    A = sparse(rows, columns, values)
    x[1], x[N] = leftBC, rightBC
    return A, b, x
end

function BVP_DD(l::Integer, a::Function, d::Function, c::Function, leftBC::Real, rightBC::Real, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, N-2)
    x = zeros(N)
    for i = 1:(N-2)
        xi = h*i
        for j = [-1, 0, +1]
            if 1<=(i+j)<=N-2
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-13)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-13)[1]
    end
    b[1] -= leftBC*quadgk(x -> (-a(x)*d_tent(x, h, h)*d_tent(x, 0, h) + d(x)*tent(x, h, h)*d_tent(x, 0, h) + c(x)*tent(x, h, h)*tent(x, 0, h)), 0, h, rtol=1e-13)[1]
    b[N-2] -= rightBC*quadgk(x -> (-a(x)*d_tent(x, 1-h, h)*d_tent(x, 1, h) + d(x)*tent(x, 1-h, h)*d_tent(x, 1, h) + c(x)*tent(x, 1-h, h)*tent(x, 1, h)), 1-h, 1, rtol=1e-13)[1]
    A = sparse(rows, columns, values)
    x[1], x[N] = leftBC, rightBC
    return A, b, x
end

function BVP_ND(l::Integer, a::Function, d::Function, c::Function, leftBC::Real, rightBC::Real, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, N-1)
    x = zeros(N)
    xi = 0
    i = 1
    for j = [0, +1]
        push!(rows, i)
        push!(columns, i+j)
        push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi, xi+h, rtol=1e-13)[1])
    end
    b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi, xi+h, rtol=1e-13)[1]
    for i = 2:(N-1)
        xi = h*(i-1)
        for j = [-1, 0, +1]
            if 1<=(i+j)<=N-1
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-13)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-13)[1]
    end
    b[1] += a(0)*leftBC
    b[N-1] -= rightBC*quadgk(x -> (-a(x)*d_tent(x, 1-h, h)*d_tent(x, 1, h) + d(x)*tent(x, 1-h, h)*d_tent(x, 1, h) + c(x)*tent(x, 1-h, h)*tent(x, 1, h)), 1-h, 1, rtol=1e-13)[1]
    A = sparse(rows, columns, values)
    x[N] = rightBC
    return A, b, x
end

function BVP_DN(l::Integer, a::Function, d::Function, c::Function, leftBC::Real, rightBC::Real, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, N-1)
    x = zeros(N)
    for i = 1:(N-2)
        xi = h*i
        for j = [-1, 0, +1]
            if 1<=(i+j)<=N-1
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-13)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-13)[1]
    end
    xi = 1
    i = N-1
    for j = [-1, 0]
        push!(rows, i)
        push!(columns, i+j)
        push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi, rtol=1e-13)[1])
    end
    b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi, rtol=1e-13)[1]
    b[1] -= leftBC*quadgk(x -> (-a(x)*d_tent(x, h, h)*d_tent(x, 0, h) + d(x)*tent(x, h, h)*d_tent(x, 0, h) + c(x)*tent(x, h, h)*tent(x, 0, h)), 0, h, rtol=1e-13)[1]
    b[N-1] -= a(1)*rightBC
    A = sparse(rows, columns, values)
    x[1] = leftBC
    return A, b, x
end

function BVP_NN(l::Integer, a::Function, d::Function, c::Function, leftBC::Real, rightBC::Real, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, N)
    x = zeros(N)
    xi = 0
    i = 1
    for j = [0, +1]
        push!(rows, i)
        push!(columns, i+j)
        push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi, xi+h, rtol=1e-13)[1])
    end
    b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi, xi+h, rtol=1e-13)[1]
    for i = 2:(N-1)
        xi = h*(i-1)
        for j = [-1, 0, +1]
            if 1<=(i+j)<=N
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-13)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-13)[1]
    end
    xi = 1
    i = N
    for j = [-1, 0]
        push!(rows, i)
        push!(columns, i+j)
        push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi, rtol=1e-13)[1])
    end
    b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi, rtol=1e-13)[1]
    b[1] += a(0)*leftBC
    b[N] -= a(1)*rightBC
    A = sparse(rows, columns, values)
    return A, b, x
end

function BVP_1D(l::Integer, F::Array{Function, 1}, BCs::Array{Array{Any,1},1})
    N = 2^l+1
    h = 1/(N-1)
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    a, d, c, f = F
    (nL, tL), (nR, tR) = BCs
    DOF = N - 2 + (tL == "N") + (tR == "N")
    b = Array{Float64,1}(undef, DOF)
    x = zeros(N)
    if tL == "N"
        first, shift, last = 2, 1, N-1
    else
        first, shift, last = 1, 0, N-2
    end
    for i = first:last
        xi = h*(i-shift)
        for j = [-1, 0, +1]
            if 1<=(i+j)<=DOF
                push!(rows, i)
                push!(columns, i+j)
                push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi+h, rtol=1e-13)[1])
            end
        end
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi+h, rtol=1e-13)[1]
    end
    if tL == "N"
        xi = 0
        i = 1
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi, xi+h, rtol=1e-13)[1]
        b[i] += a(0)*nL
        for j = [0, +1]
            push!(rows, i)
            push!(columns, i+j)
            push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi, xi+h, rtol=1e-13)[1])
        end
    else
        i = 1
        b[i] -= nL*quadgk(x -> (-a(x)*d_tent(x, h, h)*d_tent(x, 0, h) + d(x)*tent(x, h, h)*d_tent(x, 0, h) + c(x)*tent(x, h, h)*tent(x, 0, h)), 0, h, rtol=1e-13)[1]
        x[i] = nL
    end
    if tR == "N"
        xi = 1
        i = DOF
        b[i] = quadgk(x -> (f(x)*tent(x, xi, h)), xi-h, xi, rtol=1e-13)[1]
        b[i] -= a(1)*nR
        for j = [-1, 0]
            push!(rows, i)
            push!(columns, i+j)
            push!(values, quadgk(x -> (-a(x)*d_tent(x, xi, h)*d_tent(x, xi+j*h, h) + d(x)*tent(x, xi, h)*d_tent(x, xi+j*h, h) + c(x)*tent(x, xi, h)*tent(x, xi+j*h, h)), xi-h, xi, rtol=1e-13)[1])
        end
    else
        i = DOF
        b[i] -= nR*quadgk(x -> (-a(x)*d_tent(x, 1-h, h)*d_tent(x, 1, h) + d(x)*tent(x, 1-h, h)*d_tent(x, 1, h) + c(x)*tent(x, 1-h, h)*tent(x, 1, h)), 1-h, 1, rtol=1e-13)[1]
        x[N] = nR
    end
    A = sparse(rows, columns, values)
    return A, b, x
end

function test_1()
    errors, H = [], []
    for l = [7, 8, 9, 10]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=1:(N-2)]
        ex = x->sin(3*pi*x)+2*x+1
        a = x->1+x^2
        c = x->exp(-(x-1/2)^2)
        leftBC, rightBC = 1, 3
        f = x->(2*x*(3*pi*cos(3*pi*x)+2)-(3*pi)^2*(1+x^2)*sin(3*pi*x))+exp(-(x-1/2)^2)*(sin(3*pi*x)+2*x+1)
        A, b, x = BVP_DD(l, a, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
end

function test_2()
    errors, H = [], []
    for l = [7, 8, 9, 10]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=1:(N-2)]
        ex = x->exp(-x)
        a = x->exp(x)
        c = x->sin(pi*x)^2+1
        leftBC, rightBC = 1, exp(-1)
        f = x->exp(-x)*(sin(pi*x)^2+1)
        A, b, x= BVP_DD(l, a, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
end

function test_3()
    errors, H = [], []
    for l = [7, 8, 9, 10, 11, 12]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=1:(N-2)]
        ex = x -> exp(-x)
        a = x -> exp(x)
        d = x -> x^2
        c = x -> sin(pi*x)^2+1
        leftBC, rightBC = 1, exp(-1)
        f = x -> exp(-x)*(sin(pi*x)^2+1) - exp(-x)*x^2
        A, b, x= BVP_DD(l, a, d, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
    return errors, H, slope
end

function test_4()
    errors, H = [], []
    for l = [7, 8, 9, 10, 11, 12]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=0:(N-2)]
        ex = x -> x^2 + 1
        a = x -> (x + 1)
        d = x -> cos(pi*x)^2
        c = x -> sin(pi*x)^2+1
        leftBC, rightBC = 0, 2
        f = x -> (x^2+1)*(sin(pi*x)^2+1) + 2*x*cos(pi*x)^2 + (4*x+2)
        A, b, x= BVP_ND(l, a, d, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
    return errors, H, slope
end

function test_5()
    errors, H = [], []
    for l = [7, 8, 9, 10, 11, 12]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=1:(N-1)]
        ex = x -> x^2 + 1
        a = x -> (x + 1)
        d = x -> cos(pi*x)^2
        c = x -> sin(pi*x)^2+1
        leftBC, rightBC = 1, 2
        f = x -> (x^2+1)*(sin(pi*x)^2+1) + 2*x*cos(pi*x)^2 + (4*x+2)
        A, b, x= BVP_DN(l, a, d, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
    return errors, H, slope
end

function test_6()
    errors, H = [], []
    for l = [7, 8, 9, 10, 11, 12]
        N = 2^l+1
        h = 1/(N-1)
        coords = [h*i for i=0:(N-1)]
        ex = x -> x^2 + 1
        a = x -> (x + 1)
        d = x -> cos(pi*x)^2
        c = x -> sin(pi*x)^2+1
        leftBC, rightBC = 0, 2
        f = x -> (x^2+1)*(sin(pi*x)^2+1) + 2*x*cos(pi*x)^2 + (4*x+2)
        A, b, x= BVP_NN(l, a, d, c, leftBC, rightBC, f)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
    return errors, H, slope
end

function test(tL::String, tR::String)
    errors, H = [], []
    if tL == "N"
        leftBC = 1
        start = 0
    elseif tL == "D"
        leftBC = 1
        start = 1
    end
    if tR == "N"
        rightBC = 3
    elseif tR == "D"
        rightBC = 3
    end
    for l = [7, 8, 9, 10, 11, 12]
        N = 2^l+1
        DOF = N - 2 + (tL == "N") + (tR == "N")
        h = 1/(N-1)
        coords = [h*i for i=start:(DOF-1+start)]
        ex = x -> x^2 + x + 1
        a = x -> (x + 1)
        d = x -> cos(pi*x)^2
        c = x -> sin(pi*x)^2 + 1
        f = x -> (x^2 + x + 1)*(sin(pi*x)^2 + 1) + (2*x + 1)*cos(pi*x)^2 + (4*x + 3)
        F = [a, d, c, f]
        BCs = [[leftBC, tL], [rightBC, tR]]
        A, b, x = BVP_1D(l, F, BCs)
        error = norm(ex.(coords) - inv(Array(A))*b, Inf)
        push!(H, h)
        push!(errors, error)
    end
    slope = log(errors[1]/errors[end])/log(H[1]/H[end])
    return errors, H, slope
end
