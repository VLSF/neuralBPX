using SparseArrays, LinearAlgebra, HCubature, QuadGK, SymPy

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

function BVP_DD_2D(l::Integer, F::Array{Function, 1}, f::Function)
    N = 2^l+1
    h = 1/(N-1)
    a11, a12, a22, b1, b2, c = F
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, (N-2)^2)
    lex(i, j) = i + (N-2)*(j-1)
    for (i, j) = collect(Iterators.product(1:(N-2), 1:(N-2)))
        xi, yi = h*i, h*j
        for (p1, p2) = collect(Iterators.product([-1, 0, +1], [-1, 0, +1]))
            if (1<=(i+p1)<=N-2) & (1<=(j+p2)<=N-2)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p1, j+p2))
                g1(x) = - a11(x)*(d_tent(x[1], xi, h)*tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                g2(x) = - a12(x)*(d_tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h)) - a12(x)*(tent(x[1], xi, h)*d_tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                g3(x) = - a22(x)*(tent(x[1], xi, h)*d_tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h))
                v1(x) = b1(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                v2(x) = b2(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h))
                c1(x) = c(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                val = hcubature(x -> g1(x)+g2(x)+g3(x)+v1(x)+v2(x)+c1(x), (xi-h, yi-h), (xi+h, yi+h), atol=1e-10)[1]
                push!(values, val)
            end
        end
        b[lex(i, j)] = hcubature(x -> f(x)*tent(x[1], xi, h)*tent(x[2], yi, h), (xi-h, yi-h), (xi+h, yi+h), atol=1e-10)[1]
    end
    A = sparse(rows, columns, values)
    return A, b
end

function BVP_2D(l::Integer, F::Array{Function, 1}, BCs::Array{String, 1})
    N = 2^l+1
    h = 1/(N-1)
    DOF_x, DOF_y = N - 2 + sum(BCs[1:2] .== "N"), N - 2 + sum(BCs[3:4] .== "N")
    shift_x, shift_y = (BCs[1] == "N"), (BCs[3] == "N")
    a11, a12, a22, b1, b2, c, f = F
    rows, columns, values = [], [], Array{Float64,1}(undef, 0)
    b = Array{Float64,1}(undef, DOF_x*DOF_y)
    lex(i, j) = i + DOF_x*(j-1)
    for (i, j) = collect(Iterators.product(1:DOF_x, 1:DOF_y))
        xi, yi = h*(i-shift_x), h*(j-shift_y)
        for (p1, p2) = collect(Iterators.product([-1, 0, +1], [-1, 0, +1]))
            if (1<=(i+p1)<=DOF_x) & (1<=(j+p2)<=DOF_y)
                push!(rows, lex(i, j))
                push!(columns, lex(i+p1, j+p2))
                g1(x) = - a11(x)*(d_tent(x[1], xi, h)*tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                g2(x) = - a12(x)*(d_tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h)) - a12(x)*(tent(x[1], xi, h)*d_tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                g3(x) = - a22(x)*(tent(x[1], xi, h)*d_tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h))
                v1(x) = b1(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*d_tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                v2(x) = b2(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*d_tent(x[2], yi+h*p2, h))
                c1(x) = c(x)*(tent(x[1], xi, h)*tent(x[2], yi, h)*tent(x[1], xi+h*p1, h)*tent(x[2], yi+h*p2, h))
                val = hcubature(x -> g1(x)+g2(x)+g3(x)+v1(x)+v2(x)+c1(x), (max(xi-h, 0), max(yi-h, 0)), (min(xi+h, 1), min(yi+h, 1)), atol=1e-10)[1]
                if ((xi == 0) || (xi == 1)) & (p1 == 0)
                    val -= (-1)^xi*quadgk(y -> (a12([xi, y])*d_tent(y, yi+h*p2, h)*tent(y, yi, h)), max(yi-h, 0), min(yi+h, 1), rtol=1e-13)[1]
                end
                if ((yi == 0) || (yi == 1)) & (p2 == 0)
                    val -= (-1)^yi*quadgk(x -> (a12([x, yi])*d_tent(x, xi+h*p1, h)*tent(x, xi, h)), max(xi-h, 0), min(xi+h, 1), rtol=1e-13)[1]
                end
                push!(values, val)
            end
        end
        b[lex(i, j)] = hcubature(x -> f(x)*tent(x[1], xi, h)*tent(x[2], yi, h), (max(xi-h, 0), max(yi-h, 0)), (min(xi+h, 1), min(yi+h, 1)), atol=1e-10)[1]
    end
    A = sparse(rows, columns, values)
    return A, b
end

function test()
    for BCs_ = Iterators.product(["D", "N"], ["D", "N"], ["D", "N"], ["D", "N"])
        BCs = collect(BCs_)
        x, y = sympy.symbols("x, y")
        u = 1
        if BCs[1:2] == ["D", "D"]
            u *= x*(x-1)
        elseif BCs[1:2] == ["D", "N"]
            u *= ((x^2)/2-x)
        elseif BCs[1:2] == ["N", "D"]
            u *= (1+cos(pi*x))
        elseif BCs[1:2] == ["N", "N"]
            u *= cos(pi*x)
        end
        if BCs[3:4] == ["D", "D"]
            u *= y*(1-y)
        elseif BCs[3:4] == ["D", "N"]
            u *= ((y^2)/2-y)
        elseif BCs[3:4] == ["N", "D"]
            u *= (y^2-1)
        elseif BCs[3:4] == ["N", "N"]
            u *= cos(4*pi*y)
        end
        a11 = 5 + x^2 + y^2
        a12 = (x-1)*(y-1) + 3*x*y
        a22 = 5 + x^2 + 2*y^2
        b1 = x*y/10
        b2 = (x-y)/5
        c = (x+y+1)^2
        rhs = (a11*u.diff(x)).diff(x) + (a22*u.diff(y)).diff(y) + (a12*u.diff(y)).diff(x) + (a12*u.diff(x)).diff(y) + b1*u.diff(x) + b2*u.diff(y) + c*u
        @vars x, y
        a11 = lambdify(a11)
        a12 = lambdify(a12)
        a22 = lambdify(a22)
        b1 = lambdify(b1)
        b2 = lambdify(b2)
        c = lambdify(c)
        f = lambdify(rhs)
        u = lambdify(u)
        f_(x) = f(x[1], x[2])
        u_(x) = u(x[1], x[2])
        a11_(x) = a11(x[1], x[2])
        a12_(x) = a12(x[1], x[2])
        a22_(x) = a22(x[1], x[2])
        b1_(x) = b1(x[1], x[2])
        b2_(x) = b2(x[1], x[2])
        c_(x) = c(x[1], x[2])
        a = [a11_, a12_, a22_, b1_, b2_, c_, f_]

        E = []
        Hs = []
        for L = [3, 4, 5]
            N = 2^L+1
            h = 1/(N-1)
            DOF_x, DOF_y = N - 2 + sum(BCs[1:2] .== "N"), N - 2 + sum(BCs[3:4] .== "N")
            shift_x, shift_y = (BCs[1] == "N"), (BCs[3] == "N")
            exact = Array{Float64,1}(undef, DOF_x*DOF_y)
            lex(i, j) = i + DOF_x*(j-1)
            for (i, j) = collect(Iterators.product(1:DOF_x, 1:DOF_y))
                xi, yi = h*(i-shift_x), h*(j-shift_y)
                exact[lex(i, j)] = u_([xi, yi])
            end
            A, b = BVP_2D(L, a, BCs)
            F = lu(A)
            push!(E, norm(exact - F\b, Inf))
            push!(Hs, h)
        end
        slope = log10(E[end-1]/E[end])/log10(Hs[end-1]/Hs[end])
        println("Boundary conditions ", BCs, ", the slope ", round(slope, digits=4))
    end
    return nothing
end
