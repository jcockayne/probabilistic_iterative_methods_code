using LinearAlgebra

Z = vcat(range(0, stop=0.1, length=20), range(0.2, stop=0.8, length=400),
         range(0.9, stop=1, length=20))
d = length(Z)

M = 200
Y = range(0, stop=1, length=M)

f(z) = z < 0.5 ? sin(2*pi*z) : sin(4*pi*z)

f_Y = f.(Y)

k(x::Number, y::Number, lambda::Number) :: Float64 = (1 + (norm(x - y)
                                                           /lambda)^2)^(-1)

function k(x::AbstractVector{<:Number}, y::AbstractVector{<:Number},
           lambda::Number) :: Matrix{Float64}
    result = Matrix{Real}(undef, (size(x, 1), size(y, 1)))
    for i = 1:size(x, 1)
        for j = 1:size(y, 1)
            result[i, j] = k(x[i], y[j], lambda)
        end
    end
    result
end

f_Z = f.(Z)
lambda = 0.0012
k_Z = k(Z, Z, lambda)
k_YZ = k(Y, Z, lambda)

if eigmin(k_Z) < 0.
    throw(Exception("The matrix k_Z was not positive-definite."))
end

g(Y, Z, X) = k(Y, Z, lambda) * X

X_direct_solution = k_Z \ f_Z
g_direct_solution = g(Y, Z, X_direct_solution)
