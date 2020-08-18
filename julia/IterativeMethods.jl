module IterativeMethods

export Richardson, Jacobi, apply, Stepsize
using LinearAlgebra
using Distributions
using Random

@enum StepsizeType Optimal

struct StationaryIterativeMethod
    G::AbstractMatrix{<:Real}
    f::AbstractVector{<:Real}
    Name::String
    Stepsize::Union{<:Real, StepsizeType}
end

function Richardson(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, stepsize::Union{<:Real, StepsizeType})
    if stepsize == Optimal
        stepsize = 2/(eigmin(A)+eigmax(A))
    end
    G = I - stepsize*A
    f = stepsize*b
    StationaryIterativeMethod(G, f, "Richardson", stepsize)
end

function Jacobi(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, stepsize::Union{<:Real, StepsizeType})
    D = Diagonal(diag(A))
    M = D \ A
    if stepsize == Optimal
        stepsize = 2/(eigmin(M)+eigmax(M))
    end
    G = I - stepsize*M
    f = stepsize*D \ b
    StationaryIterativeMethod(G, f, "Jacobi", stepsize)
end

function apply(method::StationaryIterativeMethod, dist::AbstractMvNormal, iter::Integer=1)
    iter == 0 ? dist : apply(method,  method.f + method.G*dist, iter-1)
end

function sample(method::StationaryIterativeMethod, dist::MultivariateDistribution, iter::Integer=1)

end

function sample(method::StationaryIterativeMethod, dist::MultivariateDistribution, iter::Integer=1, samples::Integer=1)
    init_mat = Matrix{Float64}(undef, length(dist), samples)
    rand!(dist, init_mat)
    sample(method, init_mat, iter)
end

function sample(method::StationaryIterativeMethod, samp::Union{Vector{<:Real}, Matrix{<:Real}}, iter::Integer=1)
    iter == 0 ? samp : sample(method, method.G * samp .+ method.f, iter-1)
end

end
