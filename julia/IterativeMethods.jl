module IterativeMethods

export Richardson, Jacobi, Richardson2, apply, Stepsize
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
    stepsize_out = stepsize
    if stepsize == Optimal
        stepsize = 2/(eigmin(A)+eigmax(A))
    end
    G = I - stepsize*A
    f = stepsize*b
    StationaryIterativeMethod(G, f, "Richardson", stepsize_out)
end

function Jacobi(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, stepsize::Union{<:Real, StepsizeType})
    DInv = Diagonal(diag(A).^-1)
    M = DInv*A
    stepsize_out = stepsize
    if stepsize == Optimal
        stepsize = 2/(eigmin(M)+eigmax(M))
    end
    G = I - stepsize*M
    f = stepsize*DInv*b
    StationaryIterativeMethod(G, f, "Jacobi", stepsize_out)
end

function Richardson2(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    N = length(b)

    # Stepsize for First Degree Richardson 
    stepsize_out = Optimal
    stepsize = 2/(eigmin(A)+eigmax(A))

    # Matrices for First Degree Richardson
    G = I - stepsize*A
    f = stepsize*b

    """
    # GOLUB AND VARGA

    # Parameters for Second Degree Richardson
    alpha = 2/(1+sqrt(1-eigmax(G)^2))
    beta = -1

    # Matrices for Second Degree Richardson
    J = (1+beta)*I+alpha*G
    H = (-alpha-beta)*I
    f = alpha*f
    """

    # YOUNG

    # Parameters for Second Degree Richardson
    alpha = eigmin(G)
    beta = eigmax(G)
    sigma = (beta - alpha)/(2 - (beta + alpha))
    omega = 2/(1 + sqrt(1 - sigma^2))

    # Matrices for Second Degree Richardson
    J = omega*sigma*(2/(beta-alpha)*G-(beta+alpha)/(beta-alpha)*I)
    H = (1 - omega)*I
    f = 2*omega*sigma/(beta-alpha)*f
    
    # Block matrices to make Second Degree Richardson a first degree method
    BlockG = [[J H]; [I zeros(N,N)]]
    BlockF = [f; zeros(N)]
    
    StationaryIterativeMethod(BlockG, BlockF, "Richardson2", stepsize_out)
end

function apply(method::StationaryIterativeMethod, dist::AbstractMvNormal, iter::Integer=1)
    iter == 0 ? dist : apply(method,  method.f + method.G*dist, iter-1)
end


function apply(method::StationaryIterativeMethod, mean_vec::AbstractVector, cov_mat::AbstractMatrix, iter::Integer=1)
    if iter == 0
        return mean_vec, cov_mat
    else
        mean_vec = method.G*mean_vec+method.f
        cov_mat = method.G*cov_mat*method.G'
        return apply(method, mean_vec, cov_mat, iter-1)
    end
end

function apply(method::StationaryIterativeMethod, x::AbstractVector, iter::Integer=1)
    if iter == 0
        return x
    else
        x = method.G*x+method.f
        return apply(method, x, iter-1)
    end
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
