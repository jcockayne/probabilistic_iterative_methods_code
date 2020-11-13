module IterativeMethods

export Richardson, Jacobi, Richardson2, bayescg, cg, StepsizeType
using LinearAlgebra
using Distributions
using Random

@enum StepsizeType Optimal Adaptive


struct StationaryIterativeMethod
    G::AbstractMatrix{<:Real}
    f::AbstractVector{<:Real}
    Name::String
    Stepsize::Union{<:Real, StepsizeType}
end

function Richardson(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                    x::AbstractVector{<:Real},
                    iter::Int64, stepsize::Union{<:Real, StepsizeType})

    if stepsize == Optimal
        stepsize = 2/(eigmin(A)+eigmax(A))
    end

    if stepsize != Adaptive
        G = I - stepsize*A
        f = stepsize*b
    end

    for i=1:iter

        if stepsize == Adaptive
            r = b - A*x
            Ar = A*r
            adaptive_stepsize = (r'*Ar)/(Ar'*Ar)
            G = I - adaptive_stepsize*A
            f = adaptive_stepsize*b
        end
        x = G*x + f
    end

    x
end

function Richardson(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                    x::AbstractVector{<:Real}, Sigma::AbstractMatrix{<:Real},
                    iter::Int64, stepsize::Union{<:Real, StepsizeType})

    if stepsize == Optimal
        stepsize = 2/(eigmin(A)+eigmax(A))
    end

    if stepsize != Adaptive
        G = I - stepsize*A
        f = stepsize*b
    end

    for i=1:iter

        if stepsize == Adaptive
            r = b - A*x
            Ar = A*r
            adaptive_stepsize = (r'*Ar)/(Ar'*Ar)
            G = I - adaptive_stepsize*A
            f = adaptive_stepsize*b
        end
        x = G*x + f
        Sigma = G*Sigma*G'
    end

    x, Sigma
end

function Jacobi(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                x::AbstractVector{<:Real},
                iter::Int64, stepsize::Union{<:Real, StepsizeType})

    DInv = Diagonal(diag(A).^-1)
    M = DInv*A

    if stepsize == Optimal
        stepsize = 2/(eigmin(M)+eigmax(M))
    end

    if stepsize != Adaptive
        G = I - stepsize*M
        f = stepsize*DInv*b
    end

    for i=1:iter

        if stepsize == Adaptive
            r = b - A*x
            Ar = A*r
            adaptive_stepsize = (r'*Ar)/(Ar'*Ar)
            G = I - adaptive_stepsize*M
            f = adaptive_stepsize*DInv*b
        end
        x = G*x + f
    end

    x
end

function Jacobi(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                x::AbstractVector{<:Real}, Sigma::AbstractMatrix{<:Real},
                iter::Int64, stepsize::Union{<:Real, StepsizeType})

    DInv = Diagonal(diag(A).^-1)
    M = DInv*A

    if stepsize == Optimal
        stepsize = 2/(eigmin(M)+eigmax(M))
    end

    if stepsize != Adaptive
        G = I - stepsize*M
        f = stepsize*DInv*b
    end

    for i=1:iter

        if stepsize == Adaptive
            r = b - A*x
            Ar = A*r
            adaptive_stepsize = (r'*Ar)/(Ar'*Ar)
            G = I - adaptive_stepsize*M
            f = adaptive_stepsize*DInv*b
        end
        x = G*x + f
        Sigma = G*Sigma*G'
    end

    x, Sigma
end

function Richardson2(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                     x::AbstractVector{<:Real}, iter::Int64)
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
    
    for i=1:iter
        x = BlockG*x+BlockF
    end
    x
end

function Richardson2(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                     x::AbstractVector{<:Real}, Sigma::AbstractMatrix{<:Real},
                     iter::Int64)
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
    gamma = 2/(1 + sqrt(1 - sigma^2))

    # Matrices for Second Degree Richardson
    J = gamma*sigma*(2/(beta-alpha)*G-(beta+alpha)/(beta-alpha)*I)
    H = (1 - gamma)*I
    f = 2*gamma*sigma/(beta-alpha)*f
    
    # Block matrices to make Second Degree Richardson a first degree method
    BlockG = [[J H]; [I zeros(N,N)]]
    BlockF = [f; zeros(N)]

    for i=1:iter
        x = BlockG*x + BlockF
        Sigma = BlockG*Sigma*BlockG'
    end
    x, Sigma
end

function bayescg(A::AbstractMatrix, b::AbstractVector, x::AbstractVector,
                 Sigma::AbstractMatrix, it::Int64 , tol::Float64)
    N = length(x)

    # Set up residual
    r = zeros(N,it+1)
    r[:,1] = b - A*x
    rIP = zeros(it+1)
    rIP[1] = r[:,1]'*r[:,1]

    # Set up search directions
    s = r[:,1]
    SigAs_norm = zeros(N,it)

    # Convergence parameters
    res = sqrt(rIP[1])
    tol = tol*norm(b)
    i = 1
    
    while i <= it && res > tol

        # Matrix Vector Products
        SigAs = Sigma*(A*s)
        ASigAs = A*SigAs
        sIP = s'*ASigAs

        # Store vector for posterior covariance
        SigAs_norm[:,i] = SigAs/sqrt(sIP)

        # Update x
        alpha = rIP[i]/sIP
        x = x + alpha*SigAs

        # Update residual
        r[:,i+1] = r[:,i] - alpha*ASigAs

        # Reorthogonalize
        r[:,i+1] = r[:,i+1] - (r[:,1:i]*Diagonal(rIP[1:i].^-1)
                               *r[:,1:i]'*r[:,i+1])
        r[:,i+1] = r[:,i+1] - (r[:,1:i]*Diagonal(rIP[1:i].^-1)
                               *r[:,1:i]'*r[:,i+1])
        rIP[i+1] = r[:,i+1]'*r[:,i+1]

        # Update search direction
        beta = rIP[i+1]/rIP[i]
        s = r[:,i+1] + beta*s

        # Update convergence parameters
        res = sqrt(rIP[i+1])
        i = i+1
    end

    # Compute posterior covariance
    Sigma = Sigma - SigAs_norm[:,1:i-1]*SigAs_norm[:,1:i-1]'

    return x, Sigma
end

function cg(A::AbstractMatrix, b::AbstractVector, x::AbstractVector,
            it::Int64 , tol::Float64)
    N = length(x)

    # Set up residual
    r = zeros(N,it+1)
    r[:,1] = b - A*x
    rIP = zeros(it+1)
    rIP[1] = r[:,1]'*r[:,1]

    # Set up search directions
    s = r[:,1]
    SigAs_norm = zeros(N,it)

    # Convergence parameters
    res = sqrt(rIP[1])
    tol = tol*norm(b)
    i = 1
    
    while i <= it && res > tol

        # Matrix Vector Products
        As = A*s
        sIP = s'*As

        # Update x
        alpha = rIP[i]/sIP
        x = x + alpha*s

        # Update residual
        r[:,i+1] = r[:,i] - alpha*As

        # Reorthogonalize
        r[:,i+1] = r[:,i+1] - (r[:,1:i]*Diagonal(rIP[1:i].^-1)
                               *r[:,1:i]'*r[:,i+1])
        r[:,i+1] = r[:,i+1] - (r[:,1:i]*Diagonal(rIP[1:i].^-1)
                               *r[:,1:i]'*r[:,i+1])
        rIP[i+1] = r[:,i+1]'*r[:,i+1]

        # Update search direction
        beta = rIP[i+1]/rIP[i]
        s = r[:,i+1] + beta*s

        # Update convergence parameters
        res = sqrt(rIP[i+1])
        i = i+1
    end

    return x
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



function apply(method::StationaryIterativeMethod, dist::AbstractMvNormal, iter::Int64=1)
    iter == 0 ? dist : apply(method,  method.f + method.G*dist, iter-1)
end


function apply(method::StationaryIterativeMethod, mean_vec::AbstractVector, cov_mat::AbstractMatrix, iter::Int64=1)
    if iter == 0
        return mean_vec, cov_mat
    else
        mean_vec = method.G*mean_vec+method.f
        cov_mat = method.G*cov_mat*method.G'
        return apply(method, mean_vec, cov_mat, iter-1)
    end
end

function apply(method::StationaryIterativeMethod, x::AbstractVector, iter::Int64=1)
    if iter == 0
        return x
    else
        x = method.G*x+method.f
        return apply(method, x, iter-1)
    end
end


function sample(method::StationaryIterativeMethod, dist::MultivariateDistribution, iter::Int64=1, samples::Int64=1)
    init_mat = Matrix{Float64}(undef, length(dist), samples)
    rand!(dist, init_mat)
    sample(method, init_mat, iter)
end

function sample(method::StationaryIterativeMethod, samp::Union{Vector{<:Real}, Matrix{<:Real}}, iter::Int64=1)
    iter == 0 ? samp : sample(method, method.G * samp .+ method.f, iter-1)
end

end
