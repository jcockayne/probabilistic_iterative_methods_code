module AdaptiveIterativeMethods

export RichardsonAdapt, JacobiAdapt, Richardson2

using LinearAlgebra

function RichardsonAdapt(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                         x::AbstractVector{<:Real}, iter::Integer=10)
    for i = 1:iter
        Ax = A*x
        r = b - Ax
        Ar = A*r
        stepsize = (r'*Ar)/(Ar'*Ar)
        #stepsize = 2/(eigmin(A)+eigmax(A))
        # x = Gx + f; G = I - stepsize*A; f = stepsize*b
        x = I*x - stepsize*Ax + stepsize*b
    end
    return x
end

function JacobiAdapt(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                     x::AbstractVector{<:Real}, iter::Integer=10)
    for i = 1:iter
        DInv = Diagonal(diag(A).^-1)
        M = DInv*A
        r = b - A*x
        Ar = A*r
        stepsize = (r'*Ar)/(Ar'*Ar)
        # x = Gx + f; G = I - stepsize*M; f = stepsize*DInv*b
        x = I*x - stepsize*M*x + stepsize*DInv*b
    end
    return x
end

function Richardson2(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                     x::AbstractVector{<:Real},xold::AbstractVector{<:Real},
                     iter::Integer=10)
    for i = 1:iter
        stepsize = 2/(eigmin(A)+eigmax(A))
        # x = Gx + f; G = I - stepsize*A; f = stepsize*b
        G = I - stepsize*A
        f = stepsize*b

        alpha = 2/(1+sqrt(1-eigmax(G)^2))
        beta = -1

        J = (1+beta)*I+alpha*G
        H = (-alpha-beta)*I
        
        xnew = J*x + H*xold + alpha*f
        xold = x
        x = xnew
    end
    return x
end

end
