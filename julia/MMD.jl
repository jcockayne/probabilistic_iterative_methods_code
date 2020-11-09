module MMD

using LinearAlgebra
using Statistics

export mmd_slow, mmd, mmd_kernel_factory

function cdist(A::Matrix{Float64}, B::Matrix{Float64})
    M = size(A)[2]
    N = size(B)[2]

    dist = zeros(Float64,M,N)

    if A == B
        # A = B, or close enough
        for m = 1:(M-1)
            for n = (m+1):N
                dist[m,n] = norm(A[:,m]-B[:,n])
            end
        end
        dist = dist + dist'
    else
        # A not equal B
        for m = 1:M
            for n = 1:N
                dist[m,n] = norm(A[:,m]-B[:,n])
            end
        end
    end

    return dist
end

function mmd_slow(input_samples::Matrix{Float64},
                  output_samples::Matrix{Float64}, kernel::Function)
    N = size(input_samples)[2]

    mmd_est = 0
    
    for i = 1:N
        for j = 1:N
            if i != j
                mmd_est += (kernel(input_samples[:,i],input_samples[:,j])
                            + kernel(output_samples[:,i],output_samples[:,j])
                            - kernel(input_samples[:,i],output_samples[:,j])
                            - kernel(output_samples[:,i],input_samples[:,j])
                            )
            end
        end
    end
    mmd_est = mmd_est/(N*(N-1))
    return mmd_est
end

function mmd(input_samples::Matrix{Float64}, output_samples::Matrix{Float64},
             kernel::Function)
    N = size(input_samples)[2]

    mmd_est = 0
    
    for i = 1:(N-1)
        for j = (i+1):N
            mmd_est += (2*kernel(input_samples[:,i],input_samples[:,j])
                        + 2*kernel(output_samples[:,i],output_samples[:,j])
                        - 2*kernel(input_samples[:,i],output_samples[:,j])
                        - 2*kernel(output_samples[:,i],input_samples[:,j])
                        )
        end
    end
    mmd_est = mmd_est/(N*(N-1))
    return mmd_est
end

function mmd_fast(input_samples::Matrix{Float64},
                  output_samples::Matrix{Float64}, kernel::Function)

    N = size(input_samples)[2]

    d1 = kernel(input_samples,input_samples)
    d2 = kernel(output_samples,output_samples)
    d3 = kernel(input_samples,output_samples)

    d1 = d1 - Diagonal(d1)
    d2 = d2 - Diagonal(d2)
    d3 = d3 - Diagonal(d3)

    mmd_est = sum(d1 + d2 - d3 - d3')
    mmd_est = mmd_est/(N*(N-1))
    
    return mmd_est
end


function mmd_kernel_factory(input_samples::Matrix{Float64},
                            output_samples::Matrix{Float64})

    N = size(input_samples)[2]

    len_dist = convert(Int64,(2*N-1)*(2*N)/2)

    dist = Array{Float64}(undef,len_dist)
    d = 1
    for i = 1:N
        for j = (i+1):N
            dist[d] = norm(input_samples[:,i] - input_samples[:,j])
            d = d + 1
        end
        for j = 1:N
            dist[d] = norm(input_samples[:,i] - output_samples[:,j])
            d = d + 1
        end
    end
    for i = 1:(N-1)
        for j = (i+1):N
            dist[d] = norm(output_samples[:,i] - output_samples[:,j])
            d = d + 1
        end
    end
    dist_mean = mean(dist)

    function gauss_ker(x::Array{Float64,2},y::Array{Float64,2})
        d = cdist(x,y)
        return exp.(-0.5.*(d.^2)./dist_mean^2)
    end

    function gauss_ker(x::Array{Float64,1},y::Array{Float64,1})
        return exp.(-0.5.*(norm(x-y).^2)./dist_mean^2)
    end

    return gauss_ker
end
end
