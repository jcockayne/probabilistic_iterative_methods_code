@everywhere module MMD

using LinearAlgebra
using Statistics
using StatsBase
using Distributed

export mmd_kernel_factory, mmd, bootstrap_mmd

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

    function gauss_ker(x::Array{Float64,1},y::Array{Float64,1})
        return exp.(-0.5.*(norm(x-y).^2)./dist_mean^2)
    end

    return gauss_ker
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

function mmd_p(input_samples::Matrix{Float64}, output_samples::Matrix{Float64},
               kernel::Function)
    N = size(input_samples)[2]

    mmd_est = 0
    
    mmd_est = @distributed (+) for i = 1:(N-1)
        mmd_inner = 0
        for j = (i+1):N
            mmd_inner += (2*kernel(input_samples[:,i],input_samples[:,j])
                          + 2*kernel(output_samples[:,i],output_samples[:,j])
                          - 2*kernel(input_samples[:,i],output_samples[:,j])
                          - 2*kernel(output_samples[:,i],input_samples[:,j])
                          )
        end
        mmd_inner
    end
    mmd_est = mmd_est/(N*(N-1))
    return mmd_est
end

function bootstrap_mmd(input_samples::Matrix{Float64},
                       output_samples::Matrix{Float64}, kernel::Function,
                       N_mmd::Int64)

    M = size(input_samples)[2]
    combined_sample = [input_samples output_samples]

    mmd_list = Array{Tuple{Array{Float64,2},Array{Float64,2}},1}(undef,N_mmd)

    mmd_map = function (sample_tuple::Tuple{Array{Float64,2},Array{Float64,2}})
        mmd(sample_tuple[1], sample_tuple[2],kernel)
    end

    for i = 1:N_mmd
        s1_index = sample(1:2*M,M,replace=false)
        #s2_index = sample(1:2*M,M)
        not_in_s1(y) = !(y in s1_index)
        s2_index = filter(not_in_s1,1:2*M)

        mmd_list[i] = (combined_sample[:,s1_index],
                       combined_sample[:,s2_index])
    end

    return pmap(mmd_map,mmd_list)
end

end
