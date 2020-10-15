module MMD

export mmd

function mmd(input_samples::AbstractMatrix, output_samples::AbstractMatrix,
             kernel::Function)
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
end
