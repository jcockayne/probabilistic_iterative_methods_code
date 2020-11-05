module KernelPlots

export kernel_plots, pc_plots

using Plots
using LinearAlgebra

function mv_normal(mean,cov_sqrt,N_samples)
    M = length(mean)
    N = convert(Int64,round(length(cov_sqrt[:])/M))
    cov_sqrt = reshape(cov_sqrt,M,N)
    samples = zeros(M,N_samples)
    for i = 1:N_samples
        samples[:,i] = mean + cov_sqrt*randn(N)
    end
    return samples
end

function kernel_plots(solvers::Array{Function}, iters::Array{Int64},
                      solver_labels::Array{String},
                      Y::AbstractVector, K_YZ::AbstractMatrix,
                      direct::AbstractVector, N_samples::Int64)

    M = length(solvers)
    N = length(iters)
    plot_list = Any[]

    for m = 1:M
        for n = 1:N
            mean,cov = solvers[m](iters[n])

            F = svd(cov)
            cov_sqrt = F.V*((F.S.^(1/2)).*F.V')
            samples = mv_normal(mean,cov_sqrt,N_samples)

            current_plot = plot(Y, K_YZ*samples, color=:black, α=0.025,
                                legend = false)
            plot!(Y, K_YZ*direct, color=:blue, legend = false)

            if n == 1
                yaxis!(solver_labels[m])
            end
            if m == 1
                title!("m = "*string(iters[n]))
            end
            
            push!(plot_list, current_plot)
        end
    end
    plot(plot_list..., layout=(M,N), link=:y)
end

function pc_plots(solver::Function, iters::Array{Int64}, N_pc::Int64,
                 Y::AbstractVector, K_YZ::AbstractMatrix,
                 direct::AbstractVector, N_samples::Int64)

    N = length(iters)
    plot_list = Array{Any}(undef, N, N_pc)

    for n = 1:N
        mean,cov = solver(iters[n])
        interpolation_cov = K_YZ*cov*K_YZ'
        F = svd(interpolation_cov)
        
        for m = 1:N_pc
            samples = mv_normal(K_YZ*mean,F.U[:,m],N_samples)
            current_plot = plot(Y, samples, color=:black, α=0.025,
                                legend = false)
            plot!(Y, K_YZ*direct, color=:blue, legend = false)

            if n == 1
                yaxis!("PC "*string(m))
            end
            if m == 1
                title!("m = "*string(iters[n]))
            end
            
            plot_list[n,m] = current_plot
        end
    end
    plot(plot_list[:]..., layout=(N_pc,N), link=:y)
end
    
end
