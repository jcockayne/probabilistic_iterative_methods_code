module KernelPlots

export kernel_plots, pc_plots

using Plots
using LinearAlgebra

function mv_normal(mean,cov,N_samples)
    M = length(mean)
    N = convert(Int64,round(length(cov[:])/M))
    cov = reshape(cov,M,N)
    samples = zeros(M,N_samples)
    for i = 1:N_samples
        samples[:,i] = mean + cov*randn(N)
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
                                legend = false, ylims = (-1.5,1.5))
            plot!(Y, K_YZ*direct, color=:blue, legend = false)

            if n == 1
                plot!(ylabel = solver_labels[m], yticks = [-1.0,0.0,1.0],
                      grid = false)
            else
                plot!(yticks = false, grid = false)
            end
            if m == 1
                plot!(title = "m = "*string(iters[n]),titlefontsize = 10)
            end
            if m != M
                plot!(xticks = false, grid = false)
            else
                plot!(xticks = [0.0,0.5,1.0], grid = false)
            end
            
            push!(plot_list, current_plot)
        end
    end
    output_size = (200*N,200*M)
    output_plot = plot(plot_list..., layout=(M,N), link=:y, size=output_size)  
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
            current_plot = plot(Y, samples, color=:black, α=0.1,
                                legend = false)
            plot!(Y, K_YZ*direct, color=:blue, legend = false)

            if n == 1
                plot!(ylabel = "PC "*string(m), yticks = [-1.0,0.0,1.0],
                      grid = false)
            else
                plot!(yticks = false, grid = false)
            end
            if m == 1
                plot!(title = "m = "*string(iters[n])
                      *", "*string(round(F.S[m]/sum(F.S)*100,
                                         digits = 2))*" %",
                      titlefontsize = 10)
            else
                plot!(title = string(round(F.S[m]/sum(F.S)*100,
                                           digits = 2))*" %",
                      titlefontsize = 10)
            end
            if m != N_pc
                plot!(xticks = false, grid = false)
            else
                plot!(xticks = [0.0,0.5,1.0], grid = false)
            end
            
            plot_list[n,m] = current_plot
        end
    end
    output_plot = plot(plot_list[:]..., layout=(N_pc,N), link=:y,
                       size=(200*N,200*N_pc))
end
    
end
