using LinearAlgebra
using Distributions
using PDMats
#using PyPlot
using PyCall

mplt = pyimport("matplotlib.pyplot")

##
include("IterativeMethods.jl")
include("ExperimentHelpers.jl")
Z = vcat(range(0, stop=0.1, length=20), range(0.2, stop=0.8, length=400), range(0.9, stop=1, length=20))
d = length(Z)

M = 200
Y = range(0, stop=1, length=M)

f(z) = z < 0.5 ? sin(2*pi*z) : sin(4*pi*z)

f_Y = f.(Y)

k(x::Number, y::Number, lambda::Number) :: Float64 = (1 + (norm(x - y)/lambda)^2)^(-1)

function k(x::AbstractVector{<:Number}, y::AbstractVector{<:Number}, lambda::Number) :: Matrix{Float64}
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

if eigmin(k_Z) < 0.
    throw(Exception("The matrix k_Z was not positive-definite."))
end

g(Y, Z, X) = k(Y, Z, lambda) * X

X_direct_solution = k_Z \ f_Z
g_direct_solution = g(Y, Z, X_direct_solution)

## define all the combinations of methods

n_iter_vals = [0, 3, 5, 10, 100]
iter_methods = [
    IterativeMethods.Richardson(k_Z, f_Z, IterativeMethods.Optimal),
    IterativeMethods.Jacobi(k_Z, f_Z, IterativeMethods.Optimal),
    IterativeMethods.Jacobi(k_Z, f_Z, 2/3)
]
init_distributions = [
    MvNormal(zeros(d), PDiagMat(ones(d))),
    MvNormalCanon(zeros(d), PDMat(k_Z))
]
init_distribution_labels = Dict([
    (init_distributions[1], "default"),
    (init_distributions[2], "natural")
])

experiments = [
    ExperimentHelpers.Experiment(n, method, dist)
    for (n, method, dist) = Iterators.product(n_iter_vals, iter_methods, init_distributions)
]
## run all the methods and generate mean and covariance
results = Dict()
samples = 50
for experiment = experiments
    rvs = IterativeMethods.sample(experiment.Method, experiment.InitialDist, experiment.Iterations, samples)
    if experiment.InitialDist isa MvNormal
        input_cov = convert(Array,experiment.InitialDist.Σ)
    elseif experiment.InitialDist isa MvNormalCanon
        input_cov = convert(Array,inv(experiment.InitialDist.J))
    end
    empirical_mean,empirical_cov = IterativeMethods.apply(experiment.Method, experiment.InitialDist.μ, input_cov, experiment.Iterations)
    #empirical_mean = mean(rvs)
    #empirical_cov = cov(rvs)
    results[experiment] = ExperimentHelpers.ExperimentResult(empirical_mean, empirical_cov, rvs)
end
## plot the output
#pygui(true)
layout_rows = length(init_distributions)*length(iter_methods)
layout_cols = length(n_iter_vals)

scale = 1.5
fig,ax = mplt.subplots(layout_rows,layout_cols,sharex="all",sharey="all",figsize = (scale*8,scale*6),num=1)


function experiment_to_label(experiment::ExperimentHelpers.Experiment) :: String

    method_name = experiment.Method.Name
    method_stepsize = experiment.Method.Stepsize == IterativeMethods.Optimal ? "(i)" : "(ii)"
    method_prior = init_distribution_labels[experiment.InitialDist]
    "$method_name,\n $method_stepsize, $method_prior"
end
for (col_ix, n_iter) = enumerate(n_iter_vals)
    for (row_ix, (dist, method)) = enumerate(Iterators.product(init_distributions, iter_methods))
        experiment = ExperimentHelpers.Experiment(n_iter, method, dist)
        result = results[experiment]
        ax[row_ix, col_ix].plot(Y, g_direct_solution)
        ax[row_ix, col_ix].plot(Y, g(Y, Z, result.Samples[:, 1:5]), color=:black, alpha=0.2, linewidth=1)

        if col_ix == 1
            label = experiment_to_label(experiment)
            ax[row_ix,1].set_ylabel(label,rotation=90)
        end
    end
    ax[1,col_ix].set_title("\$m=$n_iter\$")
    ax[end, col_ix].set_xlabel("x")
end

fig.tight_layout()

#mplt.show()
fig.savefig("unscaled_id_prior.pdf")

fig.clear()

### Principal components

function pc_sample(mean_vector :: Array{Float64}, sqrt_cov :: Array{Float64}) :: Array{Float64}
    #N = size(sqrt_cov)[2]
    randn(1).*sqrt_cov + mean_vector
end

function float2sci(x::Float64, figs::Int=2)
    power = convert(Int,round(log10(x),RoundDown))
    if -1 <= power <= 2
        figs = max(figs-power,2)
        x = string(round(x,digits=figs))
    else
        x = round(x/(10.0^power),digits=figs)
        x = string(x,"\\times10^{",power,"}")
    end

    return x
end

princ_comp = 8
princ_comp_samples = 20

for (experiment_number, (dist, method)) = enumerate(Iterators.product(init_distributions, iter_methods))
    
    fig,ax = mplt.subplots(princ_comp,layout_cols,sharex="all",sharey="all",figsize = (scale*8,scale*6),num=1)


    experiment = ExperimentHelpers.Experiment(n_iter_vals[1], method, dist)
    
    for (col_ix, n_iter) = enumerate(n_iter_vals)
        if col_ix != 1
            experiment = ExperimentHelpers.Experiment(n_iter, method, dist)
        end
        F = svd(results[experiment].Cov)
        mean_vec = results[experiment].Mean
        for row_ix = 1:princ_comp
            ax[row_ix, col_ix].plot(Y, g_direct_solution)
            for samp = 1:princ_comp_samples
                ax[row_ix, col_ix].plot(Y, g(Y, Z, pc_sample(mean_vec,F.U[:,row_ix])), color=:black, alpha=0.2, linewidth=1)
            end

            if col_ix == 1
                label = "\$PC=$row_ix\$"
                ax[row_ix,1].set_ylabel(label,rotation=90)
            end
            if row_ix == 1
                SV = float2sci(F.S[row_ix])
                title_str = string("\$ \\sigma = ",SV,"\$")
            else
                SV = float2sci(F.S[row_ix])
                title_str = string("\$ \\sigma = ",SV,"\$")
            end
            ax[row_ix,col_ix].set_title(title_str)
        end
        ax[end, col_ix].set_xlabel("x")
    end
    
    fig.tight_layout()

    #mplt.show()
    figname = string(experiment_to_label(experiment),".pdf")
    fig.savefig(figname)

    fig.clear()
end

