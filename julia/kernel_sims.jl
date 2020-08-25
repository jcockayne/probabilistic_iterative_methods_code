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
    empirical_mean = mean(rvs)
    empirical_cov = cov(rvs)
    results[experiment] = ExperimentHelpers.ExperimentResult(empirical_mean, empirical_cov, rvs)
end
## plot the output
#pygui(true)
layout_rows = length(init_distributions)*length(iter_methods)
layout_cols = length(n_iter_vals)

scale = 1.5
fig,ax = mplt.subplots(layout_rows,layout_cols,sharex="all",sharey="all",figsize = (scale*8,scale*6))


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

mplt.tight_layout()

#mplt.show()
mplt.savefig("unscaled_id_prior.pdf")

