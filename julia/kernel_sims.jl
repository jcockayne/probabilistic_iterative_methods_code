using LinearAlgebra
using Distributions
using PDMats
using PyPlot

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
    MvNormal(zeros(d), PDiagMat(1/d*ones(d))),
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
pygui(true)
layout_rows = length(init_distributions)*length(iter_methods)
layout_cols = length(n_iter_vals)
fig = figure(figsize=(10, 15), constrained_layout=false)
super_ax = fig.add_gridspec(2, 2, width_ratios=[0.01, 1], height_ratios=[0.01,1])
subgrid = super_ax[2,2].subgridspec(layout_rows, layout_cols)

col_label_subgrid = super_ax[1,2].subgridspec(1, layout_cols)
row_label_subgrid = super_ax[2,1].subgridspec(layout_rows, 1)

ax = Matrix(undef, layout_rows, layout_cols)
for i = 1:layout_rows
    for j = 1:layout_cols
        if i > 1 && j > 1
            ax[i,j] = fig.add_subplot(subgrid[i,j], sharex=ax[1,1], sharey=ax[1,1])
        else
            ax[i,j] = fig.add_subplot(subgrid[i,j])
        end
    end
end

function experiment_to_label(experiment::ExperimentHelpers.Experiment) :: String

    method_name = experiment.Method.Name
    method_stepsize = experiment.Method.Stepsize == IterativeMethods.Optimal ? "(i)" : "(ii)"
    method_prior = init_distribution_labels[experiment.InitialDist]
    "($method_name, $method_stepsize, $method_prior)"
end
for (col_ix, n_iter) = enumerate(n_iter_vals)
    for (row_ix, (dist, method)) = enumerate(Iterators.product(init_distributions, iter_methods))
        experiment = ExperimentHelpers.Experiment(n_iter, method, dist)
        result = results[experiment]
        ax[row_ix, col_ix].plot(Y, g_direct_solution)
        ax[row_ix, col_ix].plot(Y, g(Y, Z, result.Samples[:, 1:5]), color=:black, alpha=0.2, linewidth=1)

        if col_ix == 1
            label = experiment_to_label(experiment)
            row_label_ax = fig.add_subplot(row_label_subgrid[row_ix])
            row_label_ax.annotate(label, (0,0), xytext=(0.5, 0.5), textcoords="axes fraction", ha="center", va="center", rotation=90)
            row_label_ax.axis("off")
        end
    end
    #ax[1, col_ix+1].annotate("\$m=$n_iter\$", (0,0), xytext=(0.5, 0.5), textcoords="axes fraction", ha="center", va="center", fontsize=14)
    #ax[1, col_ix+1].axis("off")
end

tight_layout()
