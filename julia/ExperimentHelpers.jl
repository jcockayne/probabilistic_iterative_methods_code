module ExperimentHelpers
using Distributions
using ..IterativeMethods
export Experiment, ExperimentResult
struct Experiment
    Iterations::Int
    Method::IterativeMethods.StationaryIterativeMethod
    InitialDist::MultivariateDistribution
end
struct ExperimentResult
    Mean
    Cov
    Samples
end
end
