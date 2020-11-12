using LinearAlgebra
using Statistics

using Distributed
rmprocs(procs()[2:end])
addprocs(7)

include("MMD.jl")
include("IterativeMethods.jl")
include("BayesCG.jl")

N = 440
samples = 1000
iterations = 3

A = randn(N,N); A = A'*A

xInput = randn(N,samples)
guesses = randn(N,samples)
xOutput = zeros(N,samples)

for i = 1:samples
    xTrue = randn(N)
    b = A*xTrue
    richardson = IterativeMethods.Richardson(A,b,IterativeMethods.Optimal)
    xOutput[:,i] = IterativeMethods.apply(richardson,guesses[:,i],iterations)
end

function gauss_ker(x::Array{Float64,1},y::Array{Float64,1})
    return exp.(-0.5.*(norm(x-y).^2))
end

print(@time gk2 = MMD.mmd_kernel_factory(xInput,xOutput))

print("Time Test \n")

print("Richardson \n")
print((@time MMD.mmd(xInput,xOutput,gauss_ker)),"\n")
print((@time MMD.mmd(xInput,xOutput,gk2)),"\n")

print("Richardson \n")
print((@time MMD.mmd_p(xInput,xOutput,gk2)),"\n")


print("Bootstrap Richardson \n")
b_mmd = @time MMD.bootstrap_mmd(xInput[:,1:100],xOutput[:,1:100],gk2,1000);

xOutputCG = zeros(N,samples)
it = convert(Int64,floor(N*.9))
InvA = inv(A)
for i = 1:samples
    xTrue = randn(N)
    b = A*xTrue
    xOutputCG[:,i],_ = BayesCG.bayescg(A,b,guesses[:,i],InvA,it,1)
end

print("BayesCG \n")
print(MMD.mmd(xInput,xOutputCG,gauss_ker),"\n")


