using LinearAlgebra

include("MMD.jl")
include("IterativeMethods.jl")

N = 50
samples = 1000
iterations = 100

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

gauss_ker(x,y) = exp(-norm(x-y)^2/2)

print(MMD.mmd(xInput,xOutput,gauss_ker))
