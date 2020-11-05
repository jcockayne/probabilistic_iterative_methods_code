using LinearAlgebra

include("MMD.jl")
include("IterativeMethods.jl")
include("BayesCG.jl")

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

print("Richardson \n")
print(MMD.mmd(xInput,xOutput,gauss_ker),"\n")

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


