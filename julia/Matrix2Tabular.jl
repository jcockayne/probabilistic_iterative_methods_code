module Matrix2Tabular

export float2sci, matrix2tabular

function float2sci(x::Float64, figs::Int64=2, dollars::Bool=false)
    power = convert(Int,round(log10(abs(x)),RoundDown))
    if 0 <= power <= 1
        figs = max(figs-power,2)
        x = string(round(x,digits=figs))
    else
        x = round(x/(10.0^power),digits=figs)
        x = string(x,"\\times 10^{",power,"}")
    end
    
    if dollars
        return string(" \$",x,"\$ ")
    else
        return " "*x*" "
    end
end

function matrix2tabular(A::Matrix{Float64}, figs::Int64=2, dollars::Bool=false)
    M,N = size(A)

    outstring = " "

    for m = 1:M
        for n = 1:N
            outstring *= float2sci(A[m,n], figs, dollars)
            if n == N
                outstring *= "\\\\ \n "
            end
        end
    end

    return outstring
end
end
