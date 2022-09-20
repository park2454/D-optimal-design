# using Convex, MosekTools, LinearAlgebra
using Convex, Mosek, LinearAlgebra

struct example
    p
    design
    f
end

function misocp(example,ngrid,supp,lambda)
    p = example.p
    n = ngrid
    aux = 0
    temp = p
    power = 0
    while temp > 1
        aux += div(temp+1,2)
        temp = div(temp+1,2)
        power += 1
    end
    aux = Int(aux)
    
    design = example.design
    x = design[1]:(design[2]-design[1])/n:design[2]
    h = zeros(n+1,p)
    for i in 1:n+1
        h[i,:] = example.f(x[i])
    end

    v = Variable(n+1,p)
    t = Variable(n+1,p)
    L = Variable(p,p)
    N = Variable(n+1, :Int)
    z = Variable(aux)
    prob = maximize(z[aux])
    if lambda > 0
        c = Variable(n+1, :Bin)
        prob.constraints += N ./ supp <= c
        prob.constraints += sum(c) <= lambda
    end
    
    prob.constraints += h' * v == L

    if p > 1
        for j = 1:(p-1)
            prob.constraints += L[j,(j+1):p] == 0
        end
    end

    for i = 1:n+1
        for j = 1:p
            prob.constraints += norm([2v[i,j]; t[i,j]-N[i]/supp]) <= t[i,j]+N[i]/supp
        end
    end

    for j = 1:p
        prob.constraints += sum(t[:,j]) <= L[j,j]
    end

    prob.constraints += N >= 0
    #prob.constraints += norm(N) == supp
    prob.constraints += sum(N) == supp
    
    prob.constraints += z >= 0 
    
    ntemp = temp = p
    loc = 0
    
    while temp > 1
        
        if temp % 2 == 0
            ntemp = Int(temp/2)
            for i in 1:ntemp
                prob.constraints += norm([2z[i]; L[2i-1,2i-1] - L[2i,2i]]) <= L[2i-1,2i-1] + L[2i,2i]
            end
        else
            ntemp = Int((temp + 1)/2)
            for i in 1:(ntemp - 1)
                prob.constraints += norm([2z[i]; L[2i-1,2i-1] - L[2i,2i]]) <= L[2i-1,2i-1] + L[2i,2i]
            end
            prob.constraints += norm([2z[ntemp]; L[2ntemp-1,2ntemp-1] - 1]) <= L[2ntemp-1,2ntemp-1] + 1
        end
        temp = ntemp
        
        while temp > 1
            
            if temp % 2 == 0
                ntemp = Int(temp/2)
                for i in 1:ntemp
                    prob.constraints += norm([2z[loc+temp+i]; z[loc+2i-1] - z[loc+2i]]) <=  z[loc+2i-1] + z[loc+2i]
                end
                loc = Int(temp + loc)
                temp = ntemp
            else
                ntemp = Int((temp + 1)/2)
                for i in 1:(ntemp - 1)
                    prob.constraints += norm([2z[loc+temp+i]; z[loc+2i-1] - z[loc+2i]]) <=  z[loc+2i-1] + z[loc+2i]
                end
                prob.constraints += norm([2z[loc+temp+ntemp]; z[loc+2ntemp-1] - 1]) <=  z[loc+2ntemp-1] + 1
                loc = Int(temp + loc)
                temp = ntemp
            end
            
        end
        
    end
    
    #solve!(prob, Mosek.Optimizer(), verbose=true)
    solve!(prob, MosekSolver())
    
    Nsol = N.value[N.value.>0]
    xsol = x[getindex.(findall(x->x>0, N.value),1)]
    optsol = (z.value[aux])^(2^power/p)
    
    return [Nsol xsol], optsol
    
end