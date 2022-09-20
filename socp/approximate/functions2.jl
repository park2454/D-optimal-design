using Convex, Mosek, LinearAlgebra

struct example
    p
    design
    f
end

function misocp(ngrid,supp,example)
    p = example.p
    n = ngrid
    aux = 0
    temp = p
    while temp > 1
        if temp%2 == 0
            aux += div(temp,2)
            temp = temp/2
            else
            aux += div(temp+1,2)
            temp = (temp+1)/2
        end
    end
    aux = Int(aux)
    
    design = example.design
    x = design[1]:(design[2]-design[1])/n:design[2]
    global x2 = zeros((n+1)^2,2)
    h = zeros((n+1)^2,p)
    for i in 1:n+1
        for j in 1:n+1
            x2[(n+1)*(i-1)+j,:] = [x[i],x[j]]
            h[(n+1)*(i-1)+j,:] = example.f(x2[(n+1)*(i-1)+j,:])
        end
    end

    global v = Variable((n+1)^2,p)
    global t = Variable((n+1)^2,p)
    global L = Variable(p,p)
    global w = Variable((n+1)^2)
    global z = Variable(aux)
    if supp > 0
        global y = Variable((n+1)^2, :Int)
    end

    global prob = maximize(z[aux])

    prob.constraints += h' * v == L

    if p > 1
        for j = 1:(p-1)
            prob.constraints += L[j,(j+1):p] == 0
        end
    end

    for i = 1:n+1
        for j = 1:p
            prob.constraints += norm([2v[i,j]; t[i,j]-w[i]]) <= t[i,j]+w[i]
        end
    end

    for j = 1:p
        prob.constraints += sum(t[:,j]) <= L[j,j]
    end

    prob.constraints += w >= 0
    prob.constraints += sum(w) == 1
    
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
    
    if supp > 0
        prob.constraints += y >= 0
        prob.constraints += y <= 1
        prob.constraints += sum(y) <= supp
        prob.constraints += w-y <= 0
    end
end