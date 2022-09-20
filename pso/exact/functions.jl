using LinearAlgebra
using Random

function psox(N, f, dim, design, niter, nptc)
    
    # projection function onto the design space
    function proj(x)
        x = max.(x,design[:,1]')
        x = min.(x,design[:,2]')
        return(x)
    end
    
    function fitness(x)
        val = f(x[1,:])*f(x[1,:])'
        if N >= 2
            for i in 2:N
                val = val + f(x[i,:])*f(x[i,:])'
            end
        end
        val = 1/N * val
        p = size(val)[1]
        fit = max(0, det(val))
        return( fit^(1/p) )
    end
        
    #initialize
    path = zeros(niter)
    tau = 0.9:(-0.5/niter):0.4
    v = zeros(nptc,N,dim)
    ptc = rand(nptc,N,dim)
    for j in 1:dim
        ptc[:,:,j] =  (design[j,2]-design[j,1]) * ptc[:,:,j] .+ design[j,1]
    end    
    pbest = ptc
    pbesth = zeros(nptc)
    for j in 1:nptc
        pbesth[j] = fitness(ptc[j,:,:])
    end
    gbesth = findmax(pbesth)[1]
    gbest = pbest[findmax(pbesth)[2],:,:]
    
    #iterate
    for i in 1:niter # iteration number
            g1 = rand(N,dim)
            g2 = rand(N,dim)
        for j in 1:nptc # particle number
            v[j,:,:] = tau[i] * v[j,:,:] + 2 * g1 .* (pbest[j,:,:]-ptc[j,:,:]) + 2 * g2 .* (gbest-ptc[j,:,:])
            ptc[j,:,:] = ptc[j,:,:] + v[j,:,:]
            ptc[j,:,:] = proj(ptc[j,:,:])
            fit = fitness(ptc[j,:,:])
            if fit > gbesth
                gbest = pbest[j,:,:] = ptc[j,:,:]
                gbesth = pbesth[j] = fit
            elseif fit > pbesth[j]
                pbest[j,:,:] = ptc[j,:,:]
                pbesth[j] = fit
            end
        end
        path[i] = gbesth
        if i % 100 == 0
            println(".")
            else 
            print(".")
        end   
    end
    
    println("PSO iteration complete")
    return [gbesth, gbest, path]
    
end