using NPZ
using LinearAlgebra
using Dates

# projection function onto the design space
function proj(x, design)
    n = Int((length(x)+1)/2)
    a = x[1:n]
    b = x[(n+1):(2*n-1)]
    b = [b; 1-sum(b)]
    
    a = max.(a,design[1])
    a = min.(a,design[2])
    
    u = sort(b, rev=true)
    j = n
    while true
        if u[j] + 1/j*(1-cumsum(u)[j]) > 0
            break
        end
        j -= 1
    end
    lambda = 1/j*(1-cumsum(u)[j])
    b = max.(b .+ lambda,0)
    b = b[1:(n-1)]
    
    return [a;b]
end

#Information matrix for quadratic logistic model
function infmat(x)
    n = Int((length(x)+1)/2)
    a = x[1:n]
    b = x[(n+1):(2*n-1)]
    b = [b; 1-sum(b)]
    p = length(theta)
    mat = zeros(p,p)
    for i in 1:n
        c = exp(theta[1]+theta[2]*(a[i]-theta[3])^2)
        f = [1,(a[i]-theta[3])^2,2*theta[2]*(theta[3]-a[i]) ]
        mat = mat + b[i] * c/(1+c)^2 * f*f'
    end
    return mat
end

function diter(k, nptc, niter, design)
    #initialize
    #srand(1992)
    tau = 0.9:(-0.5/niter):0.4
    v = zeros(nptc, 2k-1)
    x = (design[2]-design[1])*rand(nptc*k) .+ design[1]
    x = reshape(x,nptc,k)
    w = rand(nptc*(k-1))
    w = reshape(w,nptc,k-1)
    ptc = hcat(x,w)
    for j in 1:nptc
        ptc[j,:] = proj(ptc[j,:],design)
    end
    pbest = ptc
    pbesth = zeros(nptc)
    for i in 1:nptc
        pbesth[i] = det(infmat(pbest[i,:]))
    end
    gbesth = findmax(pbesth)[1]
    gbest = pbest[findmax(pbesth)[2],:]

    #iterate
    for i in 1:niter # iteration number
        for j in 1:nptc # particle number
            v[j,:] = tau[i] .*v[j,:] + 2 .*rand(2k-1) .*(pbest[j,:]-ptc[j,:]) + 2 .*rand(2k-1) .*(gbest-ptc[j,:])
            ptc[j,:] = ptc[j,:] + v[j,:]
            ptc[j,:] = proj(ptc[j,:],design)
            fit = det(infmat(ptc[j,:]))
            if fit > gbesth
                gbest = pbest[j,:] = ptc[j,:]
                gbesth = pbesth[j] = fit
            elseif fit > pbesth[j]
                pbest[j,:] = ptc[j,:]
                pbesth[j] = fit
            end
        end
    end
    supp = gbest[1:k]
    prob = gbest[(k+1):2k-1]
    prob = [prob; 1-sum(prob)]
    return [supp, prob, gbesth]
end

theta = [2,3,0]

diter(3, 128, 150, [-3,1])

nptc = [64, 128, 256, 512, 1024]
niter = [50, 100, 200, 500, 1000]
nsim = 500

result = zeros(5,5)
result2 = zeros(5,5)
timeresult = zeros(5,5)
for i in 1:5
    for j in 1:5
        temp = 0
        temp2 = 0
        t1 = time()
        for k in 1:nsim
            opt = diter(3, nptc[i], niter[j], [-3,1])[3]
            if opt > 0.99 * 5.69464649495885e-5
                temp += 1
            end
            if opt > 0.995 * 5.69464649495885e-5
                temp2 += 1
            end
        end
        t2 = time()
        result[i,j] = temp/nsim
        result2[i,j] = temp2/nsim
        timeresult[i,j] = (t2-t1)/nsim
        npzwrite("quad3stab.npy",result)
        npzwrite("quad3stab2.npy",result2)
        npzwrite("quad3time.npy",timeresult)
        println("simulation saved")
    end
end