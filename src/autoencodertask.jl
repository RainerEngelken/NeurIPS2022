function generateInputOutputOUAnalytical(B, T, seedInput, S, Iext, I1, tauS, dt)
"""
    # this solves the OU-process analytically incrementally
    #https://math.stackexchange.com/questions/345773/how-the-ornstein-uhlenbeck-process-can-be-considered-as-the-continuous-time-anal
"""
    t = 1:T # time steps
    s = zeros(Float32, T, S, B) # input
    target = zeros(Float32, T, S, B) # target output
    rng = Xoroshiro128Star(seedInput)
    Random.seed!(rng, seedInput)
    expfactorOU = exp(-dt * inv(tauS))
    stdfactorOU = Float32(I1 * sqrt.((1 .- exp.(-2dt * inv(tauS))) ./ 2tauS))
    for bi = 1:B #put in right format for batch-gradient descent with batch-size B
        for si = 1:S
            noisehere = stdfactorOU * randn(rng, Float32)
            for i = 2:T
                noisehere = randn(rng, Float32)
                if tauS == 0
                    s[i, si, bi] = I1 * noisehere
                else
                    s[i, si, bi] = expfactorOU * s[i-1, si, bi] + stdfactorOU * noisehere
                end
            end
        end
    end
    return s .+ Float32(Iext), s .+ Float32(Iext)
end
