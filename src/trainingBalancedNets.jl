# This script trains a vanilla RNN on a tracking one or multiple OU-signals
using Flux, PyPlot, Statistics, Bootstrap, LinearAlgebra, Random, DelimitedFiles
using RandomNumbers.Xorshifts, Distributed, Random
using Flux.Data: DataLoader
using Clustering
using ForwardDiff
using BSON: @load
using BSON: @save
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(4)

function trainCopy(N, Nepochs, B, S, T, seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom, plotStuff, tauS, g, gbar, Iext, I1, dt, sigmaNoise, phi, wsStd, wsMean, wrStd, wrMean, ActivityReg, resetDiagonal, subDir)
    @show N, Nepochs, B, S, T, seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom, plotStuff, tauS, g, gbar, Iext, I1, dt, sigmaNoise, phi, wsStd, wsMean, wrStd, wrMean, ActivityReg
    verbose = false
    phid(x) = ForwardDiff.derivative(phi, x)
    R = S # number of readouts
    onlyfinaltesterror = false
    cb_interval = 1000
    lr = Float32(lrIn) #learning rate
    nMeasure = 5
    tau = 1# just for plotting
    ParaString = "N=$N.Nepochs=$Nepochs.B=$B.S=$S.T=$T.seedIC=$seedIC.seedInput=$seedInput.seedNet=$seedNet.lrIn=$lrIn.beta1=$beta1.beta2=$beta2.IC=$IC.tauS=$tauS.g=$g.gbar=$gbar.Iext=$Iext.I1=$I1.dt=$dt.sigma=$sigmaNoise.ActivityReg=$ActivityReg.resetDiagonal=$resetDiagonal"
    if isfile("data-" * subDir * "-trainedPara/" * ParaString * "testError.dat")
        testError = readdlm("data-" * subDir * "-trainedPara/" * ParaString * "testError.dat")
        return testError[:]
    end
    s, rtarg = generateInputOutputOUAnalytical(B, T, seedInput, S, Iext, I1, tauS, dt)
    rmul!(rtarg, -inv(gbar))
    if startRandom
        Random.seed!(seedNet)
        ws = (wsStd * randn(Float32, N, S) + wsMean * ones(Float32, N, S))
        wr = wrStd * randn(Float32, R, N) + wrMean * ones(Float32, R, N)
        b = zeros(Float32, N)
        wsInit = copy(ws)
        wrInit = copy(wr)
        bInit = copy(b)
        if IC == 1
            J = Float32(g) * randn(Float32, N, N) / Float32(sqrt(N)) .+ Float32(gbar / N)
            resetDiagonal && (J .-= Diagonal(J))
        elseif IC == 2 #MIXED EI network, parameters from Van Vreeswijk, Sompolinsky 1996
            JEE = JIE = WE = 1.0
            JII = -1.8
            JEI = -2.0
            WI = 0.8
            J = Float32.([JEE/(N/2).+g*randn(div(N, 2), div(N, 2))/Float32(sqrt(N)) JEI/(N/2).+g*randn(div(N, 2), div(N, 2))/Float32(sqrt(N)); JIE/(N/2).+g*randn(div(N, 2), div(N, 2))/Float32(sqrt(N)) JII/(N/2).+g*randn(div(N, 2), div(N, 2))/Float32(sqrt(N))])
            plotStuff && figure(141)
            imshow(J)
            colorbar()
            figure()
            resetDiagonal && (J .-= Diagonal(J))
        else
            println("WARNING: NO IC defined")
        end
    else
        try
            println("loading ParaString:")
            println(ParaString)
            @load "data-" * subDir * "-trainedPara/" * ParaString * ".bson" ps opt
            resetDiagonal && (J .-= Diagonal(J))
            wsInit = copy(ws)
            wrInit = copy(wr)
            bInit = copy(b)
        catch

            println("stored topologies not available with correct size, generating random instead or failing")
            println("ParaString:")
            println(ParaString)
        end
    end

    function relaxIC(steps, seedIC)
        Random.seed!(seedIC)
        x = randn(Float32, N, size(s, 3))
        for i = 1:steps
            x += dt * (-x + J * phi.(x) .+ b .+ ws * ones(Float32, S, 1) * Float32(Iext) / S)
        end
        return x
    end


    function calcloss(s, rtarg)
        loss = 0
        x = copy(xInit) #+ sig*randn(Float32, N, size(s,3))
        for ti = 1:T
            x += dt * (-x + J * phi.(x) .+ b + ws * s[ti, :, :] / S) + sigmaNoise * randn(Float32, N, B) * sqrt(dt)
            if ti >= div(T, 5)
                r = wr * phi.(x) / N
                loss += sum(abs2, r - rtarg[ti, :, :]) + ActivityReg * sum(abs2, x) # loss is MSE normalized by batch size
            end
            #@show typeof(loss)
        end
        return loss / B / T / size(wr, 1) #+ norm(J)/10000 #+ norm(b)/1000  + norm(wr)/10000 # + norm(wr)/1000
    end

    function calclossReturnx(s, rtarg)
        loss = 0
        x = copy(xInit) #+ sig*randn(Float32, N, size(s,3))
        for ti = 1:T
            x += dt * (-x + J * phi.(x) .+ b + ws * s[ti, :, :] / S) + sigmaNoise * randn(Float32, N, B) * sqrt(dt)
        end
        return x
    end


    function visualLoss(x0, NTtest, s, rtarg)
        sidx = 1#rand(1:size(s, 3))
        x = x0
        loss = 0
        rAll = []
        xmeasure = []
        x = copy(xInit)[:, 1]
        for ti = 1:NTtest
            x += dt * (-x + J * phi.(x) .+ b + ws * s[ti, :, sidx] / S) + sigmaNoise * randn(Float32, N) * sqrt(dt)
            r = wr * phi.(x) / N
            push!(rAll, r)
            append!(xmeasure, x[1:nMeasure])
        end
        rAll2 = reduce(hcat, rAll)
        out = copy((rAll2)')
        rtargTest = rtarg[:, :, sidx]
        outH = 0.0f0
        subplot(242)
        plot(tau * (1:size(rtarg[:, :, sidx], 1)), rtarg[:, :, sidx], alpha=0.5, ":k")
        plot(tau * (1:size(out, 1)), out, alpha=0.5, "-r")
        xlabel("Time (steps)")
        legend(["target output", "actual output", "input", "input right", "go cue"], loc="lower right")
        boxoff()

        subplot(245)
        xmeasure = reshape(xmeasure, nMeasure, :)
        plot(tau * (1:size(xmeasure, 2)), xmeasure')
        xlabel("Time (steps)")
        ylabel("example h")
        boxoff()
        return loss
    end
    opt = ADAM(lr, (beta1, beta2))
    ps = (ws, J, wr)
    xinit = zeros(Float32, N, B)
    prevt = time()
    testError = Float32[]
    testErrorEpoch = Int[]
    global EI = 0
    xInit = relaxIC(round(Int, 100 / dt), seedIC)
    x = calclossReturnx(s, rtarg)[:, 1]
    evInit = eigvals(J * Diagonal(phid.(x)))
    meanDifferentAll = Float32[]
    meanSameAll = Float32[]
    meanAll = Float32[]
    Epochs = []
    for ei = 1:Nepochs
        # mod(ei, displayInterval) == 0 && print("Epoch: $ei Time:$(time() - prevt)", "\r")
        mod(ei, 10) == 0 && print(ei, "\r")
        if mod(ei, cb_interval) == 1
            !isdir("data-" * subDir * "-trainedParaEachEpochseedNet$seedNet") && mkdir("data-" * subDir * "-trainedParaEachEpochseedNet$seedNet")
            @save "data-" * subDir * "-trainedParaEachEpochseedNet$seedNet/" * ParaString * "ei=$ei." * "bson" ps
        end

        EI = ei
        Random.seed!(ei)
        s, rtarg = generateInputOutputOUAnalytical(B, T, seedInput + ei, S, Iext, I1, tauS, dt)#(B, T, seedInput, S, sigma,tauS,dt)

        resetDiagonal && (J .-= Diagonal(J))
        Flux.train!(calcloss, ps, DataLoader(s, rtarg; batchsize=B, shuffle=true), opt)
        if (!onlyfinaltesterror && mod(ei, cb_interval) == 1) || ei == Nepochs
            Random.seed!(ei + 1)
            xInit = relaxIC(round(Int, 100 / dt), seedIC + ei)
            xt = xinit #.+  * randn(Float32, N, B) # initial condition for testing
            s, rtarg = generateInputOutputOUAnalytical(B, T, seedInput + Nepochs + 1, S, Iext, I1, tauS, dt)#(B, T, seedInput, S, sigma,tauS,dt)
            resetDiagonal && (J .-= Diagonal(J))
            testErrorHere = calcloss(s, rtarg)
            push!(testError, testErrorHere)
            push!(testErrorEpoch, ei)
            terror = testError
            println("terror:", testError[end], " for " * ParaString * "\r")
        end
        if plotStuff && mod(ei, cb_interval) == 1

            clf()
            subplot(241)
            semilogy(testErrorEpoch, testError, ".-k")
            ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            xlabel("Epoch")
            ylabel("test error")
            boxoff()
            visualLoss(xinit, T, s, rtarg)
            figure(2)
            clf()
            subplot(221)
            semilogy(testErrorEpoch, testError, ".-")
            subplot(222)

            plot(ws)
            plot(wr')
            plot(b)
            plot(wsInit, ":")
            plot(wrInit', ":")
            plot(bInit, ":")
            legend(("ws", "wr", "b", "wsInit", "wrInit", "binit"))
            subplot(223)
            histdata = hcat([ws, wr', b, wsInit, wrInit', bInit]...)
            hist(histdata)
            subplot(224)
            hist(x)
            figure(1)

            subplot(243)
            x = calclossReturnx(s, rtarg)[:, 1]
            evAfter = eigvals(J * Diagonal(phid.(x)))
            plot(real(evInit), imag(evInit), ".k", alpha=0.5)
            plot(real(evAfter), imag(evAfter), ".", color=[0.78, 0.129, 0.867], alpha=0.5)
            plt.gcf().gca().add_artist(plt.Circle((0.0, 0.0), 1.0, fill=false, ls=":"))
            plot(ones(2), ylim(), ":", color="0.5")
            legend(["initial", "trained"])
            xlabel(L"real($\lambda_i$)")
            ylabel(L"imag($\lambda_i$)")
            boxoff()

            subplot(244)
            if S > 1
                cor11 = cor(ws[:, 1], wr[1, :])
                cor12 = cor(ws[:, 1], wr[2, :])
                #first plot large corr
                #if cor11 > cor12
                plot(ws[:, 1], wr[1, :], ".r")
                plot(ws[:, 2], wr[2, :], ".r")
                plot(ws[:, 2], wr[1, :], ".k")
                plot(ws[:, 1], wr[2, :], ".k")
                xlabel(L"w^{in}")
                ylabel(L"w^{out}")
                boxoff()
            end
            subplot(246)
            id1wr = sortperm(ws[:, 1])
            id = id1wr
            as = ifelse.(ws[:, 1] .> median(ws[:, 1]), 1, 2)
            #@time out = kmeans(ws',2;maxiter=200)
            out = kmeans(J, S; maxiter=200)
            as = assignments(out)
            id = sortperm(as)
            Jhere = J[id, id]#-Diagonal(J[id,id])
            nSame = 0
            nDifferent = 0
            sumSame = 0.0
            sumDifferent = 0.0
            for i = 1:100, j = 1:100
                if as[i] == as[j]
                    nSame += 1
                    sumSame += J[i, j]
                else
                    nDifferent += 1
                    sumDifferent += J[i, j]
                end
            end

            meanSame = sumSame / nSame
            meanDifferent = sumDifferent / nDifferent
            push!(meanDifferentAll, meanDifferent)
            push!(meanSameAll, meanSame)
            push!(meanAll, mean(J))
            push!(Epochs, ei)
            ve = maximum(abs.(extrema(Jhere)))
            imshow(Jhere, cmap=ColorMap("RdBu"), vmin=-ve, vmax=ve, origin="lower")
            colorbar(fraction=0.046, pad=0.04)
            suptitle(ParaString)
            tight_layout()
            draw()

            subplot(247)
            plot(Epochs, meanSameAll, ".-r")
            plot(Epochs, meanDifferentAll, ".-b")
            plot(Epochs, meanAll, ".-k")
            #legend("mean","difference")
            legend((L"\bar J^{same}_{ij}", L"\bar J^{different}_{ij}", L"\bar J_{ij}"), labelspacing=0.5, loc="best", bbox_to_anchor=(0.0, 0.1, 1, 1); frameon=false, handlelength=1.5)
            boxoff()
            ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            tight_layout()
            xlabel("Epoch")
            ylabel(L"J")

            subplot(248)

            plot(Epochs, abs.(meanAll / meanAll[1]), ".-k")
            ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

            xlabel("Epoch")
            ylabel(L"balance $b$")
            boxoff()
        end
    end
    !isdir("data-" * subDir * "-trainedParaEachEpochseedNet$seedNet") && mkdir("data-" * subDir * "-trainedParaEachEpochseedNet$seedNet")
    @save "data-" * subDir * "-trainedParaEachEpochseedNet$seedNet/" * ParaString * "ei=$Nepochs." * "bson" ps

    println("train time: ", time() - prevt)
    @show(calcloss(s, rtarg))
    println("training successful, saving weights")

    !isdir("data-" * subDir * "-trainedPara") && mkdir("data-" * subDir * "-trainedPara")
    @save "data-" * subDir * "-trainedPara/" * ParaString * ".bson" ps opt
    plotStuff && savefig("data-" * subDir * "-trainedPara/" * ParaString * "resultFig.png")
    writedlm("data-" * subDir * "-trainedPara/" * ParaString * "testError.dat", testError)

    return testError
end

if !isinteractive()
    funseed(seed)
end
