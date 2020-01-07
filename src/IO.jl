
function readConfig(file)
    tml = TOML.parsefile(file)
    model = 
    sim = tml["Simulation"]
    model = ModelParameters(tml["Model"]["U"], 
                            tml["Model"]["mu"], 
                            tml["Model"]["beta"], 
                            tml["Model"]["nden"],
                            tml["Model"]["Dimensions"])
    sim = SimulationParameters(tml["Simulation"]["nFermFreq"],
                               tml["Simulation"]["nBoseFreq"],
                               tml["Simulation"]["shift"],
                               tml["Simulation"]["Nk"],
                               tml["Simulation"]["NkInt"],
                               tml["Simulation"]["Nq"],
                               tml["Simulation"]["inputVars"])
    return (model, sim)
end

function readFortranSymmGF!(GF, filename; storedInverse, storeFull=false)
    GFString = open(filename, "r") do f
        readlines(f)
    end

    if size(GFString, 1)*(1 + 1*storeFull) < size(GF, 1)
        throw(BoundsError("nFermFreq in simulation parameters too large!"))
    end
    
    tmp = parse.(Float64,hcat(split.(GFString)...)[2:end,:]) # Construct a 2xN array of floats (re,im as 1st index)
    tmp = tmp[1,:] .+ tmp[2,:].*1im

    if storedInverse
        tmp = 1 ./ tmp
    end
    
    N = size(GF,1)
    if storeFull
        if N % 2 == 1
            throw(BoundsError("Use even size for symmetric storage!"))
        end
        NH = Int(N/2)
        GF[1:NH] = reverse(conj.(tmp[1:NH]))
        GF[NH+1:N] = tmp[1:NH]
    else
        GF .= tmp[1:N]
    end
    return GF
end

function readFortran3FreqFile(filename, sign = 1.0, freqInteger = true)
    InString = open(filename, "r") do f
        readlines(f)
    end
    InArr = sign .* parse.(Float64,hcat(split.(InString)...)[2:end,:])
    N = Int(sqrt(size(InArr,2)))
    if freqInteger
        ωₙ = Int(parse(Float64, split(InString[1])[1]))
        middleFreqBox = Int64.([minimum(InArr[1,:]),maximum(InArr[1,:])])
        innerFreqBox  = Int64.([minimum(InArr[2,:]),maximum(InArr[2,:])])
        freqBox = permutedims([middleFreqBox innerFreqBox], [2,1])
    else
        print("\rWarning, non integer frequencies in "*filename*" ignored!")
        ωₙ = 0
        freqBox = []
    end

    InCol1  = reshape(InArr[3,:] .+ InArr[4,:].*1im, (N, N))
    InCol2  = reshape(InArr[5,:] .+ InArr[6,:].*1im, (N, N))
    return ωₙ, freqBox, InCol1, InCol2
end

function readFortranΓ(dirName::String)
    files = readdir(dirName)
    ωₙ, freqBox, Γcharge, Γspin = readFortran3FreqFile(dirName * "/" * files[1], -1.0)
    ω_min = ωₙ
    ω_max = ωₙ

    for file in files[2:end]
        ωₙ, _, Γcharge_new, Γspin_new = readFortran3FreqFile(dirName * "/" * file, -1.0)
        ω_min = if ωₙ < ω_min ωₙ else ω_min end
        ω_max = if ωₙ > ω_max ωₙ else ω_max end
        Γcharge = cat(Γcharge, Γcharge_new, dims=3)
        Γspin = cat(Γspin, Γspin_new, dims=3)
    end
    freqBox = [ω_min ω_max; freqBox[1,1] freqBox[1,2]; freqBox[2,1] freqBox[2,2]]
    Γcharge = permutedims(Γcharge, [3,1,2])
    Γspin   = permutedims(Γspin, [3,1,2])
    return freqBox, Γcharge, Γspin
end

"""
    Returns χ_DMFT[ω, ν, ν']
"""
function readFortranχDMFT(dirName::String)
    files = readdir(dirName)
    _, _, χup, χdown = readFortran3FreqFile(dirName * "/" * files[1], 1.0, false)

    for file in files[2:end]
        _, _, χup_new, χdown_new = readFortran3FreqFile(dirName * "/" * file, 1.0, false)
        χup = cat(χup, χup_new, dims=3)
        χdown = cat(χdown, χdown_new, dims=3)
    end
    χup = permutedims(χup, [3,1,2])
    χdown   = permutedims(χdown, [3,1,2])
    χCharge = χup .+ χdown
    χSpin   = χup .- χdown
    return χCharge, χSpin 
end

function writeFortranΓ(dirName::String, fileName::String, simParams, inCol1, inCol2)
    simParams.n_iν+simParams.n_iω
    if !isdir(dirName)
        mkdir(dirName)
    end
    for (ωi,ωₙ) in enumerate(-simParams.n_iω:simParams.n_iω)
        filename = dirName * "/" * fileName * lpad(ωi-1,3,"0")
        open(filename, "w") do f
            for (νi,νₙ) in enumerate(-simParams.n_iν:(simParams.n_iν-1))
                for (ν2i,ν2ₙ) in enumerate(-simParams.n_iν:(simParams.n_iν-1))
                    @printf(f, "%8d %8d %8d %18.10f %18.10f %18.10f %18.10f\n", ωₙ, νₙ, ν2ₙ,
                            real(inCol1[ωi, νi, ν2i]), imag(inCol1[ωi, νi, ν2i]), 
                            real(inCol2[ωi, νi, ν2i]), imag(inCol2[ωi, νi, ν2i]))
                end
            end
        end
    end
end

# Cut Γ or χ to necessary size
function reduce_3Freq(inp, freqBox, simParams)
    n_iω = simParams.n_iω
    n_iν = simParams.n_iν
    start_ωₙ =  -(freqBox[1,1] + n_iω) + 1
    start_νₙ =  -(freqBox[2,1] + n_iν) + 2
    # This selects the range of [-ωₙ, ωₙ] and [-νₙ, νₙ-1] from gamma
    return inp[start_ωₙ:(start_ωₙ+2*n_iω), start_νₙ:(start_νₙ + 2*n_iν - 1), start_νₙ:(start_νₙ +2*n_iν - 1)]
end
reduce_3Freq!(inp, freqBox, simParams) = inp = reduce_3Freq(inp, freqBox, simParams)


function print_chi_bubble(qList, res, simParams)
    for j in 1:size(res,1)
        print(" ========== ω = $(j-(simParams.n_iω + 1)) =============== \n")
        for k in 1:size(res,2)
            print(" ---------- ν = $(k-1) -------------- \n")
            for (qi,q) in enumerate(qList)
                @printf("   q = (%.2f,%.2f): %.2f + %.2fi\n", q[1],q[2], real(res[j, k, qi]), imag(res[j, k, qi]))
            end
        end
    end
end

function convert_from_fortran(simParams)
    g0 = zeros(Complex{Float64}, simParams.n_iν+simParams.n_iω)
    gImp = zeros(Complex{Float64},  simParams.n_iν+simParams.n_iω)
    readFortranSymmGF!(g0, dir*"g0mand", storedInverse=true)
    readFortranSymmGF!(gImp, dir*"gm_wim", storedInverse=false)
    freqBox, Γcharge, Γspin = readFortranΓ(dir*"gamma_dir")
    χDMFTCharge, χDMFTSpin = readFortranχDMFT(dir*"chi_dir")
    Γcharge = -1.0 .* reduce_3Freq(Γcharge, freqBox, simParams)
    Γspin = -1.0 .* reduce_3Freq(Γspin, freqBox, simParams)
    χDMFTCharge = reduce_3Freq(χDMFTCharge, freqBox, simParams)
    χDMFTSpin = reduce_3Freq(χDMFTSpin, freqBox, simParams)
    writeFortranΓ("fortran_out", "gamma", simParams, Γcharge, Γspin)
    writeFortranΓ("fortran_out", "chi", simParams, 0.5 .* (χDMFTCharge .+ χDMFTSpin), 0.5 .* (χDMFTCharge .- χDMFTSpin))
    save("vars.jld", "g0", g0, "gImp", gImp, "GammaCharge", Γcharge, "GammaSpin", Γspin,
         "chiDMFTCharge", χDMFTCharge, "chiDMFTSpin", χDMFTSpin, "freqBox", freqBox)
end
