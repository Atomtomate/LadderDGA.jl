using Dispersions
using OffsetArrays
using FFTW

qi_0 = 1
qi_π = length(kG.kMult)
ωi = sP.n_iω+1

f_d(k) = cos(k[1]) - cos(k[2])
function χ0_inv(G, kG, mP, sP)
    g_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    g_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    G_νn = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...) 
    ff = expandKArr(kG, kG.ϵkGrid) .^ 2
    for νn in sP.fft_range
        G_νn  = νn < 0 ? expandKArr(kG, conj(G[:,-νn-1].parent)) : expandKArr(kG, G[:,νn].parent)
        g_fft[:,νn] .= fft(ff ./ G_νn)[:]
        g_rfft[:,νn] .= fft(reverse(1 ./ G_νn))[:]
    end
    g_fft, g_rfft
    res = sum(calc_bubble(:DMFT, g_fft, g_rfft, kG, mP, sP).data[:,sP.n_iν_shell+1:end-sP.n_iν_shell,:], dims=2)[:,1,:] / mP.β
    return res
end
