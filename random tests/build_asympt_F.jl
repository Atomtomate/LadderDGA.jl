using JLD2
using OffsetArrays
f = jldopen("test/test_data/b1u2_ED_data.jld2", "r")

nBose = 5
nFermi = 5
shift = 1
wrange = -nBose:nBose
vrange = -nFermi:(nFermi-1)

#function Freq_to_OneToIndex(ωn::Int, νn::Int, νpn::Int, shift::Union{Bool,Int}, nBose::Int, nFermi::Int)
#    (ωn+nBose+1,νn+nFermi+1+trunc(Int, shift*ωn/2), νpn+nFermi+1+trunc(Int, shift*ωn/2))
#end
#ωi,νi,νpi = Freq_to_OneToIndex(wn, vn, vpn, shift, nBose, nFermi)
#ωi_ph,νi_ph,νpi_ph = Freq_to_OneToIndex(wn - vn - vpn - 1, vn, vpn, shift, nBose, nFermi)



Fd_asym = OffsetArray(zeros(ComplexF64, length(vrange), length(vrange), length(wrange)), vrange, vrange, wrange);
Fm_asym = OffsetArray(zeros(ComplexF64, length(vrange), length(vrange), length(wrange)), vrange, vrange, wrange);
U = f["U"]
xch = f["χ_ch_asympt"]
xsp = f["χ_sp_asympt"]
xpp = f["χ_pp_asympt"]

for (wi,wn) in enumerate(wrange)
    for (vi,vn) in enumerate(vrange)
        for (vpi,vpn) in enumerate(vrange)
            nu_m_nup_i = vpn - vn < 0 ? -(vpn - vn) + 1 : vpn - vn + 1
            nup_m_ni_i = vn - vpn < 0 ? -(vn - vpn) + 1 : vn - vpn + 1
            nu_p_nup_p_w = (vn + vpn + 1) + wn < 0 ? -((vn + vpn + 1) + wn) + 1 : (vn + vpn + 1) + wn + 1
            Fd_asym[vn,vpn,wn] =  U + ((U^2)/2) * xch[nup_m_ni_i] + (3*(U^2)/2) * xsp[nup_m_ni_i] - (U^2) * xpp[nu_p_nup_p_w]
            Fm_asym[vn,vpn,wn] = -U + ((U^2)/2) * xch[nup_m_ni_i] - (1*(U^2)/2) * xsp[nup_m_ni_i] + (U^2) * xpp[nu_p_nup_p_w]
        end
    end
end

heatmap(real(Fd_asym[:,:,0].parent))