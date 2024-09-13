using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2

cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);

kGridLoc = gen_kGrid(kGridsStr[1][1], 1)
bubbleLoc = calc_bubble(:local, lDGAhelper.gImp, lDGAhelper.gImp, kGridLoc, mP, sP)

ω₀Ind = ceil(Int, size(lDGAhelper.χDMFT_d,3)/2)
χd_inv_ev, χd_inv_evec = eigen(inv(lDGAhelper.χDMFT_d[:,:,ω₀Ind]))
bubble_diff = BubbleDiff(bubble, bubbleLoc)
