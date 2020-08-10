
function loadFortranData(dir)

    Nν, Nω = parse.(Int64, split(readlines(dir * "/ladderDGA.in")[4]))[1:2]
    Nq_x, NInt, Nk = parse.(Int64, split(readlines(dir * "/ladderDGA.in")[6]))
    Nq = floor(Int64,(Nq_x+2)*(Nq_x+1)*Nq_x/6)
    bubble =  Array{ComplexF64}(undef, 2*Nω+1, 2*Nν, Nq)
    χch =  Array{ComplexF64}(undef, 2*Nω+1, Nq)
    χsp =  Array{ComplexF64}(undef, 2*Nω+1, Nq)
    trilexch = Array{ComplexF64}(undef, 2*Nω+1, 2*Nν, Nq)
    trilexsp = Array{ComplexF64}(undef, 2*Nω+1, 2*Nν, Nq)
    Σ = Array{ComplexF64}(undef, 2*Nν, Nk)
    for ωi in 0:2*Nω
        istr = lpad(ωi,3,"0")
        f = FortranFile(String(dir * "/chi_bubble_raw/chi" * istr), "r")
        bubble[ωi+1,:,:] = read(f, (ComplexF64, size(bubble)[2:end]...))
        close(f)
        f = FortranFile(dir * "/chich_omega_raw/chi" * istr, "r")
        χch[ωi+1,:] = read(f, (ComplexF64, size(χch)[2]))
        close(f)
        f = FortranFile(dir * "/chisp_omega_raw/chi" * istr, "r")
        χsp[ωi+1,:] = read(f, (ComplexF64, size(χsp)[2]))
        close(f)
        f = FortranFile(dir * "/trilexch_omega_raw/tri" * istr, "r")
        trilexch[ωi+1,:,:] = read(f, (ComplexF64, size(trilexch)[2:end]...))
        close(f)
        f = FortranFile(dir * "/trilexsp_omega_raw/tri" * istr, "r")
        trilexsp[ωi+1,:,:] = read(f, (ComplexF64, size(trilexsp)[2:end]...))
        close(f)
    end
    bubble = permutedims(bubble, [1,3,2])
    trilexsp = permutedims(trilexsp, [1,3,2])
    trilexch = permutedims(trilexch, [1,3,2])
    f = FortranFile(dir * "/SelfE.dat", "r")
    Σ = read(f, (ComplexF32, size(Σ)...))
    close(f)
    return [bubble, χch, χsp, trilexch, trilexsp, Σ]
end
