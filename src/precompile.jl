function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    for n = 1:3, T in (Float32, Float64, ComplexF32, ComplexF64), D in (UnitRange{Int}, Vector{Int}, Int)
        #TODO: example from FFTW.jl
        #@assert precompile(Tuple{typeof(fft),Array{T,n},D})
    end
end
