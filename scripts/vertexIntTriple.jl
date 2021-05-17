@fastmath @inline tripleToInt(i::UInt32, j::UInt32, k::UInt32, nB::UInt32, nB2::UInt32)::UInt32 = UInt32(i*nB2 + j*nB + k)
@fastmath @inline tripleToInt(i::UInt32, j::UInt32, k::UInt32, offset::UInt32, nB::UInt32, nB2::UInt32)::UInt32 = UInt32((i+offset)*nB2 + (j+offset)*nB + (k+offset))

tripleToInt(i,j,k, nB, nB2) = tripleToInt(UInt32(i), UInt32(j), UInt32(k), UInt32(nB), UInt32(nB2))
tripleToInt(i,j,k, offset,nB,nB2) = tripleToInt(UInt32(i), UInt32(j), UInt32(k), UInt32(offset), UInt32(nB), UInt32(nB2))
tripleToInt(i,j,k,offset,nB,nB2) = tripleToInt(i+offset,j+offset,k+offset,nB,nB2)

@fastmath @inline function intToTriple(::Type{T}, z::UInt32) where {T<:Integer}
    r,k = divrem(z,nB)
    i,j = divrem(r,nB)
    return (convert(T,i)-nBh,convert(T,j)-nBh,convert(T,k)-nBh)
end
intToTriple(z::UInt32) = intToTriple(Int64, z)
