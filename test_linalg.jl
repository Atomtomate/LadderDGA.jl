A = randn(N,N);
Ai = similar(A)
d = collect(Diagonal(1:N));
di = collect(1:N);

function manual_inv!(Ainv::Matrix{Float64}, ALU::LU{Float64, Matrix{Float64}}, A::Matrix{Float64})
    A_fak = lu(A);
    Id = collect(I)
    for i in 1:size(A,2)
        for j in 1:size(A,1)
            @inbounds Ainv[j,:] .= 
        end
    end
end

t1 = inv(A);
t2 = A \ d;



@time t2 = inv(A);
@time t2 = A \ d;
