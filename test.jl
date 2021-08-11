using Distributed
addprocs(3)
A = randn(100,100);
for i in 1:100000
 A .= A.^2
end
@fetchfrom 2 A
