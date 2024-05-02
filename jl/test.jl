using Printf, LinearAlgebra
for i=10:10:800
    a = rand(Float32, i,i)
    b = rand(Float32, i,i)
    @time c = a*b
end
