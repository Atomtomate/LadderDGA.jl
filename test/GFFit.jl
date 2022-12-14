# @testset "build fν max" begin
#     @test all(LadderDGA.build_fνmax_fast([4,3,2,1,2,3,4],2) .== [11, 19])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],3) .== [1, 5, 11])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],2) .== [5, 11])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],1) .== [11])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],3) .== [2, 6, 12])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],2) .== [6, 12])
#     @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],1) .== [12])
#     a = [3 3 3 3 3; 3 2 2 2 3; 3 2 1 2 3; 3 2 2 2 3; 3 3 3 3 3]
#     test = [1, 1+8*2, 1+8*2+16*3]
#     @test all(LadderDGA.build_fνmax_fast(a, 3) .== test)
#     @test all(LadderDGA.build_fνmax_fast(a, 2) .== test[2:3])
#     @test all(LadderDGA.build_fνmax_fast(a, 1) .== test[3])
#     cache = Array{Float64,1}(undef,5)
#     LadderDGA.build_fνmax_fast!(cache, ones(Float64,200), 5, 196)
#     cache_c = Array{ComplexF64,1}(undef,5)
#     LadderDGA.build_fνmax_fast!(cache, ones(ComplexF64,200), 5, 196)
#     @test all(cache .≈ [192.0, 194.0, 196.0, 198.0, 200.0])
#     @test all(cache_c .≈ [192.0, 194.0, 196.0, 198.0, 200.0])
# end
#
# @testset "extend_γ" begin
#     #data for beta=12,u=1 => h = 2π/β=0.5235987755982988 
#     test_γ = [875.0225752325874 - 0.047426272065926985im, -1725.9367975032499 + 0.13273183259918772im, 1106.7342694260153 - 0.09340186405920287im, -226.68570367819405 + 0.014352094227780764im, 0.9896265166414565 + 0.00043227905695820295im, 0.9978757876736383 - 0.0006585999770301414im, 0.9986930184922074 - 0.0010106467457844685im, 0.9986261046489859 - 0.0012958932989215043im, 0.9983993208183312 - 0.0016338629583344207im, 0.9980949142745263 - 0.002078738753328137im, 0.9976988866855971 - 0.002685458038367005im, 0.9972007583275463 - 0.0035383199841694316im, 0.9965368770069696 - 0.004768152090402092im, 0.995645502135919 - 0.006591805844899053im, 0.9943660393968459 - 0.00937175752297385im, 0.9923873308819153 - 0.013731243746690323im, 0.9889066965298301 - 0.020694180905500778im, 0.9816721358991439 - 0.03167341201329606im, 0.9640108858763057 - 0.04655766186896375im, 0.9244117285078746 - 0.04832854114431357im, 0.9244117318979092 + 0.04832854220003501im, 0.9640108843948348 + 0.046557660890996555im, 0.9816721358830809 + 0.03167341149714696im, 0.98890669634837 + 0.020694180217158756im, 0.992387331274425 + 0.013731243737292193im, 0.9943660396961701 + 0.009371757480616041im, 0.9956455024259752 + 0.006591805870092188im, 0.9965368769515343 + 0.004768151457081569im, 0.9972007584368632 + 0.0035383197349404417im, 0.9976988864192321 + 0.0026854567958323953im, 0.998094914275765 + 0.002078738237549894im, 0.9983993208192198 + 0.0016338624421033839im, 0.9986261046497896 + 0.0012958927826615702im, 0.9986930184916067 + 0.0010106462296497864im, 0.9978757876733757 + 0.0006585994612941748im, 0.989626516641224 - 0.00043227956742801im, -226.68570367820325 - 0.014351947062101895im, 1106.734269426076 + 0.09340114633426014im, -1725.9367975033356 - 0.13273071298071334im, 875.022575232618 + 0.04742570463505424im]
#     test_γ_2 = copy(test_γ)
#     test_γ_3 = copy(test_γ)
#     ref = ones(size(test_γ)) .+ 0im
#
#     @test all(LadderDGA.find_usable_γ(test_γ; threshold=50, prct_red=0.05) .== (7,34))
#     LadderDGA.extend_γ!(test_γ, 0.5235987755982988)
#     LadderDGA.extend_γ!(test_γ_3, ref)
#     @test all(imag.(test_γ[[1 2 3 4 36 37 38 39]]) .< 0.01)
#     @test all(abs.(real.(test_γ[[1 2 3 4 36 37 38 39]]) .- 1.00) .< 0.01)
#     @test all(test_γ[7:34] .== test_γ_2[7:34])
#     @test all(abs.(test_γ_3[[1 2 3 4 36 37 38 39]]) .== 1)
#     @test all(test_γ_3[7:34] .== test_γ_2[7:34])
# end

@testset "find_usable_χ_interval" begin
    t0 = [ 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6]
    t1 = [ 0.8, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.8]
    t2 = [-0.8, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, -0.8]
    t3 = [-0.8, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, -0.8, -0.9]
    t4 = [-0.8, 0.3, -0.2]
    @test find_usable_χ_interval(t1) == 2:8
    @test find_usable_χ_interval(t1, reduce_range_prct=0.0) == 2:8
    @test find_usable_χ_interval(t1, reduce_range_prct=0.50) == 3:7
    @test find_usable_χ_interval(t2, reduce_range_prct=0.0) == 2:8
    @test_throws ArgumentError find_usable_χ_interval(t3)
    @test find_usable_χ_interval(t0, reduce_range_prct=0.0) == 1:9
    @test find_usable_χ_interval(t2, sum_type=:full, reduce_range_prct=0.5) == 1:9
    @test find_usable_χ_interval(t2, sum_type=(2,8), reduce_range_prct=0.5) == 2:8
    @test all(find_usable_χ_interval([-1.0, -2.0, -1.0]) .== [2])
    @test find_usable_χ_interval(t4, reduce_range_prct=0.9) == [2]
end

@testset "usable indices" begin
    println(usable_ωindices(sP_1, χ_1))
    println(usable_ωindices(sP_1, χ_1, χ_2))
    @test usable_ωindices(sP_1, χ_1) == [1,2,3]
    @test usable_ωindices(sP_1, χ_1, χ_2) == [2]
end
