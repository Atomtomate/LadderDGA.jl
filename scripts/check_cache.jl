using JLD2

Nk = parse(Int, ARGS[1])
cache_name = ARGS[2]


function run_check(Nk, cache_name)
    fail_list = []

    for (root, dirs, files) in walkdir(".")
        if endswith(root, "lDGA_julia")
            fi = findfirst(x -> x == cache_name, files)
            found = false
            print("\r checking $root")
            if !isnothing(fi)
                jldopen(joinpath(root, files[fi])) do f
                    for k in keys(f)
                        if Nk == f["$k/Nk"] 
                            found = true
                            break
                       end
                    end
                end
            end
            !found && push!(fail_list, root)
        end
    end
    print("\r")

    println("Did not find cache for Nk = $Nk in dirs:")
    for f in fail_list
        println(f)
    end
end

run_check(Nk, cache_name)
