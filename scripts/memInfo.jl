using Printf: @printf

function meminfo_julia()
  # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
  # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
  @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
  @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
  @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
end

function meminfo_procfs(pid=getpid())
  smaps = "/proc/$pid/smaps_rollup"
  if !isfile(smaps)
    error("`$smaps` not found. Maybe you are using an OS without procfs support or with an old kernel.")
  end

  rss = pss = shared = private = 0
  for line in eachline(smaps)
    s = split(line)
    if s[1] == "Rss:"
      rss += parse(Int64, s[2])
    elseif s[1] == "Pss:"
      pss += parse(Int64, s[2])
    elseif s[1] == "Shared_Clean:" || s[1] == "Shared_Dirty:"
      shared += parse(Int64, s[2])
    elseif s[1] == "Private_Clean:" || s[1] == "Private_Dirty:"
      private += parse(Int64, s[2])
    end
  end

  @printf "RSS:       %9.3f MiB\n" rss/2^10
  @printf "┝ shared:  %9.3f MiB\n" shared/2^10
  @printf "┕ private: %9.3f MiB\n" private/2^10
  @printf "PSS:       %9.3f MiB\n" pss/2^10
end
