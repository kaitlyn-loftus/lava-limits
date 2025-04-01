
using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel

"""
this script compares 1e5 simulation results for default numerical conditions 
and 10% of default tolerance to construct Table S2 
"""


outdir = "out/SI/"
sweepname1 = "tabs2_zoomsweep_radbal_logbeta_default"
sweepname2 = "tabs2_zoomsweep_radbal_logbeta_10pcttol"
compare_param_sweep(sweepname1,outdir,sweepname2,outdir)