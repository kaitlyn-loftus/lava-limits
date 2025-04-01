using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using Distributed
using ProgressMeter
using Random
using LHS 
using JLD2
"""
this script runs a parallelized parameter sweep of the radiative balance cloud temperature model 
for the full parameter space for supporting figure 4
"""

# set number of parameter combinations 
nsamp = Int(1e5)

# set parameter ranges for sweep 
# exact values randomly selected via Latin hypercube 
# (rng seeds set for reproducibility) 
# here use full parameter bounds 
log_min_Π1_search = log_min_Π1
log_max_Π1_search = log_max_Π1

log_min_Π2_search = log_min_Π2
log_max_Π2_search = log_max_Π2

log_min_Π3_search = log_min_Π3
log_max_Π3_search = log_max_Π3

log_min_β_search = -3
log_max_β_search = 0.
min_ΔTcloud_search = 0.
max_ΔTcloud_search = 350.
T_ref_min = 50e3 # [K]
T_ref_max = 80e3 # [K] 
log_p_ref_min = 10. # [log(Pa)]
log_p_ref_max = 14. # [log(Pa)]
α_min = 0.35
α_max = 0.65

# directory to save outputs in 
outdir="out/SI/"
# name of parameter sweep 
sweepname = "sfig2_fullsweep_radbal_logbeta"
# note output file saved as outdir*sweepname*".nc"
# crash program if output file already exists (otherwise netcdf writing will crash at end)
fname = outdir*sweepname*".nc"
if isfile(fname)
    error("file $(fname) already exists! "*
    "change outdir (currently $(outdir)) or sweepname (currently $(sweepname)).")
end

# set figdirbase 
figdirbase = "sfigs/"

# set numerical tolerances for integration 
reltol = 1e-8 
abstol = 1e-10 

# set how long to integrate  
t̂end = 1e7

# write notes for netcdf 
notes4nc = "reltol = $(reltol), abstol = $(abstol), tend = $(t̂end) delay times"

# number of cpus to parallelize over 
# if running on a personal computer you may need to decrease ncpus
ncpus = 50

# set up workers for distributed sweep 

# check don't exceed number of cpus available  
# DO NOT CHANGE ANYTHING IN THIS BLOCK #######################################################
# these lines make sure you do not try to parallelize over more cpus than you have available 
# adjust ncpus lower if you are seeing the below error thrown 
# note: you may need to set ncpus below maxcpus for optimal performance 
# depending on how your cpus are threaded 
maxcpus = Sys.CPU_THREADS - 2
if ncpus>maxcpus
    error("number of worker processes requested $(ncpus) exceeds number of cpus available $(maxcpus)!\nset ncpus ≤ $(maxcpus) or do not run this script!")
end
# DO NOT CHANGE ANYTHING IN THIS BLOCK #######################################################

println("adding $ncpus processes")

# add workers for distributed search 
worker_procs = addprocs(ncpus)


# wrap everything in try-catch block so crash will kill all workers 
try
    # load packages on workers 
    @everywhere begin
        if VERSION < v"1.9.0"
            using Pkg
            Pkg.activate(".")
        end
        using Revise
        push!(LOAD_PATH,"./src/")
        using DDEModel
        using ProgressMeter
    end

    # set up parameters 
    ps = setupparams4sweep_βlog_refα(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
        log_min_Π3_search,log_max_Π3_search,log_min_β_search,log_max_β_search,min_ΔTcloud_search,max_ΔTcloud_search,
        T_ref_min,T_ref_max,log_p_ref_min,log_p_ref_max,α_min,α_max)

    # share pmaped function across workers 
    @everywhere begin 
        reltol=$reltol
        abstol=$abstol
        t̂end=$t̂end
        calcsolprop_pmap(i) = calcsolprop_radbal($ps[:,i];reltol=reltol,abstol=abstol,t̂end=t̂end)
    end

    # run model over all parameter combinations with progress bar 
    solprop_pmap = @showprogress dt=5.0 desc="simulating" showspeed=true pmap(calcsolprop_pmap,1:nsamp)

    #save results 
    writenc_sweep(solprop_pmap,ps,outdir,sweepname,:rad_bal,notes=notes4nc)

    # end processes 
    println("removing all worker processes!")
    rmprocs(worker_procs;waitfor=30)

    # perform checks
    check_param_sweep(sweepname,outdir;figdirbase=figdirbase,reltol=reltol,abstol=abstol,t̂end=t̂end)
catch e 
    # shut down processes before throwing error 
    println("removing all worker processes!")
    rmprocs(worker_procs;waitfor=30)
    rethrow(e)
end

# figdir = "sfigs/checkcrash/"
# mkpath(figdir)
# p_flag = [0.49968783142640166, 53.5541727381798, 7.549126852122597, 74874.75770374775, 5.241830546541136e13, 0.16154652412436424, 3.4e6, 1.0, 0.8704936781485819, 309.1329223710423, 1400.0]
# p_flag = [3.6555265702386555, 55.49513801989231, 56.787771524565905, 72386.3757185813, 1.3300796120975305e13, 0.11773651133948568, 3.4e6, 1.0, 0.4862676751987222, 218.41632880697233, 1400.0]
# p_flag = [0.021517409300516363, 40.156957324228905, 2.864244059548288, 54938.606532794154, 2.515026302583467e12, 0.14177050268309502, 3.4e6, 1.0, 0.7596112338614838, 253.3426473604065, 1400.0]
# p_flag = [0.08260150009772235, 7.09521925645882, 32.24289985350491, 71731.65396812378, 6.50842813001369e13, 0.32556940675010315, 3.4e6, 1.0, 0.8709651713002388, 227.14658036421514, 1400.0]
# calcsolprop_radbal(p_flag;isplot=true,figdir=figdir,runname="test4",reltol=reltol,abstol=abstol,t̂end=1e4)
# checksol_radbal(p_flag,figdir,"test4";reltol=reltol,abstol=abstol,t̂end=1e4)
