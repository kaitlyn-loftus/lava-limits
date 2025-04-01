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
using OrdinaryDiffEq
"""
this script runs a parallelized parameter sweep of the ODE radiative balance cloud temperature model 
for the full parameter space for SI text "The role of lag d in variability"
"""

# set number of parameter combinations 
nsamp = Int(1e5)

# set parameter ranges for sweep 
# exact values randomly selected via Latin hypercube 
# (rng seeds set for reproducibility) 
# here use full parameter bounds 
log_min_Π1_search = log_min_Π1_prime
log_max_Π1_search = log_max_Π1_prime

log_min_Π2_search = log_min_Π2_prime
log_max_Π2_search = log_max_Π2_prime

log_min_Π3_search = log_min_Π3_prime
log_max_Π3_search = log_max_Π3_prime

log_min_β_search = -3.
log_max_β_search = log10(4)
min_ΔTcloud_search = 0.
max_ΔTcloud_search = 350.

# directory to save outputs in 
outdir="out/SI/"
# name of parameter sweep 
# sweepname = "nod_fullsweep_radbal_logbeta_1e5_Tsit5_fullbeta" # original name 
sweepname = "nod_fullsweep_radbal" # rerun as this 
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
t̂end = 5e8

# adjust time checking periods 
t̂check1=500.
Δt̂=250.
alg = Tsit5()

# write notes for netcdf 
notes4nc = "reltol = $(reltol), abstol = $(abstol), tend = $(t̂end) delay times"

# number of cpus to parallelize over 
# if running on a personal computer you may need to decrease ncpus
ncpus = 15

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
        using OrdinaryDiffEq
    end

    # set up parameters 
    ps = setupparams4sweep_βlog(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
        log_min_Π3_search,log_max_Π3_search,log_min_β_search,log_max_β_search,min_ΔTcloud_search,max_ΔTcloud_search)

    # share pmaped function across workers 
    @everywhere begin 
        reltol=$reltol
        abstol=$abstol
        t̂end=$t̂end
        t̂check1=$t̂check1
        Δt̂=$Δt̂
        alg=$alg
        calcsolprop_pmap(i) = calcsolprop_ode($ps[:,i];reltol=reltol,abstol=abstol,t̂end=t̂end,t̂check1=t̂check1,Δt̂=Δt̂)
    end

    # run model over all parameter combinations with progress bar 
    solprop_pmap = @showprogress dt=5.0 desc="simulating" showspeed=true pmap(calcsolprop_pmap,1:nsamp)

    #save results 
    writenc_sweep(solprop_pmap,ps,outdir,sweepname,:rad_bal,notes=notes4nc)

    # end processes 
    println("removing all worker processes!")
    rmprocs(worker_procs;waitfor=30)

    # perform checks
    check_param_sweep_ode(sweepname,outdir;figdirbase=figdirbase,reltol=reltol,abstol=abstol,t̂end=t̂end,isplot=false)
catch e 
    # shut down processes before throwing error 
    println("removing all worker processes!")
    rmprocs(worker_procs;waitfor=30)
    rethrow(e)
end