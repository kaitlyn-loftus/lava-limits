module DDEModel

using OrdinaryDiffEq
using JLD2
using Roots
using Statistics 
using CairoMakie
import DelayDiffEq:MethodOfSteps
using NetCDF
using Interpolations
using Distributed
using ProgressMeter
using LHS

"""
main model functions 
"""

# export statements 
export calcsolprop_radbal,checksol_radbal,outputrun_radbal
export calcsolprop_const,checksol_const,outputrun_const
export writenc_sweep,setupparams4sweep,setupparams4sweep_Πonly,setupparams4sweep_βlog
export p_ref_SiO_0vap,T_ref_SiO_0vap
export log_min_Π1,log_max_Π1,log_min_Π2,log_max_Π2,log_min_Π3,log_max_Π3
export check_param_sweep,compare_param_sweep
export make_Π_Tcloudβ_3σ_figs_scatter,make_Π_Tcloudβ_3σ_figs_2model_poly
export make_Π_Tcloudβ_3σ_figs_radbal_nolw_poly


# CONSTANTS  ######################################################

# physical constants 
const σ = 5.670374419e-8 # [W m⁻² K⁻⁴]
const h = 6.62607015e-34 # [J Hz⁻¹]
const c = 299792458. # [m s⁻¹]
const kB = 1.380649e-23 # [J K⁻¹]

# 55 Cnc e (planet) constants 
const a55cnce=2.3397e9 # [m] semi-major orbital axis 

# 55 Cnc A (star) constants 
const T55cncA=5172. # [K] stellar emission temperature 
const R55cncA=6.56e8 # [m] stellar radius 

# 0% vap BSE Mg and SiO pref and Tref 
const p_ref_Mg_0vap = 2.91870298980527e12 # [Pa]
const T_ref_Mg_0vap = 65853.14435237370 # [K]
const p_ref_SiO_0vap = 8.07946873717266e13 # [Pa]
const T_ref_SiO_0vap = 71024.42973307280 # [K]
# 20% vap BSE Mg and SiO pref and Tref 
const p_ref_Mg_20vap = 1.92860843495236e11 # [Pa]
const T_ref_Mg_20vap = 56492.709933784800 # [K]
const p_ref_SiO_20vap = 8.16648887123696e11 # [Pa]
const T_ref_SiO_20vap = 58759.07408917050 # [K]

# bulk parameter bounds 
const log_min_Π1 = -3.19975517725347 # log([Pa⁻¹])
const log_max_Π1 = 2.40671393297954 # log([Pa⁻¹])
const log_min_Π2 = -7.3406727607345 # [ ]
const log_max_Π2 = 3.71024848746481 # [ ]
const log_min_Π3 = 0.439369249066335 # log([kg s⁻³ K⁻¹])
const log_max_Π3 = 6.26376753354504 # log([kg s⁻³ K⁻¹])

# CONSTANTS  ######################################################

####################################################################

# FUNCTIONS TO INTEGRATE MODEL  ####################################
function solvedTτLdt̂_radbal(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,t̂end,T₀,τ₀,Tsurfeq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)
    """
    integrate coupled equations for dT/dt̂, dτ/dt̂, & dL/dt̂ following the radiative balance model 
    inputs:
        * Π₁ [Pa⁻¹] - parameter 1
        * Π₂ [ ] - parameter 2
        * Π₃ [kg s⁻³ K⁻¹] - parameter 3
        * p_ref [Pa] - reference pressure for Clausius-Clapeyron
        * T_ref [K] - reference temperature for Clausius-Clapeyron
        * α [ ] - albedo parameter 
        * S₀ [W m⁻²] - incident stellar insolation 
        * f [ ] - heat redistribution factor 
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
        * Tmagma_solidus [K] - solidus temperature of magma 
        * ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
        * t̂end [ ] - maximum (model) time to integrate for 
        * T₀ [K] - initial surface temperature 
        * τ₀ [ ] - initial shortwave optical depth 
        * Tsurfeq [K] - surface temperature fixed point 
        * maxiters [int] - maximum iterations of the DDE solver 
            + optional, default value: 1e7 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
    output:
        * DDE solution [SciMLBase.ODESolution] - solution object 
    """ 
    # set up times to check for early stopping 
    t̂checks = t̂check1:Δt̂:t̂end

    # don't allow initial Tsurf condition below or at solidus
    T₀ = max(T₀,Tmagma_solidus+1.)
    # therefore, start with L integration off 
    doLint = 0

    # set initial condition 
    TτL₀ = [T₀,τ₀,0.]

    # non-allocating history function 
    # see discussion of history functions in DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/types/dde_types/#dde_prob
    # https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/
    calch₀(p,t;idxs=nothing) = typeof(idxs) <: Number ? TτL₀[idxs] : TτL₀

    # set up DDE problem 
    # see DifferentialEquations.jl documentation: 
    # https://docs.sciml.ai/DiffEqDocs/stable/types/dde_types/
    prob = DDEProblem(calcdTτLdt̂_radbal!, TτL₀, calch₀, (0.,t̂end), [Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,doLint];
    constant_lags=[1.],isoutofdomain=isoutofdomain)

    # convert ODE integration algorithm to DDE equivalent 
    # see Widmann & Rackauckas (2022) [doi:10.48550/arXiv.2208.12879]
    # and DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/solvers/dde_solve/
    alg = MethodOfSteps(alg)

    # set up callbacks 
    # see DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/

    # check whether solution has stabilized to fix point or limit cycle 
    checkisstop = (u,t,integrator) -> checkisstoplong(u,t,integrator,t̂checks,Tsurfeq,Δt̂,ΔTthres)
    cb_stop = DiscreteCallback(checkisstop,terminate!) #save_positions=(true, false))
    # check whether surface temperature hits magma ocean solidus temperature 
    cb_magma_solidus = ContinuousCallback(checkTmagma_solidus,nothing,affectTmagma_solidus_radbal!)
    # check whether enough latent heat of melting has been supplied to resume surface temperature increasing 
    # (when surface temperature at magma ocean solidus temperature)
    cb_increaseTsurf = ContinuousCallback(checkL_increaseTsurf,nothing,affectL_increaseTsurf_radbal!)
    # group callbacks together 
    cbset = CallbackSet(cb_stop,cb_increaseTsurf,cb_magma_solidus)

    # integrate! 
    # return solution 
    # see DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/
    solve(prob,alg;maxiters=maxiters,callback=cbset,abstol=abstol,reltol=reltol,tstops=t̂checks,dtmax=Δt̂/4) 
end

function solvedTτLdt̂_const(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,t̂end,T₀,τ₀,Tsurfeq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)
    """
    integrate coupled equations for dT/dt̂, dτ/dt̂, & dL/dt̂ following the constant model 
    inputs:
        * Π₁ [Pa⁻¹] - parameter 1
        * Π₂ [ ] - parameter 2
        * Π₃ [kg s⁻³ K⁻¹] - parameter 3
        * p_ref [Pa] - reference pressure for Clausius-Clapeyron
        * T_ref [K] - reference temperature for Clausius-Clapeyron
        * α [ ] - albedo parameter 
        * S₀ [W m⁻²] - incident stellar insolation 
        * f [ ] - heat redistribution factor 
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
        * Tmagma_solidus [K] - solidus temperature of magma 
        * Tcloud [K] - constant cloud downward and upward emission temperature (Tcloud = Tcloud↓ = Tcloud↑)
        * t̂end [ ] - maximum (model) time to integrate for 
        * T₀ [K] - initial surface temperature 
        * τ₀ [ ] - initial shortwave optical depth 
        * Tsurfeq [K] - surface temperature fixed point 
        * maxiters [int] - maximum iterations of the DDE solver 
            + optional, default value: 1e8 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
        * ΔTthres [K] - surface temperature buffer around fixed point for determining end state 
            note, lower values will cause longer integration
            do not set below about 10*reltol*Teq 
            + optional, default value: 1 K 
        * Δt̂ [[delay time] - time duration to use to check ending conditions, note Δt̂ ≤ t̂check1
            + optional, default value: 100 delay times
        * t̂check1 [delay time] - first time to check DDE ending conditions 
            + optional, default value: 300 delay times
    output:
        * DDE solution [SciMLBase.ODESolution] - solution object 
    """ 
    # set up times to check for early stopping 
    t̂checks = t̂check1:Δt̂:t̂end

    # don't allow initial Tsurf condition below or at solidus
    T₀ = max(T₀,Tmagma_solidus+1.)
    # therefore, start with L integration off 
    doLint = 0

    # set initial condition 
    TτL₀ = [T₀,τ₀,0.]

    # non-allocating history function 
    # see discussion of history functions in DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/types/dde_types/#dde_prob
    # https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/
    calch₀(p,t;idxs=nothing) = typeof(idxs) <: Number ? TτL₀[idxs] : TτL₀

    # set up DDE problem 
    # see DifferentialEquations.jl documentation: 
    # https://docs.sciml.ai/DiffEqDocs/stable/types/dde_types/
    prob = DDEProblem(calcdTτLdt̂_const!, TτL₀, calch₀, (0.,t̂end), [Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,doLint];
    constant_lags=[1.])

    # convert ODE integration algorithm to DDE equivalent 
    # see Widmann & Rackauckas (2022) [doi:10.48550/arXiv.2208.12879]
    # and DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/solvers/dde_solve/
    alg = MethodOfSteps(alg)

    # set up callbacks 
    # see DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/

    # check whether solution has stabilized to fix point or limit cycle 
    checkisstop = (u,t,integrator) -> checkisstoplong(u,t,integrator,t̂checks,Tsurfeq,Δt̂,ΔTthres)
    cb_stop = DiscreteCallback(checkisstop,terminate!) #save_positions=(true, false))
    # check whether surface temperature hits magma ocean solidus temperature 
    cb_magma_solidus = ContinuousCallback(checkTmagma_solidus,nothing,affectTmagma_solidus_const!)
    # check whether enough latent heat of melting has been supplied to resume surface temperature increasing 
    # (when surface temperature at magma ocean solidus temperature)
    cb_increaseTsurf = ContinuousCallback(checkL_increaseTsurf,nothing,affectL_increaseTsurf_const!)
    # group callbacks together 
    cbset = CallbackSet(cb_stop,cb_increaseTsurf,cb_magma_solidus)

    # integrate! 
    # return solution 
    # see DifferentialEquations.jl documentation:
    # https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/
    solve(prob,alg;maxiters=maxiters,callback=cbset,abstol=abstol,reltol=reltol,tstops=t̂checks,dtmax=Δt̂/4) 
end

# FUNCTIONS TO INTEGRATE MODEL  ####################################

####################################################################

# FUNCTION FOR DERIVATIVES  ########################################

function calcdTτLdt̂_radbal!(dTτLdt̂,TτL,h,p,t̂)
    """
    calculate dTτLdt̂ in place for radiative balance model  
    inputs:
        * dTτLdt̂ [Array] - array with dTsurfdt̂, dτSWdt̂, dLdt̂ to be modified in place
            + dTτLdt̂[1] = dTsurf/dt̂ [K / delay time]
            + dTτLdt̂[2] = dτSW/dt̂ [1 / delay time]
            + dTτLdt̂[3] = dL/dt̂ [J m⁻² / delay time]
        * TτL [Array] - array with Tsurf(t̂), τSW(t̂), L(t̂)
            + TτL[1] - Tsurf [K]
            + TτL[2] - τSW [ ]
            + TτL[3] - L [J m⁻²]
        * h [function] - history function, see DifferentialEquations documentation
        * t̂ [delay time] - time 
        * p [Array] - array of parameters of DDE 
            + Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,doLint
    output:
        nothing 
    """
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,doLint = view(p,:) # unpack parameters
    Tsurf, τSW, L = TτL # unpack T(t̂), τ(t̂), L(t̂)

    # reset when integrator tries crazy test values for Tsurf and τSW 
    Tsurf = max(Tsurf,ΔTcloud+1.)
    # τSW = max(τSW,0.)
 
    εcloud = calcεcloud(τSW,β) # [ ] LW cloud emissivity 
    T_surf_delay = h(p,t̂-1;idxs=1) # [K] delay Tsurf

    dTτLdt̂[2] = Π₁*p_ref*exp(-T_ref/T_surf_delay) - Π₂*τSW # dτSW/dt̂ [ ]

    # solve for Tcloud_down 
    Tcloud_down = calcTclouddown(ΔTcloud,Tsurf) # [K]

    # note doLint set externally via callback 
    # determines whether to integrate L or Tsurf 
    # not a Boolean for performance purposes 
    if doLint==1.
        # when Tsurf==Tsolidus 
        # if cooling, put cooling toward latent heat 
        # if heating, put toward latent heat until L = 0
        dTτLdt̂[1] = 0. # dTsurf/dt̂ [K s⁻¹]
        dTτLdt̂[3] = -(f*S₀/(1+α*τSW) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4) # dL/dt̂ [J m⁻² s⁻¹]
    else
        dTτLdt̂[1] = (f*S₀/(1+α*τSW) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4)/Π₃ # dTsurf/dt̂ [K s⁻¹]
        dTτLdt̂[3] = 0. # dL/dt̂ [J m⁻² s⁻¹]
    end
  
    nothing
end

function calcdTτLdt̂_const!(dTτLdt̂,TτL,h,p,t̂)
    """
    calculate dTτLdt̂ in place for constant model  
    inputs:
        * dTτLdt̂ [Array] - array with dTsurfdt̂, dτSWdt̂, dLdt̂ to be modified in place
            + dTτLdt̂[1] = dTsurf/dt̂ [K / delay time]
            + dTτLdt̂[2] = dτSW/dt̂ [1 / delay time]
            + dTτLdt̂[3] = dL/dt̂ [J m⁻² / delay time]
        * TτL [Array] - array with Tsurf(t̂), τSW(t̂), L(t̂)
            + TτL[1] - Tsurf [K]
            + TτL[2] - τSW [ ]
            + TτL[3] - L [J m⁻²]
        * h [function] - history function, see DifferentialEquations documentation
        * t̂ [delay time] - time 
        * p [Array] - array of parameters of DDE 
            + Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,doLint
    output:
        nothing 
    """
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,doLint = view(p,:) # unpack parameters
    Tsurf, τSW, L = TτL # unpack T(t̂), τ(t̂), L(t̂)

    # reset when integrator tries crazy test values for Tsurf and τSW 
    Tsurf = max(Tsurf,Tmagma_solidus - 10.)
    τSW = max(τSW,0.)
 
    εcloud = calcεcloud(τSW,β) # [ ] LW cloud emissivity 
    T_surf_delay = h(p,t̂-1;idxs=1) # [K] delay Tsurf

    dTτLdt̂[2] = Π₁*p_ref*exp(-T_ref/T_surf_delay) - Π₂*τSW # dτSW/dt̂ [ ]

    # set Tcloud_down 
    Tcloud_down = Tcloud # [K]

    # note doLint set externally via callback 
    # determines whether to integrate L or Tsurf 
    # not a Boolean for performance purposes 
    if doLint==1.
        # when Tsurf==Tsolidus 
        # if cooling, put cooling toward latent heat 
        # if heating, put toward latent heat until L = 0
        dTτLdt̂[1] = 0. # dTsurf/dt̂ [K s⁻¹]
        dTτLdt̂[3] = -(f*S₀/(1+α*τSW) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4) # dL/dt̂ [J m⁻² s⁻¹]
    else
        dTτLdt̂[1] = (f*S₀/(1+α*τSW) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4)/Π₃ # dTsurf/dt̂ [K s⁻¹]
        dTτLdt̂[3] = 0. # dL/dt̂ [J m⁻² s⁻¹]
    end
  
    nothing
end

function calcdTτLdt̂_radbal_sourcesink(Tsurf,τSW,T_surf_delay,Π₁,Π₂,p_ref,T_ref,α,S₀,f,β,ΔTcloud)
    """
    calculate key components of radiative balance model for output
    """
    τLW = τSW*β
    εcloud = calcεcloud(τSW,β)
    Tcloud_down = calcTclouddown(ΔTcloud,Tsurf)
    Tcloud_up = calcTcloudup(ΔTcloud,Tsurf)
    FcloudLW_down = εcloud*σ*Tcloud_down^4
    FcloudLW_up = εcloud*σ*Tcloud_up^4
    FsurfLWemit = σ*Tsurf^4
    FsurfSW = f*S₀/(1+α*τSW)
    FsurfLWTOA = σ*(1 .- εcloud).*Tsurf.^4
    FLWTOA = FsurfLWTOA + FcloudLW_up

    dτdt̂_source = Π₁*p_ref*exp(-T_ref/T_surf_delay) 
    dτdt̂_sink = Π₂*τSW

    τLW,FcloudLW_down,FcloudLW_up,FsurfLWemit,FsurfSW,FsurfLWTOA,FLWTOA,dτdt̂_source,dτdt̂_sink
end

function calcdTτLdt̂_const_sourcesink(Tsurf,τSW,T_surf_delay,Π₁,Π₂,p_ref,T_ref,α,S₀,f,β,Tcloud)
    """
    calculate key components of constant model for output
    """
    τLW = τSW*β
    εcloud = calcεcloud(τSW,β)
    Tcloud_down = Tcloud
    Tcloud_up = Tcloud
    FcloudLW_down = εcloud*σ*Tcloud_down^4
    FcloudLW_up = εcloud*σ*Tcloud_up^4
    FsurfLWemit = σ*Tsurf^4
    FsurfSW = f*S₀/(1+α*τSW)
    FsurfLWTOA = σ*(1 .- εcloud).*Tsurf.^4
    FLWTOA = FsurfLWTOA + FcloudLW_up

    dτdt̂_source = Π₁*p_ref*exp(-T_ref/T_surf_delay) 
    dτdt̂_sink = Π₂*τSW

    τLW,FcloudLW_down,FcloudLW_up,FsurfLWemit,FsurfSW,FsurfLWTOA,FLWTOA,dτdt̂_source,dτdt̂_sink
end

# FUNCTION FOR DERIVATIVES  ########################################

####################################################################

# FUNCTIONS ABOUT FIXED POINTS  ####################################

function findTeqLWradbal0long(T,Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)
    """
    function that returns 0 at Tsurf fixed point for radiative balance model 
    (i.e., when dTdt̂ = 0 & dτdt̂ = 0)
    """
    # set dTdt̂ = 0, dτdt̂ = 0 and solve for associated T
    τ = Π₁*p_ref*exp(-T_ref/T)/Π₂
    εcloud = calcεcloud(τ,β)
    Tcloud_down = calcTclouddown(ΔTcloud,T)
    f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*T^4
end

function findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud;Teq_min=1000.,Teq_max=3500.)
    """
    find surface temperature and shortwave cloud optical depth for radiative balance model
    set dTdt̂ = 0, dτdt̂ = 0 and solve for associated Tsurf,τSW numerically

    inputs:
        * Π₁ [Pa⁻¹] - bulk parameter 1
        * Π₂ [ ] - bulk parameter 2
        * T_ref [K] - reference temperature for Clausius-Clapeyron
        * p_ref [Pa] - reference pressure for Clausius-Clapeyron
        * α [ ] - albedo parameter 
        * S₀ [W m⁻²] - incident stellar insolation 
        * f [ ] - heat redistribution factor 
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
        * ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
        * Teq_min [K] - minimum guess for Teq 
        * Teq_max [K] - maximum guess for Teq, this should be set from S₀
    outputs:
        * Teq [K] - fixed point surface T
        * τeq [ ] - fixed point shortwave τ
    """
    # make version of zero function which only requires T
    # as an input to put into root finder
    findTeq0 = (T) -> findTeqLWradbal0long(T,Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud) 
    # root find fixed point T
    Teq = try 
        find_zero(findTeq0,(Teq_min,Teq_max))
    catch e
        println("root finding error in findTτeqnum_radbal!")
        @show Π₁ Π₂ T_ref p_ref α S₀ f β ΔTcloud
        rethrow(e)
    end
    # calculate fixed point τ from T
    τeq = Π₁*p_ref*exp(-T_ref/Teq)/Π₂
    # return values
    Teq,τeq
end

function findTeqLWconst0long(T,Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud)
    """
    function that returns 0 at Tsurf fixed point for constant model 
    (i.e., when dTdt̂ = 0 & dτdt̂ = 0)
    """
    # set dTdt̂ = 0, dτdt̂ = 0 and solve for associated T
    τ = Π₁*p_ref*exp(-T_ref/T)/Π₂
    εcloud = calcεcloud(τ,β)
    Tcloud_down = Tcloud
    f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*T^4
end

function findTτeqnum_const(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud;Teq_min=1000.,Teq_max=3500.)
    """
    find surface temperature and shortwave cloud optical depth fixed points for constant model
    set dTdt̂ = 0, dτdt̂ = 0 and solve for associated T,τ numerically

    inputs:
        * Π₁ [Pa⁻¹] - bulk parameter 1
        * Π₂ [ ] - bulk parameter 2
        * T_ref [K] - reference temperature for Clausius-Clapeyron
        * p_ref [Pa] - reference pressure for Clausius-Clapeyron
        * α [ ] - albedo parameter 
        * S₀ [W m⁻²] - incident stellar insolation 
        * f [ ] - heat redistribution factor 
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
        * Tcloud [K] - constant cloud downward and upward emission temperature (Tcloud = Tcloud↓ = Tcloud↑)
        * Teq_min [K] - minimum guess for Teq 
        * Teq_max [K] - maximum guess for Teq, this should be set from S₀
    outputs:
        * Teq [K] - fixed point surface T
        * τeq [ ] - fixed point shortwave τ
    """
    # make version of zero function which only requires T
    # as an input to put into root finder
    findTeq0 = (T) -> findTeqLWconst0long(T,Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud) 
    # root find fixed point T
    Teq = try 
        find_zero(findTeq0,(Teq_min,Teq_max))
    catch e
        println("root finding error in findTτeqnum_const!")
        @show Π₁ Π₂ T_ref p_ref α S₀ f β Tcloud
        rethrow(e)
    end
    # calculate fixed point τ from T
    τeq = Π₁*p_ref*exp(-T_ref/Teq)/Π₂
    # return values
    Teq,τeq
end
# FUNCTIONS ABOUT FIXED POINTS  ####################################

####################################################################

####################################################################

# FUNCTIONS DEALING WITH CALLBACKS  ################################
function findperiod0long(t̂,sol,Teq)
    """
    find the t̂ where the surface temperature equals its fixed point value 
    i.e., t̂ s.t. Tsurf(t̂) = Teq
    (long version)
    used to check whether to stop DDE model integration
    """
    sol(t̂;idxs=1) - Teq
end

function checkisstoplong(u,t,integrator,tchecks,Teq,Δt,ΔTthres;no_pts=20)::Bool
    """
    check whether to stop integration (long version)
    """
    # set default to not stop 
    isstop = false
    if t ∈ tchecks # check if checking time 
        sol = integrator.sol
        filt = sol.t .>= (t-Δt)
        endTs = sol[1,filt]
        Tmin = minimum(endTs)
        Tmax = maximum(endTs)
        if (Tmin <= (Teq + ΔTthres)) && (Tmax>=(Teq - ΔTthres)) # check if T solution bounding Teq 
            if (Tmin >= (Teq - ΔTthres)) && (Tmax <= (Teq + ΔTthres)) # check if within threshold
                isstop = true 
            else # check what oscillations present 
                findperiod0 = (t̂) -> findperiod0long(t̂,sol,Teq)
                endt̂s = sol.t[filt]
                endΔTs = endTs .- Teq
                filt_pos_endΔT = endΔTs .> 0
                filt_neg_endΔT = endΔTs .< 0
                if (sum(filt_pos_endΔT) > 0) && (sum(filt_neg_endΔT) > 0)
                    t̂start = endt̂s[filt_pos_endΔT][1]
                    t̂end = endt̂s[filt_neg_endΔT][end]
                    t̂whereTeq = find_zeros(findperiod0,t̂start,t̂end;no_pts=no_pts)
                    nt̂whereTeq = length(t̂whereTeq)
                    if nt̂whereTeq>5
                        t̂Pstart = t̂whereTeq[3:2:end]
                        t̂Pend = t̂whereTeq[1:2:(end - 2 + nt̂whereTeq%2)]
                        Ps = t̂Pstart .- t̂Pend
                        P = mean(Ps)   
                        if std(Ps)/P > 1e-2 # period erratic 
                            isstop = false
                        else # regular period 
                            isstop = true 
                        end
                    else # very slowly oscilating 
                        isstop = false
                    end
                else # T only bounding Teq given relative error 
                    isstop = false
                end
            end
        else # T solution is not bounding Teq 
            isstop = false
        end
    else # not designated time to evaluate whether to end simulation
        isstop = false
    end
    isstop
end

function checkL_increaseTsurf(TτL,t,integrator)
    """
    check when L = 0 to know when to switch back from integrating L to integrating Tsurf
    """
    TτL[3] 
end

function affectL_increaseTsurf_radbal!(integrator)
    """
    apply this function when L transitions from L>0 to L = 0
    turn off L integration, set L = 0 (exactly)
    version for radiative balance model 
    """
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,dointL = view(integrator.p,:)
    Tsurf,τ,L = integrator.u
    εcloud = calcεcloud(τ,β)
    Tcloud_down = calcTclouddown(ΔTcloud,Tsurf)
    F = f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4
    if F > 0.
        integrator.u[3] = 0.
        integrator.p[end] = 0. # turn off L integration 
        # adjust time step to be small given switch in integration 
        Δt = get_proposed_dt(integrator)
        set_proposed_dt!(integrator,max(Δt*1e-1,integrator.opts.dtmin*1e4))
        # adjust τ in case of interpolator error 
        if τ<0
            integrator.u[2] = 0.
        end
    else
        # ignore if L integration turned off and L smaller than absolute tolerance
        if dointL==0 && L < integrator.opts.abstol 
            integrator.u[3] = 0.
        else # otherwise alert user 
            @info "L crossed to 0 but F <= 0"
            @show Tsurf L F dointL
        end
    end
    nothing 
end

function affectL_increaseTsurf_const!(integrator)
    """
    apply this function when L transitions from L>0 to L = 0
    turn off L integration, set L = 0 (exactly)
    version for constant model
    """
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,dointL = view(integrator.p,:)
    Tsurf,τ,L = integrator.u
    εcloud = calcεcloud(τ,β)
    Tcloud_down = Tcloud
    F = f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4
    if F > 0.
        integrator.u[3] = 0.
        integrator.p[end] = 0. # turn off L integration 
    else
        # ignore if L integration turned off and L smaller than absolute tolerance
        if dointL==0 && L < integrator.opts.abstol 
            integrator.u[3] = 0.
        else # otherwise alert user 
            @info "L crossed to 0 but F <= 0"
            @show Tsurf L F dointL
        end
    end
    nothing 
end

function checkTmagma_solidus(TτL,t,integrator)
    """
    check when Tsurf hits Tsolidus to switch from integrating Tsurf to integrating L 
    """
    Tmagma_solidus = integrator.p[10] # get Tsolidus 
    Tsurf = TτL[1] # get Tsurf
    Tsurf - Tmagma_solidus
end

function affectTmagma_solidus_radbal!(integrator)
    """
    apply this function when Tsurf decreases to hit Tsolidus 
    turn on L integration instead of Tsurf integration,
    set Tsurf = Tsolidus (exactly)
    exclude edge case where Tsurf derivative switches to increasing again when hitting Tsolidus
    version for radiative balance model
    """
    # turn on L integration unless dTsurf/dt switches sign exactly at Tsurf = Tmagma_solidus 
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,dointL = view(integrator.p,:)
    Tsurf,τ,L = integrator.u
    εcloud = calcεcloud(τ,β)
    Tcloud_down = calcTclouddown(ΔTcloud,Tsurf)
    F = f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4
    if F < 0. # exclude case where F ≥ 0 exactly at Tsurf = Tmagma_solidus 
        integrator.p[end] = 1. # turn on L integration 
        # adjust time step to be small to prevent overshooting 
        # rapid magma ocean transition from freezing to melting 
        # and getting L < 0 
        Δt = get_proposed_dt(integrator)
        set_proposed_dt!(integrator,max(Δt*1e-4,integrator.opts.dtmin*1e4))
        integrator.u[1] = integrator.p[10] # set Tsurf = Tmagma_solidus
        if L < 0. 
            integrator.u[3] = 0. 
        end
    end
    nothing 
end

function affectTmagma_solidus_const!(integrator)
    """
    apply this function when Tsurf decreases to hit Tsolidus 
    turn on L integration instead of Tsurf integration,
    set Tsurf = Tsolidus (exactly)
    exclude edge case where Tsurf derivative switches to increasing again when hitting Tsolidus
    version for constant model
    """
    # turn on L integration unless dTsurf/dt switches sign exactly at Tsurf = Tmagma_solidus 
    Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,dointL = view(integrator.p,:)
    Tsurf,τ,L = integrator.u
    εcloud = calcεcloud(τ,β)
    Tcloud_down = Tcloud
    F = f*S₀/(1+α*τ) + εcloud*σ*Tcloud_down^4 - σ*Tsurf^4
    if F < 0. # exclude case where F ≥ 0 exactly at Tsurf = Tmagma_solidus 
        integrator.p[end] = 1. # turn on L integration 
        # adjust time step to be small to prevent overshooting 
        # rapid magma ocean transition from freezing to melting 
        # and getting L < 0 
        Δt = get_proposed_dt(integrator)
        set_proposed_dt!(integrator,max(Δt*1e-5,integrator.opts.dtmin*10.))
        integrator.u[1] = integrator.p[10] # set Tsurf = Tmagma_solidus
    end
    nothing 
end

function isoutofdomain(u,p,t)
    """
    return true when solution out of domain 
    (slighly relaxed due to callbacks)
    """
    Tsurf,τ,L = u 
    Tmagma_solidus = p[10]
    (Tsurf < (Tmagma_solidus - 1)) || (τ<0.) #|| (L < -1e-2)
end

# FUNCTIONS DEALING WITH CALLBACKS  ################################

####################################################################


# FUNCTIONS TO DIAGNOSE CLOUD PROPERTIES  ##########################
function calcA(τSW,α)
    """
    calculate shortwave cloud albedo as a function of optical depth 
    input:
        * τSW [ ] - shortwave optical depth
    output:
        * A [ ] - cloud albedo
    """
    1. - 1/(1. + α*τSW)  # [ ]
end

function calcεcloud(τSW,β)
    """
    calculate cloud longwave emissivity 
    inputs:
        * τSW [ ] - shortwave optical depth
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
    output:
        * εcloud [ ] - cloud longwave emissivity [ ]
    """
    1 - exp(-β*τSW)
end

function calcTcloudup(ΔTcloud,Tsurf)
    """
    calculate cloud upward emission temperature
    analytical solution to radiative energy balance in cloud layer: 
    σ*εcloud*Tsurf^4 = σ*εcloud*Tcloud↑^4 + σ*εcloud*Tcloud↓^4
    assuming ΔTcloud ≥ 0, Tsurf > 0, Tcloud↑ > 0, and 
    ΔTcloud = Tcloud↓ - Tcloud↑ 
    inputs:
        * ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
        * Tsurf [K] - surface temperature 
    outputs:
        * Tcloudup [K] - upward cloud emission temperature 
    """
    0.5*(-ΔTcloud + sqrt(-3*ΔTcloud^2 + 2*sqrt(2)*sqrt(ΔTcloud^4 + Tsurf^4)))
end

function calcTclouddown(ΔTcloud,Tsurf)
    """
    calculate cloud downward emission temperature
    analytical solution to radiative energy balance in cloud layer:
    σ*εcloud*Tsurf^4 = σ*εcloud*Tcloud↑^4 + σ*εcloud*Tcloud↓^4
    assuming ΔTcloud ≥ 0, Tsurf > 0, Tcloud↑ > 0, and 
    ΔTcloud = Tcloud↓ - Tcloud↑ 
    inputs:
        * ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
        * Tsurf [K] - surface temperature 
    outputs:
        * Tcloud↓ [K] - downward cloud emission temperature 
    """
    0.5*(ΔTcloud + sqrt(-3*ΔTcloud^2 + 2*sqrt(2)*sqrt(ΔTcloud^4 + Tsurf^4)))
end

# FUNCTIONS TO DIAGNOSE CLOUD PROPERTIES ##########################

###################################################################

# FUNCTIONS TO CHARACTERIZE MODEL SOLUTIONS ########################
function checkendsol_radbal(sol,ΔTthres,Tsurfeq,Δt̂,β,ΔTcloud,α,Rstar,aplanet,Tstar,ratA;λLW=4.5e-6,λSW=0.5e-6,isplot=false,figdir="",runname="",no_pts=20)
    """
    check the end state of integration for radiative balance version of model 
    """
    # set default values 
    endstateflag = 0 
    P = NaN

    # filter for times in the last Δt̂ 
    # (excluding last step since this is how end criterion is determined)
    # (last two steps are usually same time due to callbacks... 
    # turning off callback saving seems to cause issues so just index here)
    filt = sol.t .>= (sol.t[end]-Δt̂) 
    if sol.t[end] == sol.t[end-1]
        filt[(end-1):end] .= false
    else
        filt[end] = false
    end

    # get prognostic variables in the last Δt̂
    endTsurfs = sol[1,filt]
    endτSWs = sol[2,filt]
    endLs = sol[3,filt]

    # calculate LW brightness temperature 
    endTcloud_ups = calcTcloudup.(ΔTcloud,endTsurfs)
    endTbLWs = calcTbrightλ_LW(λLW,endTsurfs,endτSWs,endTcloud_ups,β)
    # calculate SW brightness temperature 
    endTbSWs = calcTbrightλ_SW.(λSW,endTsurfs,endτSWs,α,Rstar,aplanet,Tstar,ratA)

    
    # calculate minimum and maximum values for output 
    TbLWmin = minimum(endTbLWs)
    TbLWmax = maximum(endTbLWs)

    TbSWmin = minimum(endTbSWs)
    TbSWmax = maximum(endTbSWs)

    Tsurfmin = minimum(endTsurfs)
    Tsurfmax = maximum(endTsurfs)

    τSWmin = minimum(endτSWs)
    τSWmax = maximum(endτSWs)

    Lmin = minimum(endLs)
    Lmax = maximum(endLs)

    # conditions used in isstoplong with additional endstateflag and plotting details 
    if (Tsurfmin <= (Tsurfeq + ΔTthres)) && (Tsurfmax>=(Tsurfeq - ΔTthres)) # check if T solution bounding Teq 
        if (Tsurfmin >= (Tsurfeq - ΔTthres)) && (Tsurfmax <= (Tsurfeq + ΔTthres)) # check if within threshold
            endstateflag = 1
        else # check what oscillations (if any) present 
            findperiod0 = (t̂) -> findperiod0long(t̂,sol,Tsurfeq)
            endt̂s = sol.t[filt]
            endΔTs = endTsurfs .- Tsurfeq
            filt_pos_endΔT = endΔTs .> 0
            filt_neg_endΔT = endΔTs .< 0
            if (sum(filt_pos_endΔT) > 0) && (sum(filt_neg_endΔT) > 0)
                t̂start = endt̂s[filt_pos_endΔT][1]
                t̂end = endt̂s[filt_neg_endΔT][end]
                # note no_pts handles number of initial samples of interval for 0s
                t̂whereTeq = find_zeros(findperiod0,t̂start,t̂end;no_pts=no_pts) # Int(round(Δt̂))
                nt̂whereTeq = length(t̂whereTeq)
                if nt̂whereTeq>5
                    t̂Pstart = t̂whereTeq[3:2:end]
                    t̂Pend = t̂whereTeq[1:2:(end - 2 + nt̂whereTeq%2)]
                    Ps = t̂Pstart .- t̂Pend
                    P = mean(Ps)    
                    if std(Ps)/P > 1e-2 # period erratic 
                        endstateflag = 3
                        P = NaN 
                    else # regular period 
                        endstateflag = 2 
                    end
                    # plotting option 
                    if isplot
                        fig = Figure()
                        ax = Axis(fig[1,1],xlabel="time [delay times]",ylabel="surface temperature [K]")
                        if endstateflag == 2
                            ax.title = "period = $(round(P,sigdigits=3))"
                        else endstateflag == 3
                            ax.title = "erratic period"
                        end
                        lines!(ax,endt̂s,endTsurfs,color=:black)
                        vlines!(ax,t̂whereTeq,linestyle=:dash,color=:red)
                        hlines!(ax,Tsurfeq,linestyle=:dash,color=:red)
                        hspan!(ax,Tsurfeq-ΔTthres,Tsurfeq+ΔTthres,color=(:red,0.3))
                        save(figdir*"checkperiod"*runname*".pdf",fig)
                    end
                end
            end
        end
    end
    # possible plotting option
    if (endstateflag == 0) && isplot
        endt̂s = sol.t[filt]
        fig = Figure()        
        ax = Axis(fig[1,1],xlabel="time [delay times]",ylabel="surface temperature [K]")
        ax.title = "endstateflag = 0, retcode = $(sol.retcode)"
        scatter!(ax,endt̂s,endTsurfs,color=:black)
        hlines!(ax,Tsurfeq,linestyle=:dash,color=:red)
        hspan!(ax,Tsurfeq-ΔTthres,Tsurfeq+ΔTthres,color=(:red,0.3))
        save(figdir*"checkperiod"*runname*".pdf",fig)
    end

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax
end

function checkendsol_const(sol,ΔTthres,Tsurfeq,Δt̂,β,Tcloud,α,Rstar,aplanet,Tstar,ratA;λLW=4.5e-6,λSW=0.5e-6,isplot=false,figdir="",runname="",no_pts=20)
    """
    check the end state of integration for constant version of model 
    """
    # set default values 
    endstateflag = 0 
    P = NaN

    # filter for times in the last Δt̂ 
    # (excluding last step since this is how end criterion is determined)
    # (last two steps are usually same time due to callbacks... 
    # turning off callback saving seems to cause issues so just index here)
    filt = sol.t .>= (sol.t[end]-Δt̂) 
    if sol.t[end] == sol.t[end-1]
        filt[(end-1):end] .= false
    else
        filt[end] = false
    end

    # get prognostic variables in the last Δt̂
    endTsurfs = sol[1,filt]
    endτSWs = sol[2,filt]
    endLs = sol[3,filt]

    # calculate LW brightness temperature 
    endTcloud_ups = Tcloud
    endTbLWs = calcTbrightλ_LW(λLW,endTsurfs,endτSWs,endTcloud_ups,β)
    # calculate SW brightness temperature 
    endTbSWs = calcTbrightλ_SW.(λSW,endTsurfs,endτSWs,α,Rstar,aplanet,Tstar,ratA)

    
    # calculate minimum and maximum values for output 
    TbLWmin = minimum(endTbLWs)
    TbLWmax = maximum(endTbLWs)

    TbSWmin = minimum(endTbSWs)
    TbSWmax = maximum(endTbSWs)

    Tsurfmin = minimum(endTsurfs)
    Tsurfmax = maximum(endTsurfs)

    τSWmin = minimum(endτSWs)
    τSWmax = maximum(endτSWs)

    Lmin = minimum(endLs)
    Lmax = maximum(endLs)

    # conditions used in isstoplong with additional endstateflag and plotting details 
    if (Tsurfmin <= (Tsurfeq + ΔTthres)) && (Tsurfmax>=(Tsurfeq - ΔTthres)) # check if T solution bounding Teq 
        if (Tsurfmin >= (Tsurfeq - ΔTthres)) && (Tsurfmax <= (Tsurfeq + ΔTthres)) # check if within threshold
            endstateflag = 1
        else # check what oscillations (if any) present 
            findperiod0 = (t̂) -> findperiod0long(t̂,sol,Tsurfeq)
            endt̂s = sol.t[filt]
            endΔTs = endTsurfs .- Tsurfeq
            filt_pos_endΔT = endΔTs .> 0
            filt_neg_endΔT = endΔTs .< 0
            if (sum(filt_pos_endΔT) > 0) && (sum(filt_neg_endΔT) > 0)
                t̂start = endt̂s[filt_pos_endΔT][1]
                t̂end = endt̂s[filt_neg_endΔT][end]
                # note no_pts handles number of initial samples of interval for 0s
                t̂whereTeq = find_zeros(findperiod0,t̂start,t̂end;no_pts=no_pts) # Int(round(Δt̂))
                nt̂whereTeq = length(t̂whereTeq)
                if nt̂whereTeq>5
                    t̂Pstart = t̂whereTeq[3:2:end]
                    t̂Pend = t̂whereTeq[1:2:(end - 2 + nt̂whereTeq%2)]
                    Ps = t̂Pstart .- t̂Pend
                    P = mean(Ps)    
                    if std(Ps)/P > 1e-2 # period erratic 
                        endstateflag = 3
                        P = NaN 
                    else # regular period 
                        endstateflag = 2 
                    end
                    # plotting option 
                    if isplot
                        fig = Figure()
                        ax = Axis(fig[1,1],xlabel="time [delay times]",ylabel="surface temperature [K]")
                        if endstateflag == 2
                            ax.title = "period = $(round(P,sigdigits=3))"
                        else endstateflag == 3
                            ax.title = "erratic period"
                        end
                        lines!(ax,endt̂s,endTsurfs,color=:black)
                        vlines!(ax,t̂whereTeq,linestyle=:dash,color=:red)
                        hlines!(ax,Tsurfeq,linestyle=:dash,color=:red)
                        hspan!(ax,Tsurfeq-ΔTthres,Tsurfeq+ΔTthres,color=(:red,0.3))
                        save(figdir*"checkperiod"*runname*".pdf",fig)
                    end
                end
            end
        end
    end
    # possible plotting option
    if (endstateflag == 0) && isplot
        endt̂s = sol.t[filt]
        fig = Figure()        
        ax = Axis(fig[1,1],xlabel="time [delay times]",ylabel="surface temperature [K]")
        ax.title = "endstateflag = 0, retcode = $(sol.retcode)"
        scatter!(ax,endt̂s,endTsurfs,color=:black)
        hlines!(ax,Tsurfeq,linestyle=:dash,color=:red)
        hspan!(ax,Tsurfeq-ΔTthres,Tsurfeq+ΔTthres,color=(:red,0.3))
        save(figdir*"checkperiod"*runname*".pdf",fig)
    end

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax
end

# FUNCTIONS TO CHARACTERIZE MODEL SOLUTIONS ########################

####################################################################


# FUNCTIONS FOR BRIGHTNESS TEMPERATURE ############################
function calcBλ(λ,T)
    """
    calculate Planck spectral radiance in terms of wavelength 
    inputs:
        * λ [m] - wavelength of light 
        * T [K] - temperature 
    outputs:
        * Bλ [W sr⁻¹ m⁻³] - Planck blackbody spectral radiance
    """
    2*h*c^2/λ^5 /(exp(h*c/(λ*kB*T))-1.)
end

function calcTbrightλ(λ,I)
    """
    calculate brightness temperature for a single wavelength  
    inputs:
        * λ [m] - wavelength of light 
        * I [W sr⁻¹ m⁻³] - spectral radiance
    output:
        * Tbright [K] - brightness temperature 
    """
    h*c/(kB*λ)*(log(1. + 2*h*c^2/(I*λ^5)))^(-1)
end

function calcTbrightλ_LW(λ,Tsurf,τSW,Tcloud_up,β)
    """
    calculate brightness temperature Tbright for longwave assumptions 
    inputs:
        * λ [m] - wavelength of light 
        * Tsurf [K] - surface temperature 
        * τSW [ ] - cloud shortwave opacity (used only to calculate cloud longwave opacity)
        * Tcloud_up [K] - upward cloud emission temperature 
        * β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
    output:
        * Tbright [K] - brightness temperature 
    """
    # calculate longwave cloud emissivity
    εcloud = calcεcloud.(τSW,β) # [ ]
    # calculate spectral radiance (only considering planetary emission)
    I = εcloud.*calcBλ.(λ,Tcloud_up) .+ (1. .- εcloud).*calcBλ.(λ,Tsurf) # [W sr⁻¹ m⁻³]
    calcTbrightλ.(λ,I)
end

function calcTbrightλ_SW(λ,Tsurf,τSW,α,Rstar,aplanet,Tstar,ratA)
    """
    calculate brightness temperature Tbright for shortwave assumptions 
    inputs:
        * λ [m] - wavelength of light 
        * Tsurf [K] - surface temperature 
        * τSW [ ] - cloud shortwave opacity (used only to calculate cloud longwave opacity)
        * α [ ] - albedo parameter
        * Rstar [m] - radius of star 
        * aplanet [m] - planet semi-major axis 
        * Tstar [K] - temperature of star 
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
    output:
        * Tbright [K] - brightness temperature 
    """
    # calculate cloud SW albedo 
    Acloud = calcA.(τSW,α) # [ ]
    # calculate spectral radiance
    # including stellar reflection and planetary emission 
    # assuming SW cloud emissivity is 0 (i.e., all planetary SW emission is from surface)
    I = ((1 .- Acloud)*calcBλ.(λ,Tsurf) 
    .+ ratA.*Acloud.*calcBλ(λ,Tstar).*(Rstar/aplanet).^2) # [W sr⁻¹ m⁻³]
    calcTbrightλ.(λ,I)
end

function calcTbrightλ_refonly(λ,τSW,α,Rstar,aplanet,Tstar,ratA)
    """
    calculate brightness temperature Tbright only considering reflection of stellar radiation 
    inputs:
        * λ [m] - wavelength of light 
        * τSW [ ] - cloud shortwave opacity (used only to calculate cloud longwave opacity)
        * α [ ] - albedo parameter 
        * Rstar [m] - radius of star 
        * aplanet [m] - planet semi-major axis 
        * Tstar [K] - temperature of star 
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
    output:
        * Tbright [K] - brightness temperature 
    """
    # calculate cloud SW albedo 
    Acloud = calcA.(τSW,α) # [ ]
    # calculate spectral radiance only including stellar reflection
    I = ratA.*Acloud.*calcBλ.(λ,Tstar).*(Rstar/aplanet).^2 # [W sr⁻¹ m⁻³]
    calcTbrightλ.(λ,I)
end

# FUNCTIONS FOR BRIGHTNESS TEMPERATURE ############################

###################################################################

# FUNCTIONS TO RUN MODEL ##########################################

function calcsolprop_radbal(p;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fTsurf₀=0.99,fτSW₀=0.99,addTsurf₀=0.,addτSW₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e4,ΔTthres=1.,
    Rstar=R55cncA,aplanet=a55cnce,Tstar=T55cncA,ratA=1.,isplot=false,figdir="",runname="")
    """
    calculate model solution properties for radiative balance model 
    inputs:
        * p [Array, dim:11] - array of parameters for model 
            + Π₁ [Pa⁻¹] - bulk parameter 1 
            + Π₂ [ ] - bulk parameter 2 
            + Π₃ [kg s⁻³ K⁻¹] - bulk parameter 3 
            + T_ref [T] - reference temperature for Clausius-Clapeyron 
            + p_ref [Pa] - reference pressure for Clausius-Clapeyron
            + α [ ] - albedo parameter 
            + S₀ [W m⁻²] - incident stellar insolation 
            + f [ ] - heat redistribution factor 
            + β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
            + ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
            + Tmagma_solidus [K] - solidus temperature of magma
        * maxiters [Int] - maximum iterations of the DDE solver 
            + optional, default value: 1e7 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
        > note, Tsurf(t̂=0) = fTsurf₀ * Tsurfeq + addTsurf₀
            * fTsurf₀ [ ] - multiplicative factor for Tsurf for initial condition 
                + optional, default value: 0.99
            * addTsurf₀ [K] - additive factor for Tsurf for initial condition 
                + optional, default value: 0 K
        > note, τSW(t̂=0) = fτSW₀ * τSWeq + addτSW₀
            * fτSW₀ [ ] - multiplicative factor for shortwave optical depth for initial condition
                + optional, default value: 0.99
            * addτSW₀ [ ] - additive factor for shortwave optical depth for initial condition
                + optional, default value: 0
        * t̂check1 [delay time] - first time to check DDE ending conditions 
            + optional, default value: 300 delay times
        * Δt̂ [delay time] - time duration to use to check ending conditions, note Δt̂ ≤ t̂check1
            + optional, default value: 100 delay times 
        * t̂end [delay time] - end time for integration if DDE solution does not stabilize
            note, important in chaos regimes of parameter space !
            + optional, default value: 1e4 delay time 
        * ΔTthres [K] - surface temperature buffer around fixed point for determining end state 
            note, lower values will cause longer integration
            do not set below about 10*reltol*Teq 
            + optional, default value: 1 K 
        * Rstar [m] - radius of star, used for SW brightness temperature diagnostic 
            + optional, default value: R55cncA (radius of 55 Cnc A)
        * aplanet [m] - planet semi-major axis, used for SW brightness temperature diagnostic
            + optional, default value: a55cnce (semi-major axis of 55 Cnc e)
        * Tstar [K] - effective temperature of star, used for SW brightness temperature diagnostic
            + optional, default value: T55cncA (effective temperature of 55 Cnc A)
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
            + optional, default value: 1
        * isplot [Bool] - whether to make plots associated with diagnosing solution end state
            note, do not set to true for a large parameter sweep  
            + optional, default value: false 
        * figdir [String] - figure directory if isplot=true 
            + optional, default value: ""
        * runname [String] - name of run for saving figure if isplot=true 
            + optional, default value: ""

    outputs:
        * endstateflag [Int] - integer flag associated with integration end state 
        * P [delay time] - period of limit cycle if applicable (otherwise NaN)
        * Tsurfmin [K] - minimum surface temperature in last Δt̂ of integration time 
        * Tsurfmax [K] - maximum surface temperature in last Δt̂ of integration time 
        * TbLWmin [K] - minimum LW brightness temperature (λ = 4.5 μm) in last Δt̂ of integration time 
        * TbLWmax [K] - maximum LW brightness temperature (λ = 4.5 μm) in last Δt̂ of integration time 
        * TbSWmin [K] - minimum SW brightness temperature (λ = 500 nm) in last Δt̂ of integration time
        * TbSWmax [K] - maximum LW brightness temperature (λ = 500 nm) in last Δt̂ of integration time
        * τSWmin [ ] - minimum SW cloud optical depth in last Δt̂ of integration time 
        * τSWmax [ ] - maximum SW cloud optical depth in last Δt̂ of integration time
        * Lmin [J m⁻²] - minimum magma ocean column accumulated latent heat of fusion in last Δt̂ of integration time
        * Lmax [J m⁻²] - maximum magma ocean column accumulated latent heat of fusion in last Δt̂ of integration time
    """
    # throw error if Δt̂ > t̂check1
    if Δt̂ > t̂check1
        @show Δt̂ t̂check1
        error("Δt̂ > t̂check1 not permitted! set Δt̂ ≤ t̂check1!")
    end

    # expand parameters 
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = p

    Tsurfeq,τSWeq = findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)

    # don't integeate if Tsurfeq<Tmagma_solidus
    if Tsurfeq<Tmagma_solidus
        return -2,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
    end
    
    Tsurf₀ = Tsurfeq*fTsurf₀ + addTsurf₀
    τSW₀ = τSWeq*fτSW₀ + addτSW₀

    sol = solvedTτLdt̂_radbal(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,t̂end,Tsurf₀,τSW₀,Tsurfeq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)

    endstateflag = -1 
    P = NaN 
    Tsurfmin = NaN
    Tsurfmax = NaN
    TbLWmin = NaN
    TbLWmax = NaN
    TbSWmin = NaN
    TbSWmax = NaN
    τSWmin = NaN
    τSWmax = NaN
    Lmin = NaN
    Lmax = NaN

    # alert user if return code not successful 
    if SciMLBase.successful_retcode(sol)
        endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax = checkendsol_radbal(sol,ΔTthres,Tsurfeq,Δt̂,β,ΔTcloud,α,Rstar,aplanet,Tstar,ratA;isplot=isplot,figdir=figdir,runname=runname)
    else
        @info "unsuccessful integration!"
        @show p sol.retcode
    end

    # adjust endstateflag if hit t̂end 
    if sol.t[end] == t̂end
        if endstateflag == 0
            endstateflag = 4
        elseif endstateflag == 1
            endstateflag = 5
        elseif endstateflag == 2
            endstateflag = 6
        elseif endstateflag == 3
            endstateflag = 7
        end
    end

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax
end


function calcsolprop_const(p;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fTsurf₀=0.99,fτSW₀=0.99,addTsurf₀=0.,addτSW₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e4,ΔTthres=1.,
    Rstar=R55cncA,aplanet=a55cnce,Tstar=T55cncA,ratA=1.,isplot=false,figdir="",runname="")
    """
    calculate model solution properties for constant model 
    inputs:
        * p [Array, dim:11] - array of parameters for model 
            + Π₁ [Pa⁻¹] - bulk parameter 1 
            + Π₂ [ ] - bulk parameter 2 
            + Π₃ [kg s⁻³ K⁻¹] - bulk parameter 3 
            + T_ref [T] - reference temperature for Clausius-Clapeyron 
            + p_ref [Pa] - reference pressure for Clausius-Clapeyron
            + α [ ] - albedo parameter 
            + S₀ [W m⁻²] - incident stellar insolation 
            + f [ ] - heat redistribution factor 
            + β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
            + Tcloud [K] - constant cloud downward and upward emission temperature (Tcloud = Tcloud↓ = Tcloud↑)
            + Tmagma_solidus [K] - solidus temperature of magma
        * maxiters [Int] - maximum iterations of the DDE solver 
            + optional, default value: 1e7 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
        > note, Tsurf(t̂=0) = fTsurf₀ * Tsurfeq + addTsurf₀
            * fTsurf₀ [ ] - multiplicative factor for Tsurf for initial condition 
                + optional, default value: 0.99
            * addTsurf₀ [K] - additive factor for Tsurf for initial condition 
                + optional, default value: 0 K
        > note, τSW(t̂=0) = fτSW₀ * τSWeq + addτSW₀
            * fτSW₀ [ ] - multiplicative factor for shortwave optical depth for initial condition
                + optional, default value: 0.99
            * addτSW₀ [ ] - additive factor for shortwave optical depth for initial condition
                + optional, default value: 0
        * t̂check1 [delay time] - first time to check DDE ending conditions 
            + optional, default value: 300 delay times
        * Δt̂ [delay time] - time duration to use to check ending conditions, note Δt̂ ≤ t̂check1
            + optional, default value: 100 delay times 
        * t̂end [delay time] - end time for integration if DDE solution does not stabilize
            note, important in chaos regimes of parameter space !
            + optional, default value: 1e4 delay time 
        * ΔTthres [K] - surface temperature buffer around fixed point for determining end state 
            note, lower values will cause longer integration
            do not set below about 10*reltol*Teq 
            + optional, default value: 1 K 
        * Rstar [m] - radius of star, used for SW brightness temperature diagnostic 
            + optional, default value: R55cncA (radius of 55 Cnc A)
        * aplanet [m] - planet semi-major axis, used for SW brightness temperature diagnostic
            + optional, default value: a55cnce (semi-major axis of 55 Cnc e)
        * Tstar [K] - effective temperature of star, used for SW brightness temperature diagnostic
            + optional, default value: T55cncA (effective temperature of 55 Cnc A)
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
            + optional, default value: 1
        * isplot [Bool] - whether to make plots associated with diagnosing solution end state
            note, do not set to true for a large parameter sweep  
            + optional, default value: false 
        * figdir [String] - figure directory if isplot=true 
            + optional, default value: ""
        * runname [String] - name of run for saving figure if isplot=true 
            + optional, default value: ""

    outputs:
        * endstateflag [Int] - integer flag associated with integration end state 
        * P [delay time] - period of limit cycle if applicable (otherwise NaN)
        * Tsurfmin [K] - minimum surface temperature in last Δt̂ of integration time 
        * Tsurfmax [K] - maximum surface temperature in last Δt̂ of integration time 
        * TbLWmin [K] - minimum LW brightness temperature (λ = 4.5 μm) in last Δt̂ of integration time 
        * TbLWmax [K] - maximum LW brightness temperature (λ = 4.5 μm) in last Δt̂ of integration time 
        * TbSWmin [K] - minimum SW brightness temperature (λ = 500 nm) in last Δt̂ of integration time
        * TbSWmax [K] - maximum LW brightness temperature (λ = 500 nm) in last Δt̂ of integration time
        * τSWmin [ ] - minimum SW cloud optical depth in last Δt̂ of integration time 
        * τSWmax [ ] - maximum SW cloud optical depth in last Δt̂ of integration time
        * Lmin [J m⁻²] - minimum magma ocean column accumulated latent heat of fusion in last Δt̂ of integration time
        * Lmax [J m⁻²] - maximum magma ocean column accumulated latent heat of fusion in last Δt̂ of integration time
    """
    # throw error if Δt̂ > t̂check1
    if Δt̂ > t̂check1
        @show Δt̂ t̂check1
        error("Δt̂ > t̂check1 not permitted! set Δt̂ ≤ t̂check1!")
    end

    # expand parameters 
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tmagma_solidus = p

    Tsurfeq,τSWeq = findTτeqnum_const(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud)

    # don't integeate if Tsurfeq<Tmagma_solidus
    if Tsurfeq<Tmagma_solidus
        return -2,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
    end

    # otherwise integrate 
    
    
    Tsurf₀ = Tsurfeq*fTsurf₀ + addTsurf₀
    τSW₀ = τSWeq*fτSW₀ + addτSW₀

    sol = solvedTτLdt̂_const(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,t̂end,Tsurf₀,τSW₀,Tsurfeq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)

    endstateflag = -1 
    P = NaN 
    Tsurfmin = NaN
    Tsurfmax = NaN
    TbLWmin = NaN
    TbLWmax = NaN
    TbSWmin = NaN
    TbSWmax = NaN
    τSWmin = NaN
    τSWmax = NaN
    Lmin = NaN
    Lmax = NaN

    # alert user if return code not successful 
    if SciMLBase.successful_retcode(sol)
        endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax = checkendsol_const(sol,ΔTthres,Tsurfeq,Δt̂,β,Tcloud,α,Rstar,aplanet,Tstar,ratA;isplot=isplot,figdir=figdir,runname=runname)
    else
        @info "unsuccessful integration!"
        @show p sol.retcode
    end

    # adjust endstateflag if hit t̂end 
    if sol.t[end] == t̂end
        if endstateflag == 0
            endstateflag = 4
        elseif endstateflag == 1
            endstateflag = 5
        elseif endstateflag == 2
            endstateflag = 6
        elseif endstateflag == 3
            endstateflag = 7
        end
    end

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax
end

function outputrun_radbal(p,outdir,runname;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fT₀=0.99,fτ₀=0.99,addT₀=0.,addτ₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e7,ΔTthres=1.,
    isverbose=true,fsteps=10,Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.)
    """
    run radiative balance model for two initial conditions and save many outputs to netcdf 
    note, can generate large output files size for limit cycles! 
    do not use for a large number of parameters at once
    
    inputs:
        * p [Array, dim:11] - array of parameters for model
            + Π₁ [Pa⁻¹] - bulk parameter 1 
            + Π₂ [ ] - bulk parameter 2 
            + Π₃ [kg s⁻³ K⁻¹] - bulk parameter 3 
            + T_ref [T] - reference temperature for Clausius-Clapeyron 
            + p_ref [Pa] - reference pressure for Clausius-Clapeyron
            + α [ ] - albedo parameter 
            + S₀ [W m⁻²] - incident stellar insolation 
            + f [ ] - heat redistribution factor 
            + β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
            + ΔTcloud [K] - difference between cloud downward and upward emission temperature (ΔTcloud = Tcloud↓ - Tcloud↑)
            + Tmagma_solidus [K] - solidus temperature of magma
        * outdir [String] - directory to save outputs to 
        * runname [String] - name of run 
        * maxiters [Int] - maximum iterations of the DDE solver 
            + optional, default value: 1e7 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
        > note, Tsurf(t̂=0) = fTsurf₀ * Tsurfeq + addTsurf₀
            * fTsurf₀ [ ] - multiplicative factor for Tsurf for initial condition 
                + optional, default value: 0.99
            * addTsurf₀ [K] - additive factor for Tsurf for initial condition 
                + optional, default value: 0 K
        > note, τSW(t̂=0) = fτSW₀ * τSWeq + addτSW₀
            * fτSW₀ [ ] - multiplicative factor for shortwave optical depth for initial condition
                + optional, default value: 0.99
            * addτSW₀ [ ] - additive factor for shortwave optical depth for initial condition
                + optional, default value: 0
        * t̂check1 [delay time] - first time to check DDE ending conditions 
            + optional, default value: 300 delay times
        * Δt̂ [delay time] - time duration to use to check ending conditions, note Δt̂ ≤ t̂check1
            + optional, default value: 100 delay times 
        * t̂end [delay time] - end time for integration if DDE solution does not stabilize
            note, important in chaos regimes of parameter space !
            + optional, default value: 1e4 delay time 
        * ΔTthres [K] - surface temperature buffer around fixed point for determining end state 
            note, lower values will cause longer integration
            do not set below about 10*reltol*Teq 
            + optional, default value: 1 K 
        * isverbose [Bool] - whether include extra outputs  
            + optional, default value: true
        * fsteps [Int] - factor for number of time steps outputted relative to number of time steps taken by integrator
            note, can be important for plotting smooth limit cycle phase diagrams
            + optional, default value: 10 
        * Rstar [m] - radius of star, used for SW brightness temperature diagnostic 
            + optional, default value: R55cncA (radius of 55 Cnc A)
        * aplanet [m] - planet semi-major axis, used for SW brightness temperature diagnostic
            + optional, default value: a55cnce (semi-major axis of 55 Cnc e)
        * Tstar [K] - effective temperature of star, used for SW brightness temperature diagnostic
            + optional, default value: T55cncA (effective temperature of 55 Cnc A)
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
            + optional, default value: 1
    """
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = p

    Teq,τeq = findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)

    # handle Teq<Tmagma_solidus case
    if Teq<Tmagma_solidus
        @show Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus
        error("Teq<Tmagma_solidus")
    end

    # inner initial coniditions 
    T₀_inner = Teq*fT₀ + addT₀
    τ₀_inner = τeq*fτ₀ + addτ₀

    sol_inner = solvedTτLdt̂_radbal(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,t̂end,T₀_inner,τ₀_inner,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)
  
    writenc_radbal(outdir,runname*"_inner",sol_inner,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Teq,τeq,reltol,abstol,Δt̂,ΔTthres;fsteps=fsteps,isverbose=isverbose,Tstar=Tstar,aplanet=aplanet,Rstar=Rstar,ratA=ratA)

    # outer initial conditions 
    T₀_outer = Tmagma_solidus + 1.
    τ₀_outer = 1e-6 

    sol_outer = solvedTτLdt̂_radbal(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,t̂end,T₀_outer,τ₀_outer,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)

    writenc_radbal(outdir,runname*"_outer",sol_outer,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Teq,τeq,reltol,abstol,Δt̂,ΔTthres;fsteps=fsteps,isverbose=isverbose,Tstar=Tstar,aplanet=aplanet,Rstar=Rstar,ratA=ratA)
    
    nothing 
end

function outputrun_const(p,outdir,runname;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fT₀=0.99,fτ₀=0.99,addT₀=0.,addτ₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e7,ΔTthres=1.,
    isverbose=true,fsteps=10,Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.)
    """
    run constant model for two initial conditions and save many outputs to netcdf 
    note, can generate large output files size for limit cycles! 
    do not use for a large number of parameters at once
    
    inputs:
        * p [Array, dim:11] - array of parameters for model
            + Π₁ [Pa⁻¹] - bulk parameter 1 
            + Π₂ [ ] - bulk parameter 2 
            + Π₃ [kg s⁻³ K⁻¹] - bulk parameter 3 
            + T_ref [T] - reference temperature for Clausius-Clapeyron 
            + p_ref [Pa] - reference pressure for Clausius-Clapeyron
            + α [ ] - albedo parameter 
            + S₀ [W m⁻²] - incident stellar insolation 
            + f [ ] - heat redistribution factor 
            + β [ ] - ratio of longwave cloud optical depth to shortwave cloud optical depth (i.e., τLW/τSW)
            + Tcloud [K] - constant cloud downward and upward emission temperature (Tcloud = Tcloud↓ = Tcloud↑)
            + Tmagma_solidus [K] - solidus temperature of magma
        * outdir [String] - directory to save outputs to 
        * runname [String] - name of run 
        * maxiters [Int] - maximum iterations of the DDE solver 
            + optional, default value: 1e7 
        * alg [OrdinaryDiffEq integration algorithm]
            note, see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ for options
            + optional, default value: KenCarp4()
        * reltol [Float] - relative tolerance of the DDE solver 
            + optional, default value: 1e-8
        * abstol [Float] - absolute tolerance of the DDE solver 
            + optional, default value: 1e-10
        > note, Tsurf(t̂=0) = fTsurf₀ * Tsurfeq + addTsurf₀
            * fTsurf₀ [ ] - multiplicative factor for Tsurf for initial condition 
                + optional, default value: 0.99
            * addTsurf₀ [K] - additive factor for Tsurf for initial condition 
                + optional, default value: 0 K
        > note, τSW(t̂=0) = fτSW₀ * τSWeq + addτSW₀
            * fτSW₀ [ ] - multiplicative factor for shortwave optical depth for initial condition
                + optional, default value: 0.99
            * addτSW₀ [ ] - additive factor for shortwave optical depth for initial condition
                + optional, default value: 0
        * t̂check1 [delay time] - first time to check DDE ending conditions 
            + optional, default value: 300 delay times
        * Δt̂ [delay time] - time duration to use to check ending conditions, note Δt̂ ≤ t̂check1
            + optional, default value: 100 delay times 
        * t̂end [delay time] - end time for integration if DDE solution does not stabilize
            note, important in chaos regimes of parameter space !
            + optional, default value: 1e4 delay time 
        * ΔTthres [K] - surface temperature buffer around fixed point for determining end state 
            note, lower values will cause longer integration
            do not set below about 10*reltol*Teq 
            + optional, default value: 1 K 
        * isverbose [Bool] - whether include extra outputs  
            + optional, default value: true
        * fsteps [Int] - factor for number of time steps outputted relative to number of time steps taken by integrator
            note, can be important for plotting smooth limit cycle phase diagrams
            + optional, default value: 10 
        * Rstar [m] - radius of star, used for SW brightness temperature diagnostic 
            + optional, default value: R55cncA (radius of 55 Cnc A)
        * aplanet [m] - planet semi-major axis, used for SW brightness temperature diagnostic
            + optional, default value: a55cnce (semi-major axis of 55 Cnc e)
        * Tstar [K] - effective temperature of star, used for SW brightness temperature diagnostic
            + optional, default value: T55cncA (effective temperature of 55 Cnc A)
        * ratA [ ] - ratio of geometric albedo to substellar point albedo 
            + optional, default value: 1
    """
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tmagma_solidus = p

    Teq,τeq = findTτeqnum_const(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud)

    # handle Teq<Tmagma_solidus case
    if Teq<Tmagma_solidus
        @show Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tmagma_solidus
        error("Teq<Tmagma_solidus")
    end

    # inner initial coniditions 
    T₀_inner = Teq*fT₀ + addT₀
    τ₀_inner = τeq*fτ₀ + addτ₀

    sol_inner = solvedTτLdt̂_const(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,t̂end,T₀_inner,τ₀_inner,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)
  
    writenc_const(outdir,runname*"_inner",sol_inner,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Teq,τeq,reltol,abstol,Δt̂,ΔTthres;fsteps=fsteps,isverbose=isverbose,Tstar=Tstar,aplanet=aplanet,Rstar=Rstar,ratA=ratA)

    # outer initial conditions 
    T₀_outer = Tmagma_solidus + 1.
    τ₀_outer = 1e-6 

    sol_outer = solvedTτLdt̂_const(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,t̂end,T₀_outer,τ₀_outer,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)

    writenc_const(outdir,runname*"_outer",sol_outer,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Teq,τeq,reltol,abstol,Δt̂,ΔTthres;fsteps=fsteps,isverbose=isverbose,Tstar=Tstar,aplanet=aplanet,Rstar=Rstar,ratA=ratA)
    
    nothing 
end


function checksol_radbal(p,figdir,runname;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fT₀=0.99,fτ₀=0.99,addT₀=0.,addτ₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e4,ΔTthres=1.,
    Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.,Δt̂plot=15.,λ_LW=4.5,λ_SW=0.5e-6,λ_SWname="0.5",lw=2,fsteps=10,
    isplottimesteps=false)
    """
    run radiative balance model, make plots, and return solution object 
    """
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = p


    Teq,τeq = findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)

    # handle Teq<Tmagma_solidus case
    if Teq<Tmagma_solidus
        @show Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus
        error("Teq<Tmagma_solidus")
    end
    

    T₀ = Teq*fT₀ + addT₀
    τ₀ = τeq*fτ₀ + addτ₀

    sol = solvedTτLdt̂_radbal(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,ΔTcloud,t̂end,T₀,τ₀,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)

    if SciMLBase.successful_retcode(sol) # avoid plotting maxiters errors 
    
        try 
            filt = sol.t .>= max(0,sol.t[end]-Δt̂plot)

            if sum(filt)<2
                filt[:] .= true
            end

            tfilt = sol.t[filt]

            tsteps = unique(tfilt)
            nsteps = length(tsteps)
            interp_linear = linear_interpolation(1:nsteps, tsteps)
            ts_interp = interp_linear(LinRange(1,nsteps,nsteps*fsteps))


            Tfilt = sol.(ts_interp,idxs=1)
            τSWfilt = sol.(ts_interp,idxs=2)
            # don't fail plotting from negative / 0 values (only occurs for interpolation)
            τSWfilt[τSWfilt.<abstol] .= NaN
    
            Lfilt = sol.(ts_interp,idxs=3)
            Tcloudupfilt = calcTcloudup.(ΔTcloud,Tfilt)
            Tbright_4p5 = calcTbrightλ_LW.(4.5e-6,Tfilt,τSWfilt,Tcloudupfilt,β)
            Tbright_1 = calcTbrightλ_SW.(λ_SW,Tfilt,τSWfilt,α,Rstar,aplanet,Tstar,ratA)

            τSWfilt = sol[2,filt]
            # don't fail plotting from negative / 0 values (only occurs for interpolation)
            τSWfilt[τSWfilt.<abstol] .= NaN

            tfilt = ts_interp .- ts_interp[1]

            # plot T bright LW and SW phase space 
            fig = Figure() 
            Tbrightlabel_1 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=$(λ_SWname)",font=:italic),"μm",rich(")",font=:italic)," [K]")
            Tbrightlabel_4p5 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=4.5",font=:italic),"μm",rich(")",font=:italic)," [K]")
            ax = Axis(fig[1,1],xlabel=Tbrightlabel_4p5,ylabel=Tbrightlabel_1)
            lines!(ax,Tbright_4p5,Tbright_1,color=:black,linewidth=lw)
            save(figdir*"Tbright1v4p5_run$(runname).pdf",fig)

            # plot Tsurf and τSW vs t 
            fig = Figure() 
            axT = Axis(fig[1,1],xlabel="time [delay time]",ylabel=rich(rich("T",font=:italic),subscript("surf")," [K]"))
            if tfilt[1]!=tfilt[end]
                xlims!(axT,tfilt[1],tfilt[end])
                axT.xticks = tfilt[1]:tfilt[end]
            end
            lines!(axT,tfilt,Tfilt,linewidth=lw,color=:black)
            if isplottimesteps
                scatter!(axT,sol.t[filt] .- sol.t[filt][1],sol[1,filt],color=:blue,marker=:xcross,markersize=5)
            end
            hlines!(axT,Teq,linestyle=:dash,color=:red)
            hspan!(axT,Teq-ΔTthres,Teq+ΔTthres,color=(:red,0.25))
            axτ = Axis(fig[2,1],xlabel="time [delay time]",ylabel=rich(rich("τ",font=:italic),subscript("SW")," [ ]"),yscale=log10)
            if tfilt[1]!=tfilt[end]
                xlims!(axτ,tfilt[1],tfilt[end])
                axτ.xticks = tfilt[1]:tfilt[end]
            end
            lines!(axτ,sol.t[filt] .- sol.t[filt][1],τSWfilt,linewidth=lw,color=:black)
            hlines!(axτ,τeq,linestyle=:dash,color=:red)
            save(figdir*"TsurfτSWvt_run$(runname).pdf",fig)

            # plot L with coloring for Tsurf  
            filt_Tsolidus = abs.(Tfilt .- Tmagma_solidus) .< 10*reltol
            filt_Tsolidus_1K = (.!filt_Tsolidus) .&& (abs.(Tfilt .- Tmagma_solidus) .< 1.)
            filt_gtTsolidus = Tfilt .> (Tmagma_solidus+1)
            filt_ltTsolidus = Tfilt .< (Tmagma_solidus .-1)
            ms = 5
            if sum(filt_Tsolidus)>0
                fig = Figure()
                ax = Axis(fig[1,1],xlabel="time [delay time]",ylabel=rich(rich("L",font=:italic)," [J m⁻²]"))
                if tfilt[1]!=tfilt[end]
                    xlims!(ax,tfilt[1],tfilt[end])
                    ax.xticks = tfilt[1]:tfilt[end]
                end
                lines!(ax,tfilt,Lfilt,color=:black,linewidth=2)
                scatter!(ax,tfilt[filt_Tsolidus],Lfilt[filt_Tsolidus],color=:red,markersize=ms,
                label=rich(rich("T",font=:italic),subscript("surf")," = ",rich("T",font=:italic),subscript("solidus")))
                if sum(filt_Tsolidus_1K)>0
                    scatter!(ax,tfilt[filt_Tsolidus_1K],Lfilt[filt_Tsolidus_1K],color=:coral1,markersize=ms,
                    label=rich(rich("T",font=:italic),subscript("surf")," ≈ ",rich("T",font=:italic),subscript("solidus")))
                end
                if sum(filt_gtTsolidus)>0
                    scatter!(ax,tfilt[filt_gtTsolidus],Lfilt[filt_gtTsolidus],color=:blue,markersize=ms,marker=:xcross,
                    label=rich(rich("T",font=:italic),subscript("surf")," > ",rich("T",font=:italic),subscript("solidus")))
                end
                if sum(filt_ltTsolidus)>0
                    scatter!(ax,tfilt[filt_gtTsolidus],Lfilt[filt_gtTsolidus],color=:deepskyblue3,markersize=ms,marker=:xcross,
                    label=rich(rich("T",font=:italic),subscript("surf")," < ",rich("T",font=:italic),subscript("solidus")))
                end
                axislegend()
                save(figdir*"Lvt_Tsoldiushighlight_run$(runname).pdf",fig)
            end

        catch e 
            println("error in plotting: $e")
        end
    else
        @info "unsuccessful integration!"
        @show p sol.retcode
    end
    

    sol 
end

function checksol_const(p,figdir,runname;maxiters=1e7,alg=KenCarp4(),reltol=1e-8,
    abstol=1e-10,fT₀=0.99,fτ₀=0.99,addT₀=0.,addτ₀=0.,t̂check1=300.,Δt̂=100.,t̂end=1e4,ΔTthres=1.,
    Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.,Δt̂plot=15.,λ_LW=4.5,λ_SW=0.5e-6,λ_SWname="0.5",lw=2)
    """
    run constant model, make plots, and return solution object 

    """
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tmagma_solidus = p


    Teq,τeq = findTτeqnum_const(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,Tcloud)

    # handle Teq<Tmagma_solidus case
    if Teq<Tmagma_solidus
        @show Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tmagma_solidus
        error("Teq<Tmagma_solidus")
    end
    

    T₀ = Teq*fT₀ + addT₀
    τ₀ = τeq*fτ₀ + addτ₀

    sol = solvedTτLdt̂_const(Π₁,Π₂,Π₃,p_ref,T_ref,α,S₀,f,β,Tmagma_solidus,Tcloud,t̂end,T₀,τ₀,Teq,maxiters,alg,reltol,abstol,ΔTthres,Δt̂,t̂check1)
    
    try 
        filt = sol.t .>= max(0,sol.t[end]-Δt̂plot)

        if sum(filt)<2
            filt[:] .= true
        end

        tfilt = sol.t[filt]

        tfilt .-= tfilt[1]

        if tfilt[1] == tfilt[end]
            @show length(sol.t) sol.t[1] sol.t[end]
        end


        Tfilt = sol[1,filt]
        τSWfilt = sol[2,filt]
        Lfilt = sol[3,filt]
        Tcloudupfilt = Tcloud
        Tbright_4p5 = calcTbrightλ_LW.(4.5e-6,Tfilt,τSWfilt,Tcloudupfilt,β)
        Tbright_1 = calcTbrightλ_SW.(λ_SW,Tfilt,τSWfilt,α,Rstar,aplanet,Tstar,ratA)

        # plot T bright LW and SW phase space 
        fig = Figure() 
        Tbrightlabel_1 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=$(λ_SWname)",font=:italic),"μm",rich(")",font=:italic)," [K]")
        Tbrightlabel_4p5 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=4.5",font=:italic),"μm",rich(")",font=:italic)," [K]")
        ax = Axis(fig[1,1],xlabel=Tbrightlabel_4p5,ylabel=Tbrightlabel_1)
        lines!(ax,Tbright_4p5,Tbright_1,color=:black,linewidth=lw)
        save(figdir*"Tbright1v4p5_run$(runname).pdf",fig)

        # plot Tsurf and τSW vs t 
        fig = Figure() 
        axT = Axis(fig[1,1],xlabel="time [delay time]",ylabel=rich(rich("T",font=:italic),subscript("surf")," [K]"))
        if tfilt[1]!=tfilt[end]
            xlims!(axT,tfilt[1],tfilt[end])
        end
        lines!(axT,tfilt,Tfilt,linewidth=lw,color=:black)
        hlines!(axT,Teq,linestyle=:dash,color=:red)
        hspan!(axT,Teq-ΔTthres,Teq+ΔTthres,color=(:red,0.25))
        axτ = Axis(fig[2,1],xlabel="time [delay time]",ylabel=rich(rich("τ",font=:italic),subscript("SW")," [ ]"))
        if tfilt[1]!=tfilt[end]
            xlims!(axτ,tfilt[1],tfilt[end])
        end
        lines!(axτ,tfilt,τSWfilt,linewidth=lw,color=:black)
        hlines!(axτ,τeq,linestyle=:dash,color=:red)
        save(figdir*"TsurfτSWvt_run$(runname).pdf",fig)

        # plot L with coloring for Tsurf  
        filt_Tsolidus = abs.(Tfilt .- Tmagma_solidus) .< 10*reltol
        filt_Tsolidus_1K = (.!filt_Tsolidus) .&& (abs.(Tfilt .- Tmagma_solidus) .< 1.)
        filt_gtTsolidus = Tfilt .> (Tmagma_solidus+1)
        filt_ltTsolidus = Tfilt .< (Tmagma_solidus .-1)
        ms = 5
        if sum(filt_Tsolidus)>0
            fig = Figure()
            ax = Axis(fig[1,1],xlabel="time [delay time]",ylabel=rich(rich("L",font=:italic)," [J m⁻²]"))
            if tfilt[1]!=tfilt[end]
                xlims!(ax,tfilt[1],tfilt[end])
            end
            lines!(ax,tfilt,Lfilt,color=:black,linewidth=2)
            scatter!(ax,tfilt[filt_Tsolidus],Lfilt[filt_Tsolidus],color=:red,markersize=ms,
            label=rich(rich("T",font=:italic),subscript("surf")," = ",rich("T",font=:italic),subscript("solidus")))
            if sum(filt_Tsolidus_1K)>0
                scatter!(ax,tfilt[filt_Tsolidus_1K],Lfilt[filt_Tsolidus_1K],color=:coral1,markersize=ms,
                label=rich(rich("T",font=:italic),subscript("surf")," ≈ ",rich("T",font=:italic),subscript("solidus")))
            end
            if sum(filt_gtTsolidus)>0
                scatter!(ax,tfilt[filt_gtTsolidus],Lfilt[filt_gtTsolidus],color=:blue,markersize=ms,marker=:xcross,
                label=rich(rich("T",font=:italic),subscript("surf")," > ",rich("T",font=:italic),subscript("solidus")))
            end
            if sum(filt_ltTsolidus)>0
                scatter!(ax,tfilt[filt_gtTsolidus],Lfilt[filt_gtTsolidus],color=:deepskyblue3,markersize=ms,marker=:xcross,
                label=rich(rich("T",font=:italic),subscript("surf")," < ",rich("T",font=:italic),subscript("solidus")))
            end
            axislegend()
            save(figdir*"Lvt_Tsoldiushighlight_run$(runname).pdf",fig)
        end

    catch e 
        println("error in plotting: $e")
    end

    sol 
end

function setupparams4sweep(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
    log_min_Π3_search,log_max_Π3_search,min_β_search,max_β_search,min_Tcloud_var_search,max_Tcloud_var_search;
    α=0.5,S₀=3.4e6,f=1.,Tmagma_solidus=1400.,seed1=42,seed2=300)
    """
    setup model parameters for sweep 
    """
    # set up model parameter combinations to run 
    ps = zeros(11,nsamp)
    # assign values 
    ps[1:3,:] = 10 .^ doLHSND([log_min_Π1_search,log_min_Π2_search,log_min_Π3_search],[log_max_Π1_search,log_max_Π2_search,log_max_Π3_search],nsamp;seed=seed1)'
    ps[4,:] .= T_ref_SiO_0vap
    ps[5,:] .= p_ref_SiO_0vap
    ps[6,:] .= α
    ps[7,:] .= S₀
    ps[8,:] .= f
    ps[9:10,:] .= doLHSND([min_β_search,min_Tcloud_var_search],[max_β_search,max_Tcloud_var_search],nsamp;seed=seed2)' # β, ΔTcloud 
    ps[11,:] .= Tmagma_solidus
    ps
end

function setupparams4sweep_βlog(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
    log_min_Π3_search,log_max_Π3_search,log_min_β_search,log_max_β_search,min_Tcloud_var_search,max_Tcloud_var_search;
    α=0.5,S₀=3.4e6,f=1.,Tmagma_solidus=1400.,seed1=42,seed2=300,seed3=13)
    """
    setup model parameters for sweep 
    """
    # set up model parameter combinations to run 
    ps = zeros(11,nsamp)
    # assign values 
    ps[1:3,:] = 10 .^ doLHSND([log_min_Π1_search,log_min_Π2_search,log_min_Π3_search],[log_max_Π1_search,log_max_Π2_search,log_max_Π3_search],nsamp;seed=seed1)'
    ps[4,:] .= T_ref_SiO_0vap
    ps[5,:] .= p_ref_SiO_0vap
    ps[6,:] .= α
    ps[7,:] .= S₀
    ps[8,:] .= f
    ps[9,:] .= 10. .^ doLHSND([log_min_β_search],[log_max_β_search],nsamp;seed=seed2) # β
    ps[10,:] .= doLHSND([min_Tcloud_var_search],[max_Tcloud_var_search],nsamp;seed=seed3) # ΔTcloud
    ps[11,:] .= Tmagma_solidus
    ps
end

function setupparams4sweep_βconst(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
    log_min_Π3_search,log_max_Π3_search,min_Tcloud_var_search,max_Tcloud_var_search;
    β=0.,α=0.5,S₀=3.4e6,f=1.,Tmagma_solidus=1400.,seed1=42,seed2=300)
    """
    setup model parameters for sweep with β constant
    """
    # set up model parameter combinations to run 
    ps = zeros(11,nsamp)
    # assign values 
    ps[1:3,:] = 10 .^ doLHSND([log_min_Π1_search,log_min_Π2_search,log_min_Π3_search],[log_max_Π1_search,log_max_Π2_search,log_max_Π3_search],nsamp;seed=seed1)'
    ps[4,:] .= T_ref_SiO_0vap
    ps[5,:] .= p_ref_SiO_0vap
    ps[6,:] .= α
    ps[7,:] .= S₀
    ps[8,:] .= f
    ps[9,:] .= β
    ps[10,:] .= doLHSND([min_Tcloud_var_search],[max_Tcloud_var_search],nsamp;seed=seed2) # ΔTcloud 
    ps[11,:] .= Tmagma_solidus
    ps
end

function setupparams4sweep_Πonly(nsamp,log_min_Π1_search,log_max_Π1_search,log_min_Π2_search,log_max_Π2_search,
    log_min_Π3_search,log_max_Π3_search;
    Tcloud_var=0.,β=0.,α=0.5,S₀=3.4e6,f=1.,Tmagma_solidus=1400.,seed=42)
    """
    setup model parameters for sweep with β constant
    """
    # set up model parameter combinations to run 
    ps = zeros(11,nsamp)
    # assign values 
    ps[1:3,:] = 10 .^ doLHSND([log_min_Π1_search,log_min_Π2_search,log_min_Π3_search],[log_max_Π1_search,log_max_Π2_search,log_max_Π3_search],nsamp;seed=seed)'
    ps[4,:] .= T_ref_SiO_0vap
    ps[5,:] .= p_ref_SiO_0vap
    ps[6,:] .= α
    ps[7,:] .= S₀
    ps[8,:] .= f
    ps[9,:] .= β
    ps[10,:] .= Tcloud_var
    ps[11,:] .= Tmagma_solidus
    ps
end

# FUNCTIONS TO RUN MODEL ##########################################

###################################################################

# FUNCTIONS TO SAVE SOLUTIONS #####################################

function writenc_radbal(outdir,runname,sol,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tsurfeq,τeq,reltol,abstol,Δt̂,ΔTthres;
    fsteps::Int=10,isverbose=false,Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.)
    """
    save extensive individual run details for radiative balance model 
        + model parameters
        + integration parameters
        + solution properties
        + time evolving solution and diagnostics 
    to netcdf 

    this can generate a relatively large file
    do not run for large parameter sweep!
    """

    # convert solution into what outputted 
    tsteps = unique(sol.t)
    nsteps = length(tsteps)
    interp_linear = linear_interpolation(1:nsteps, tsteps)
    t̂s = interp_linear(LinRange(1,nsteps,nsteps*fsteps))
    Tsurfs = sol.(t̂s,idxs=1)
    
    τs = sol.(t̂s,idxs=2)
    τs = max.(0.,τs)
    Ls = sol.(t̂s,idxs=3)
    As = calcA.(τs,α)

    filt = t̂s .>= (t̂s[end] - Δt̂)
    Tsurfmin = minimum(Tsurfs[filt])
    Tsurfmax = maximum(Tsurfs[filt])
    Tcloudups = calcTcloudup.(ΔTcloud,Tsurfs)
    Tclouddowns = calcTclouddown.(ΔTcloud,Tsurfs)
    T4p5s = calcTbrightλ_LW.(4.5e-6,Tsurfs,τs,Tcloudups,β)
    T0p5s = calcTbrightλ_SW.(0.5e-6,Tsurfs,τs,α,Rstar,aplanet,Tstar,ratA)
    T0p5refs = calcTbrightλ_refonly(0.5e-6,τs,α,Rstar,aplanet,Tstar,ratA)

    # determine solution end state (and period if limit cycle) 
    endstateflag,P,Tsurfmin,Tsurfmax,T4p5min,T4p5max,TbSWmin,TbSWmax,τmin,τmax,Lmin,Lmax = checkendsol_radbal(sol,ΔTthres,Tsurfeq,Δt̂,β,ΔTcloud,α,Rstar,aplanet,Tstar,ratA)

    endstate = if endstateflag==2 
        "limit cycle"
    elseif endstateflag==1
        "fixed point"
    elseif endstateflag==7
        "irregular oscillations / chaotic regime (tentative)"
    else
         "problem!"
    end

    # make nc file
    mkpath(outdir)
	fnc = outdir*runname*".nc"

    # set up attributes 
    t̂atts = Dict("longname" => "time",
	"units"=>"delay time")
    Tatts = Dict("longname" => "surface temperature",
	"units"=>"K")
    T4p5atts = Dict("longname" => "4.5 um brightness temperature, only planetary emission",
	"units"=>"K")
    T0p5atts = Dict("longname" => "0.5 um brightness temperature, planetary emission and stellar reflection",
	"units"=>"K")
    T0p5refatts = Dict("longname" => "0.5 um brightness temperature, only stellar reflection",
	"units"=>"K")
    Tcloudupatts = Dict("longname" => "cloud upward emission temperature",
	"units"=>"K")
    Tclouddownatts = Dict("longname" => "cloud downward emission temperature",
	"units"=>"K")
    Aatts = Dict("longname" => "albedo",
	"units"=>"1")
    τatts = Dict("longname" => "shortwave optical depth",
	"units"=>"1")
    Latts = Dict("longname" => "column integrated latent heat toward magma ocean solidification",
	"units"=>"J/m2")


    # add parameters 
    Π₁atts = Dict("longname" => "bulk parameter 1",
	"units"=>"1/Pa")
    Π₂atts = Dict("longname" => "bulk parameter 2",
	"units"=>"1")
    Π₃atts = Dict("longname" => "bulk parameter 3",
	"units"=>"kg/s^3/K")
    Trefatts = Dict("longname" => "Clausius Clapeyron reference temperature",
	"units"=>"K")
    prefatts = Dict("longname" => "Clausius Clapeyron reference pressure",
	"units"=>"Pa")
    αatts = Dict("longname" => "multiple scattering albedo factor",
	"units"=>"1")
    S₀atts = Dict("longname" => "incident stellar radiation",
	"units"=>"W/m2")
    fatts = Dict("longname" => "heat redistribution factor",
	"units"=>"1")
    ΔTcloudatts = Dict("longname" => "downward cloud emission temperature minus upward cloud emission temperature",
	"units"=>"K")
    βatts = Dict("longname" => "ratio of longwave optical depth to shortwave optical depth (i.e., tau_lw = beta*tau_sw)",
	"units"=>"1")

    T55cncAatts = Dict("longname" => "55 Cnc A emission temperature",
	"units"=>"K")
    a55cnceatts = Dict("longname" => "55 Cnc e orbital semi-major axis",
	"units"=>"m")
    R55cncAatts = Dict("longname" => "55 Cnc A radius",
	"units"=>"m")
    ratAatts = Dict("longname" => "ratio of geometric albedo to substellar point albedo",
	"units"=>"1")

    Tsurf0atts = Dict("longname" => "fixed point surface temperature",
	"units"=>"K")
    T4p50atts = Dict("longname" => "fixed point 4.5 um brightness temperature",
	"units"=>"K")
    τsw0atts = Dict("longname" => "fixed point shortwave cloud optical depth",
	"units"=>"1")

    # only used if limit cycle 
    Tsurfminatts = Dict("longname" => "minimum surface temperature",
	"units"=>"K")
    Tsurfmaxatts = Dict("longname" => "maximum surface temperature",
	"units"=>"K")
    T4p5minatts = Dict("longname" => "minimum 4.5 um brightness temperature",
	"units"=>"K")
    T4p5maxatts = Dict("longname" => "maximum 4.5 um brightness temperature",
	"units"=>"K")
    τswminatts = Dict("longname" => "minimum shortwave cloud optical depth",
	"units"=>"1")
    τswmaxatts = Dict("longname" => "maximum shortwave cloud optical depth",
	"units"=>"1")
    Patts = Dict("longname" => "limit cycle period",
	"units"=>"delay time")

    nccreate(fnc,"Pi1","parameter",atts=Π₁atts)
	ncwrite([Π₁],fnc,"Pi1")

    nccreate(fnc,"Pi2","parameter",atts=Π₂atts)
	ncwrite([Π₂],fnc,"Pi2")

    nccreate(fnc,"Pi3","parameter",atts=Π₃atts)
	ncwrite([Π₃],fnc,"Pi3")

    nccreate(fnc,"T_ref","parameter",atts=Trefatts)
	ncwrite([T_ref],fnc,"T_ref")

    nccreate(fnc,"p_ref","parameter",atts=prefatts)
	ncwrite([p_ref],fnc,"p_ref")

    nccreate(fnc,"alpha","parameter",atts=αatts)
	ncwrite([α],fnc,"alpha")

    nccreate(fnc,"S0","parameter",atts=S₀atts)
	ncwrite([S₀],fnc,"S0")

    nccreate(fnc,"f","parameter",atts=fatts)
	ncwrite([f],fnc,"f")

    nccreate(fnc,"Delta_T_cloud","parameter",atts=ΔTcloudatts)
	ncwrite([ΔTcloud],fnc,"Delta_T_cloud")

    nccreate(fnc,"beta","parameter",atts=βatts)
	ncwrite([β],fnc,"beta")

    nccreate(fnc,"T55cncA","parameter",atts=T55cncAatts)
	ncwrite([T55cncA],fnc,"T55cncA")

    nccreate(fnc,"a55cnce","parameter",atts=a55cnceatts)
	ncwrite([a55cnce],fnc,"a55cnce")

    nccreate(fnc,"R55cncA","parameter",atts=R55cncAatts)
	ncwrite([R55cncA],fnc,"R55cncA")

    nccreate(fnc,"ratA","parameter",atts=ratAatts)
	ncwrite([ratA],fnc,"ratA")


    # add properties of solution 
    nccreate(fnc,"T_surf_0","property",atts=Tsurf0atts)
	ncwrite([Tsurfeq],fnc,"T_surf_0")

    Tcloudup_eq = calcTcloudup(ΔTcloud,Tsurfeq)
    T4p5_eq = calcTbrightλ_LW.(4.5e-6,Tsurfeq,τeq,Tcloudup_eq,β)

    nccreate(fnc,"T_4p5_0","property",atts=T4p50atts)
	ncwrite([T4p5_eq],fnc,"T_4p5_0")
   
    nccreate(fnc,"tau_sw_0","property",atts=τsw0atts)
	ncwrite([τeq],fnc,"tau_sw_0")

    nccreate(fnc,"P","property",atts=Patts)
	ncwrite([P],fnc,"P")

    nccreate(fnc,"T_4p5_min","property",atts=T4p5minatts)
	ncwrite([T4p5min],fnc,"T_4p5_min")

    nccreate(fnc,"T_4p5_max","property",atts=T4p5maxatts)
	ncwrite([T4p5max],fnc,"T_4p5_max")

    nccreate(fnc,"T_surf_min","property",atts=Tsurfminatts)
	ncwrite([Tsurfmin],fnc,"T_surf_min")

    nccreate(fnc,"T_surf_max","property",atts=Tsurfmaxatts)
	ncwrite([Tsurfmax],fnc,"T_surf_max")

    nccreate(fnc,"tau_sw_min","property",atts=τswminatts)
	ncwrite([τmin],fnc,"tau_sw_min")
    
    nccreate(fnc,"tau_sw_max","property",atts=τswmaxatts)
	ncwrite([τmax],fnc,"tau_sw_max")
	
	# add surface Ts
	nccreate(fnc,"T_surf","t-hat",t̂s,t̂atts,atts=Tatts)
	ncwrite(Tsurfs,fnc,"T_surf")

    # add 4.5 um brightness Ts
    nccreate(fnc,"T_4p5","t-hat",t̂s,t̂atts,atts=T4p5atts)
	ncwrite(T4p5s,fnc,"T_4p5")

    # add 0.5 um brightness Ts
    nccreate(fnc,"T_0p5","t-hat",t̂s,t̂atts,atts=T0p5atts)
	ncwrite(T0p5s,fnc,"T_0p5")

    # add 0.5 um brightness Ts only reflection 
    nccreate(fnc,"T_0p5_refonly","t-hat",t̂s,t̂atts,atts=T0p5refatts)
	ncwrite(T0p5refs,fnc,"T_0p5_refonly")

    # add Tcloud up 
    nccreate(fnc,"T_cloud_up","t-hat",t̂s,t̂atts,atts=Tcloudupatts)
	ncwrite(Tcloudups,fnc,"T_cloud_up")

    # add Tcloud down 
    nccreate(fnc,"T_cloud_down","t-hat",t̂s,t̂atts,atts=Tclouddownatts)
	ncwrite(Tclouddowns,fnc,"T_cloud_down")

    # add L 
    nccreate(fnc,"L","t-hat",t̂s,t̂atts,atts=Latts)
	ncwrite(Ls,fnc,"L")

	# add τs 
	nccreate(fnc,"tau_sw","t-hat",t̂s,t̂atts,atts=τatts)
	ncwrite(τs,fnc,"tau_sw")

	# add As
	nccreate(fnc,"A","t-hat",t̂s,t̂atts,atts=Aatts)
	ncwrite(As,fnc,"A")

    # add solver info 
    ncputatt(fnc,"global",Dict("endstate"=>endstate,"reltol"=>reltol,"abstol"=>abstol,"model_type"=>"rad_bal"))

    if isverbose # output extra outputs for understanding model behavior 
        Tdelays = sol.(max.(t̂s .- 1,0.),idxs=1)
        nt = length(t̂s)
        τLWs = zeros(nt)
        FcloudLW_downs = zeros(nt)
        FcloudLW_ups = zeros(nt)
        FsurfLWemits = zeros(nt)
        FsurfSWs = zeros(nt)
        FsurfLWTOAs = zeros(nt)
        FLWTOAs = zeros(nt)
        dτdt̂_sources = zeros(nt)
        dτdt̂_sinks = zeros(nt)

        for i ∈ 1:nt
            τLWs[i],FcloudLW_downs[i],FcloudLW_ups[i],FsurfLWemits[i],FsurfSWs[i],FsurfLWTOAs[i],FLWTOAs[i],dτdt̂_sources[i],dτdt̂_sinks[i] = calcdTτLdt̂_radbal_sourcesink(Tsurfs[i],τs[i],Tdelays[i],Π₁,Π₂,p_ref,T_ref,α,S₀,f,β,ΔTcloud)
        end

        τLWatts = Dict("longname" => "longwave optical depth",
	    "units"=>"1")

        FsurfSWatts = Dict("longname" => "stellar radiative energy flux absorbed by the surface",
        "units"=>"W/m2")

        FsurfLWemitatts = Dict("longname" => "emitted radiative energy flux from the surface",
        "units"=>"W/m2")

        FcloudLWemitdownatts = Dict("longname" => "radiative energy flux emitted by the cloud that is absorbed by the surface",
        "units"=>"W/m2")

        FcloudLWemitupatts = Dict("longname" => "contribution of cloud emission to the outgoing radiative flux above the cloud",
        "units"=>"W/m2")

        FsurfLWemitTOAatts = Dict("longname" => "contribution of surface emission to the outgoing radiative flux above the cloud",
        "units"=>"W/m2")

        FplanetLWemitatts = Dict("longname" => "total outgoing radiative flux above the cloud",
        "units"=>"W/m2")


        dτdt̂_source_atts = Dict("longname" => "optical depth time derivative source term",
	    "units"=>"1 / delay time")
        dτdt̂_sink_atts = Dict("longname" => "optical depth time derivative sink term",
	    "units"=>"1 / delay time")

        nccreate(fnc,"tau_lw","t-hat",t̂s,t̂atts,atts=τLWatts)
	    ncwrite(τLWs,fnc,"tau_lw")

        nccreate(fnc,"Fsurf_sw","t-hat",t̂s,t̂atts,atts=FsurfSWatts)
	    ncwrite(FsurfSWs,fnc,"Fsurf_sw")

        nccreate(fnc,"Fsurf_lw_emit","t-hat",t̂s,t̂atts,atts=FsurfLWemitatts)
	    ncwrite(FsurfLWemits,fnc,"Fsurf_lw_emit")

        nccreate(fnc,"Fcloud_lw_emit_down","t-hat",t̂s,t̂atts,atts=FcloudLWemitdownatts)
	    ncwrite(FcloudLW_downs,fnc,"Fcloud_lw_emit_down")

        nccreate(fnc,"Fcloud_lw_emit_up","t-hat",t̂s,t̂atts,atts=FcloudLWemitupatts)
	    ncwrite(FcloudLW_ups,fnc,"Fcloud_lw_emit_up")

        nccreate(fnc,"Fsurf_lw_emit_TOA","t-hat",t̂s,t̂atts,atts=FsurfLWemitTOAatts)
	    ncwrite(FsurfLWTOAs,fnc,"Fsurf_lw_emit_TOA")

        nccreate(fnc,"Fplanet_lw_TOA","t-hat",t̂s,t̂atts,atts=FplanetLWemitatts)
	    ncwrite(FLWTOAs,fnc,"Fplanet_lw_TOA")

        nccreate(fnc,"dtaudt-hat_source","t-hat",t̂s,t̂atts,atts=dτdt̂_source_atts)
	    ncwrite(dτdt̂_sources,fnc,"dtaudt-hat_source")

        nccreate(fnc,"dtaudt-hat_sink","t-hat",t̂s,t̂atts,atts=dτdt̂_sink_atts)
	    ncwrite(dτdt̂_sinks,fnc,"dtaudt-hat_sink")
    end

    nothing
end

function writenc_const(outdir,runname,sol,Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,Tcloud,Tsurfeq,τeq,reltol,abstol,Δt̂,ΔTthres;
    fsteps::Int=10,isverbose=false,Tstar=T55cncA,aplanet=a55cnce,Rstar=R55cncA,ratA=1.)
    """
    save extensive individual run details for constant model version 
        + model parameters
        + integration parameters
        + solution properties
        + time evolving solution and diagnostics 
    to netcdf 

    this can generate a relatively large file
    do not run for large parameter sweep!
    """

    # convert solution into what outtputed 
    tsteps = unique(sol.t)
    nsteps = length(tsteps)
    interp_linear = linear_interpolation(1:nsteps, tsteps)
    t̂s = interp_linear(LinRange(1,nsteps,nsteps*fsteps))
    Tsurfs = sol.(t̂s,idxs=1)
    
    τs = sol.(t̂s,idxs=2)
    τs = max.(0.,τs)
    Ls = sol.(t̂s,idxs=3)
    As = calcA.(τs,α)

    filt = t̂s .>= (t̂s[end] - Δt̂)
    Tsurfmin = minimum(Tsurfs[filt])
    Tsurfmax = maximum(Tsurfs[filt])
    Tcloudups = Tcloud
    Tclouddowns = Tcloud
    T4p5s = calcTbrightλ_LW.(4.5e-6,Tsurfs,τs,Tcloudups,β)
    T0p5s = calcTbrightλ_SW.(0.5e-6,Tsurfs,τs,α,Rstar,aplanet,Tstar,ratA)
    T0p5refs = calcTbrightλ_refonly(0.5e-6,τs,α,Rstar,aplanet,Tstar,ratA)

    # determine solution end state (and period if limit cycle) 
    endstateflag,P,Tsurfmin,Tsurfmax,T4p5min,T4p5max,TbSWmin,TbSWmax,τmin,τmax,Lmin,Lmax = checkendsol_const(sol,ΔTthres,Tsurfeq,Δt̂,β,Tcloud,α,Rstar,aplanet,Tstar,ratA)

    endstate = if endstateflag==2 
        "limit cycle"
    elseif endstateflag==1
        "fixed point"
    elseif endstateflag==7
        "irregular oscillations / chaotic regime (tentative)"
    else
         "problem!"
    end

    # make nc file
    mkpath(outdir)
	fnc = outdir*runname*".nc"

    # set up attributes 
    t̂atts = Dict("longname" => "time",
	"units"=>"delay time")
    Tatts = Dict("longname" => "surface temperature",
	"units"=>"K")
    T4p5atts = Dict("longname" => "4.5 um brightness temperature, only planetary emission",
	"units"=>"K")
    T0p5atts = Dict("longname" => "0.5 um brightness temperature, planetary emission and stellar reflection",
	"units"=>"K")
    T0p5refatts = Dict("longname" => "0.5 um brightness temperature, only stellar reflection",
	"units"=>"K")
    Tcloudupatts = Dict("longname" => "cloud upward emission temperature",
	"units"=>"K")
    Tclouddownatts = Dict("longname" => "cloud downward emission temperature",
	"units"=>"K")
    Aatts = Dict("longname" => "albedo",
	"units"=>"1")
    τatts = Dict("longname" => "shortwave optical depth",
	"units"=>"1")
    Latts = Dict("longname" => "column integrated latent heat toward magma ocean solidification",
	"units"=>"J/m2")


    # add parameters 
    Π₁atts = Dict("longname" => "bulk parameter 1",
	"units"=>"1/Pa")
    Π₂atts = Dict("longname" => "bulk parameter 2",
	"units"=>"1")
    Π₃atts = Dict("longname" => "bulk parameter 3",
	"units"=>"kg/s^3/K")
    Trefatts = Dict("longname" => "Clausius Clapeyron reference temperature",
	"units"=>"K")
    prefatts = Dict("longname" => "Clausius Clapeyron reference pressure",
	"units"=>"Pa")
    αatts = Dict("longname" => "multiple scattering albedo factor",
	"units"=>"1")
    S₀atts = Dict("longname" => "incident stellar radiation",
	"units"=>"W/m2")
    fatts = Dict("longname" => "heat redistribution factor",
	"units"=>"1")
    Tcloudatts = Dict("longname" => "constant upward and downward cloud emission temperature",
	"units"=>"K")
    βatts = Dict("longname" => "ratio of longwave optical depth to shortwave optical depth (i.e., tau_lw = beta*tau_sw)",
	"units"=>"1")

    T55cncAatts = Dict("longname" => "55 Cnc A emission temperature",
	"units"=>"K")
    a55cnceatts = Dict("longname" => "55 Cnc e orbital semi-major axis",
	"units"=>"m")
    R55cncAatts = Dict("longname" => "55 Cnc A radius",
	"units"=>"m")
    ratAatts = Dict("longname" => "ratio of geometric albedo to substellar point albedo",
	"units"=>"1")

    Tsurf0atts = Dict("longname" => "fixed point surface temperature",
	"units"=>"K")
    T4p50atts = Dict("longname" => "fixed point 4.5 um brightness temperature",
	"units"=>"K")
    τsw0atts = Dict("longname" => "fixed point shortwave cloud optical depth",
	"units"=>"1")

    # only used if limit cycle 
    Tsurfminatts = Dict("longname" => "minimum surface temperature",
	"units"=>"K")
    Tsurfmaxatts = Dict("longname" => "maximum surface temperature",
	"units"=>"K")
    T4p5minatts = Dict("longname" => "minimum 4.5 um brightness temperature",
	"units"=>"K")
    T4p5maxatts = Dict("longname" => "maximum 4.5 um brightness temperature",
	"units"=>"K")
    τswminatts = Dict("longname" => "minimum shortwave cloud optical depth",
	"units"=>"1")
    τswmaxatts = Dict("longname" => "maximum shortwave cloud optical depth",
	"units"=>"1")
    Patts = Dict("longname" => "limit cycle period",
	"units"=>"delay time")

    nccreate(fnc,"Pi1","parameter",atts=Π₁atts)
	ncwrite([Π₁],fnc,"Pi1")

    nccreate(fnc,"Pi2","parameter",atts=Π₂atts)
	ncwrite([Π₂],fnc,"Pi2")

    nccreate(fnc,"Pi3","parameter",atts=Π₃atts)
	ncwrite([Π₃],fnc,"Pi3")

    nccreate(fnc,"T_ref","parameter",atts=Trefatts)
	ncwrite([T_ref],fnc,"T_ref")

    nccreate(fnc,"p_ref","parameter",atts=prefatts)
	ncwrite([p_ref],fnc,"p_ref")

    nccreate(fnc,"alpha","parameter",atts=αatts)
	ncwrite([α],fnc,"alpha")

    nccreate(fnc,"S0","parameter",atts=S₀atts)
	ncwrite([S₀],fnc,"S0")

    nccreate(fnc,"f","parameter",atts=fatts)
	ncwrite([f],fnc,"f")

    nccreate(fnc,"T_cloud","parameter",atts=Tcloudatts)
	ncwrite([Tcloud],fnc,"T_cloud")

    nccreate(fnc,"beta","parameter",atts=βatts)
	ncwrite([β],fnc,"beta")

    nccreate(fnc,"T55cncA","parameter",atts=T55cncAatts)
	ncwrite([T55cncA],fnc,"T55cncA")

    nccreate(fnc,"a55cnce","parameter",atts=a55cnceatts)
	ncwrite([a55cnce],fnc,"a55cnce")

    nccreate(fnc,"R55cncA","parameter",atts=R55cncAatts)
	ncwrite([R55cncA],fnc,"R55cncA")

    nccreate(fnc,"ratA","parameter",atts=ratAatts)
	ncwrite([ratA],fnc,"ratA")


    # add properties of solution 
    nccreate(fnc,"T_surf_0","property",atts=Tsurf0atts)
	ncwrite([Tsurfeq],fnc,"T_surf_0")

    T4p5_eq = calcTbrightλ_LW.(4.5e-6,Tsurfeq,τeq,Tcloud,β)

    nccreate(fnc,"T_4p5_0","property",atts=T4p50atts)
	ncwrite([T4p5_eq],fnc,"T_4p5_0")
   
    nccreate(fnc,"tau_sw_0","property",atts=τsw0atts)
	ncwrite([τeq],fnc,"tau_sw_0")

    nccreate(fnc,"P","property",atts=Patts)
	ncwrite([P],fnc,"P")

    nccreate(fnc,"T_4p5_min","property",atts=T4p5minatts)
	ncwrite([T4p5min],fnc,"T_4p5_min")

    nccreate(fnc,"T_4p5_max","property",atts=T4p5maxatts)
	ncwrite([T4p5max],fnc,"T_4p5_max")

    nccreate(fnc,"T_surf_min","property",atts=Tsurfminatts)
	ncwrite([Tsurfmin],fnc,"T_surf_min")

    nccreate(fnc,"T_surf_max","property",atts=Tsurfmaxatts)
	ncwrite([Tsurfmax],fnc,"T_surf_max")

    nccreate(fnc,"tau_sw_min","property",atts=τswminatts)
	ncwrite([τmin],fnc,"tau_sw_min")
    
    nccreate(fnc,"tau_sw_max","property",atts=τswmaxatts)
	ncwrite([τmax],fnc,"tau_sw_max")
	
	# add surface Ts
	nccreate(fnc,"T_surf","t-hat",t̂s,t̂atts,atts=Tatts)
	ncwrite(Tsurfs,fnc,"T_surf")

    # add 4.5 um brightness Ts
    nccreate(fnc,"T_4p5","t-hat",t̂s,t̂atts,atts=T4p5atts)
	ncwrite(T4p5s,fnc,"T_4p5")

    # add 0.5 um brightness Ts
    nccreate(fnc,"T_0p5","t-hat",t̂s,t̂atts,atts=T0p5atts)
	ncwrite(T0p5s,fnc,"T_0p5")

    # add 0.5 um brightness Ts only reflection 
    nccreate(fnc,"T_0p5_refonly","t-hat",t̂s,t̂atts,atts=T0p5refatts)
	ncwrite(T0p5refs,fnc,"T_0p5_refonly")

    # add L 
    nccreate(fnc,"L","t-hat",t̂s,t̂atts,atts=Latts)
	ncwrite(Ls,fnc,"L")

	# add τs 
	nccreate(fnc,"tau_sw","t-hat",t̂s,t̂atts,atts=τatts)
	ncwrite(τs,fnc,"tau_sw")

	# add As
	nccreate(fnc,"A","t-hat",t̂s,t̂atts,atts=Aatts)
	ncwrite(As,fnc,"A")

    # add solver info 
    ncputatt(fnc,"global",Dict("endstate"=>endstate,"reltol"=>reltol,"abstol"=>abstol,"model_type"=>"const"))

    if isverbose # output extra outputs for understanding model behavior 
        Tdelays = sol.(max.(t̂s .- 1,0.),idxs=1)
        nt = length(t̂s)
        τLWs = zeros(nt)
        FcloudLW_downs = zeros(nt)
        FcloudLW_ups = zeros(nt)
        FsurfLWemits = zeros(nt)
        FsurfSWs = zeros(nt)
        FsurfLWTOAs = zeros(nt)
        FLWTOAs = zeros(nt)
        dτdt̂_sources = zeros(nt)
        dτdt̂_sinks = zeros(nt)

        for i ∈ 1:nt
            τLWs[i],FcloudLW_downs[i],FcloudLW_ups[i],FsurfLWemits[i],FsurfSWs[i],FsurfLWTOAs[i],FLWTOAs[i],dτdt̂_sources[i],dτdt̂_sinks[i] = calcdTτLdt̂_const_sourcesink(Tsurfs[i],τs[i],Tdelays[i],Π₁,Π₂,p_ref,T_ref,α,S₀,f,β,Tcloud)
        end

        τLWatts = Dict("longname" => "longwave optical depth",
	    "units"=>"1")

        FsurfSWatts = Dict("longname" => "stellar radiative energy flux absorbed by the surface",
        "units"=>"W/m2")

        FsurfLWemitatts = Dict("longname" => "emitted radiative energy flux from the surface",
        "units"=>"W/m2")

        FcloudLWemitdownatts = Dict("longname" => "radiative energy flux emitted by the cloud that is absorbed by the surface",
        "units"=>"W/m2")

        FcloudLWemitupatts = Dict("longname" => "contribution of cloud emission to the outgoing radiative flux above the cloud",
        "units"=>"W/m2")

        FsurfLWemitTOAatts = Dict("longname" => "contribution of surface emission to the outgoing radiative flux above the cloud",
        "units"=>"W/m2")

        FplanetLWemitatts = Dict("longname" => "total outgoing radiative flux above the cloud",
        "units"=>"W/m2")


        dτdt̂_source_atts = Dict("longname" => "optical depth time derivative source term",
	    "units"=>"1 / delay time")
        dτdt̂_sink_atts = Dict("longname" => "optical depth time derivative sink term",
	    "units"=>"1 / delay time")

        nccreate(fnc,"tau_lw","t-hat",t̂s,t̂atts,atts=τLWatts)
	    ncwrite(τLWs,fnc,"tau_lw")

        nccreate(fnc,"Fsurf_sw","t-hat",t̂s,t̂atts,atts=FsurfSWatts)
	    ncwrite(FsurfSWs,fnc,"Fsurf_sw")

        nccreate(fnc,"Fsurf_lw_emit","t-hat",t̂s,t̂atts,atts=FsurfLWemitatts)
	    ncwrite(FsurfLWemits,fnc,"Fsurf_lw_emit")

        nccreate(fnc,"Fcloud_lw_emit_down","t-hat",t̂s,t̂atts,atts=FcloudLWemitdownatts)
	    ncwrite(FcloudLW_downs,fnc,"Fcloud_lw_emit_down")

        nccreate(fnc,"Fcloud_lw_emit_up","t-hat",t̂s,t̂atts,atts=FcloudLWemitupatts)
	    ncwrite(FcloudLW_ups,fnc,"Fcloud_lw_emit_up")

        nccreate(fnc,"Fsurf_lw_emit_TOA","t-hat",t̂s,t̂atts,atts=FsurfLWemitTOAatts)
	    ncwrite(FsurfLWTOAs,fnc,"Fsurf_lw_emit_TOA")

        nccreate(fnc,"Fplanet_lw_TOA","t-hat",t̂s,t̂atts,atts=FplanetLWemitatts)
	    ncwrite(FLWTOAs,fnc,"Fplanet_lw_TOA")

        nccreate(fnc,"dtaudt-hat_source","t-hat",t̂s,t̂atts,atts=dτdt̂_source_atts)
	    ncwrite(dτdt̂_sources,fnc,"dtaudt-hat_source")

        nccreate(fnc,"dtaudt-hat_sink","t-hat",t̂s,t̂atts,atts=dτdt̂_sink_atts)
	    ncwrite(dτdt̂_sinks,fnc,"dtaudt-hat_sink")
    end

    nothing
end


function writenc_sweep(solprop_pmap,ps,outdir,sweepname,modeltype;notes="")
    """
    write parameter sweep outputs to netcdf file 
    saved as outdir*sweepname*".nc"
    """

    # get number of samples 
    nsamp = size(ps)[2]
    # set number of properties outtputed for each run 
    nprop = 12

    # convert what pmap returns into something useable 
    solprop = zeros(nprop,nsamp)
    for i ∈ 1:nsamp 
        solprop[:,i] .= solprop_pmap[i]
    end 

    # make output directory if needed  
    mkpath(outdir)

    # set output file path 
    fnc = outdir*sweepname*".nc"

    # check model type 
    if modeltype ∉ [:rad_bal,:const]
        error("invalid modeltype $(modeltype)! only :const or :rad_bal accepted!")
    end 

    # set up attributes 

    # for dimension 
    runatts = Dict("longname"=>"model run in parameter sweep")

    # for model parameters 
    Π₁atts = Dict("longname" => "bulk parameter 1",
    "units"=>"1/Pa")
    Π₂atts = Dict("longname" => "bulk parameter 2",
    "units"=>"1")
    Π₃atts = Dict("longname" => "bulk parameter 3",
    "units"=>"kg/s^3/K")
    Trefatts = Dict("longname" => "Clausius Clapeyron reference temperature",
    "units"=>"K")
    prefatts = Dict("longname" => "Clausius Clapeyron reference pressure",
    "units"=>"Pa")
    αatts = Dict("longname" => "multiple scattering albedo factor",
    "units"=>"1")
    fatts = Dict("longname" => "heat redistribution factor",
    "units"=>"1")
    S₀atts = Dict("longname" => "incident stellar radiation",
    "units"=>"W/m2")
    βatts = Dict("longname" => "ratio of longwave optical depth to shortwave optical depth (i.e., tau_lw = beta*tau_sw)",
    "units"=>"1")
    ΔTcloudatts = Dict("longname" => "downward cloud emission temperature minus upward cloud emission temperature",
    "units"=>"K")
    Tcloudatts = Dict("longname" => "constant (downward and upward) cloud emission temperature",
    "units"=>"K")
    Tsoldiusatts = Dict("longname" => "magma solidus temperature",
    "units"=>"K")
    

    # for solution properties 
    Tsurfminatts = Dict("longname" => "minimum surface temperature",
    "units"=>"K")
    Tsurfmaxatts = Dict("longname" => "maximum surface temperature",
    "units"=>"K")
    T4p5minatts = Dict("longname" => "minimum 4.5 um (longwave) brightness temperature",
    "units"=>"K")
    T4p5maxatts = Dict("longname" => "maximum 4.5 um (longwave) brightness temperature",
    "units"=>"K")
    T0p5minatts = Dict("longname" => "minimum 0.5 um (shortwave) brightness temperature",
    "units"=>"K")
    T0p5maxatts = Dict("longname" => "maximum 0.5 um (shortwave) brightness temperature",
    "units"=>"K")
    τswminatts = Dict("longname" => "minimum shortwave cloud optical depth",
    "units"=>"1")
    τswmaxatts = Dict("longname" => "maximum shortwave cloud optical depth",
    "units"=>"1")
    Patts = Dict("longname" => "limit cycle period",
    "units"=>"delay time", "note" => "NaN for non-limit cycle runs")
    Lminatts = Dict("longname" => "minimum column integrated latent heat toward magma ocean solidification",
    "units"=>"J/m2","note" => "should be 0 J/m2")
    Lmaxatts = Dict("longname" => "maximum column integrated latent heat toward magma ocean solidification",
    "units"=>"J/m2")

    endstateflagatts = Dict("longname" => "flag to describe end state of model run",
    "flag_values" => "-2: equilibrium surface temperature below magma solidus, "*
    "-1: model run errored, 0: end state test inconclusive (early termination), "* 
    "1: steady state (early termination), 2: limit cycle (early termination), 3: erratic period (early termination), "*
    "4: end state test inconclusive (ran to end time), 5: steady state (ran to end time), "*
    "6: limit cycle (ran to end time), 7: erratic period (ran to end time)", 
    "note" => "expected end flags are -2, 1, 2, and 7. other end flags suggest model is behaving unexpectedly. "*
    "double check the parameter inputs. try lowering the tolerances or extending the end time. "*
    "flag 7 indicates erratic periods and is tentatively associated with chaotic regime.")

    # set up run dimension netcdf file 
    iruns = 1:nsamp

    # add parameters 
    nccreate(fnc,"Pi1","run",iruns,runatts,atts=Π₁atts)
    ncwrite(ps[1,:],fnc,"Pi1")

    nccreate(fnc,"Pi2","run",atts=Π₂atts)
    ncwrite(ps[2,:],fnc,"Pi2")

    nccreate(fnc,"Pi3","run",atts=Π₃atts)
    ncwrite(ps[3,:],fnc,"Pi3")

    nccreate(fnc,"T_ref","run",atts=Trefatts)
    ncwrite(ps[4,:],fnc,"T_ref")

    nccreate(fnc,"p_ref","run",atts=prefatts)
    ncwrite(ps[5,:],fnc,"p_ref")

    nccreate(fnc,"alpha","run",atts=αatts)
    ncwrite(ps[6,:],fnc,"alpha")

    nccreate(fnc,"S0","run",atts=S₀atts)
    ncwrite(ps[7,:],fnc,"S0")

    nccreate(fnc,"f","run",atts=fatts)
    ncwrite(ps[8,:],fnc,"f")

    nccreate(fnc,"beta","run",atts=βatts)
    ncwrite(ps[9,:],fnc,"beta")

    if modeltype==:const
        nccreate(fnc,"T_cloud","run",atts=Tcloudatts)
        ncwrite(ps[10,:],fnc,"T_cloud")
    elseif modeltype==:rad_bal
        nccreate(fnc,"Delta_T_cloud","run",atts=ΔTcloudatts)
        ncwrite(ps[10,:],fnc,"Delta_T_cloud")
    end 

    nccreate(fnc,"T_solidus","run",atts=Tsoldiusatts)
    ncwrite(ps[11,:],fnc,"T_solidus")


    # add outputs 
    nccreate(fnc,"end_state_flag","run",atts=endstateflagatts)
    ncwrite(solprop[1,:],fnc,"end_state_flag")

    nccreate(fnc,"P","run",atts=Patts)
    ncwrite(solprop[2,:],fnc,"P")

    nccreate(fnc,"T_surf_min","run",atts=Tsurfminatts)
    ncwrite(solprop[3,:],fnc,"T_surf_min")

    nccreate(fnc,"T_surf_max","run",atts=Tsurfmaxatts)
    ncwrite(solprop[4,:],fnc,"T_surf_max")

    nccreate(fnc,"T_4p5_min","run",atts=T4p5minatts)
    ncwrite(solprop[5,:],fnc,"T_4p5_min")

    nccreate(fnc,"T_4p5_max","run",atts=T4p5maxatts)
    ncwrite(solprop[6,:],fnc,"T_4p5_max")

    nccreate(fnc,"T_0p5_min","run",atts=T0p5minatts)
    ncwrite(solprop[7,:],fnc,"T_0p5_min")

    nccreate(fnc,"T_0p5_max","run",atts=T0p5maxatts)
    ncwrite(solprop[8,:],fnc,"T_0p5_max")

    nccreate(fnc,"tau_sw_min","run",atts=τswminatts)
    ncwrite(solprop[9,:],fnc,"tau_sw_min")

    nccreate(fnc,"tau_sw_max","run",atts=τswmaxatts)
    ncwrite(solprop[10,:],fnc,"tau_sw_max")

    nccreate(fnc,"L_min","run",atts=Lminatts)
    ncwrite(solprop[11,:],fnc,"L_min")

    nccreate(fnc,"L_max","run",atts=Lmaxatts)
    ncwrite(solprop[12,:],fnc,"L_max")

    # add meta data 
    ncputatt(fnc,"global",Dict("model_type"=>string(modeltype),"notes"=>notes))

    nothing 
end


# FUNCTIONS TO SAVE SOLUTIONS #####################################

###################################################################

# FUNCTIONS TO ANALYZE SOLUTIONS ##################################
function get_modeltype(sweepname,outdir)
    # file name 
    fnc = outdir*sweepname*".nc"
    # get model type 
    ncgetatt(fnc,"Global","model_type")
end

function get_variedparam_obsfilt(sweepname,outdir,Tmin_max_obs,Tmax_min_obs)
    """
    load parameter combinations consistent with observations  
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # load 4.5 μm brightness temperatures to check obs consistency 
    T_04p5_min = ncread(fnc,"T_4p5_min")
    T_04p5_max = ncread(fnc,"T_4p5_max")
    # load end state flags to ensure limit cycles 
    endstateflag = ncread(fnc,"end_state_flag")
    # get model type 
    model_type = ncgetatt(fnc,"Global","model_type")

    # filter min and max 4.5 μm brightness temperatures for consistency with inputted observation limits
    # also make sure results are limit cycles 
    iobs = (T_04p5_min .< Tmin_max_obs) .&& (T_04p5_max .> Tmax_min_obs) .&& (endstateflag .== 2)

    # read in appropriate cloud parameter (ΔTcloud or Tcloud)
    # depending on model type 
    cloud_param = if model_type == "rad_bal"
        ncread(fnc,"Delta_T_cloud")[iobs]
    else model_type == "const"
        ncread(fnc,"T_cloud")[iobs]
    end
    Π1 = ncread(fnc,"Pi1")[iobs]
    Π2 = ncread(fnc,"Pi2")[iobs]
    Π3 = ncread(fnc,"Pi3")[iobs]
    β = ncread(fnc,"beta")[iobs]

    Π1,Π2,Π3,cloud_param,β
end

function get_properties_obsfilt(outdir,sweepname,Tmin_max_obs,Tmax_min_obs)
    """
    get model solution properties consistent with min and max bounds on observed temperature 
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # indices for runs consistent with Tmin_max_obs and Tmax_min_obs
    iobs = find_obsfilt(sweepname,outdir,Tmin_max_obs,Tmax_min_obs)
    # properties for all runs 
    _,P,Tsurf_min,Tsurf_max,T_4p5_min,T_4p5_max,T_0p5_min,T_0p5_max,τSW_min,τSW_max,Lmin,Lmax = get_properties(sweepname,outdir)
    # properties for runs consistent with obs
    P[iobs],Tsurf_min[iobs],Tsurf_max[iobs],T_4p5_min[iobs],T_4p5_max[iobs],T_0p5_min[iobs],T_0p5_max[iobs],τSW_min[iobs],τSW_max[iobs],Lmin[iobs],Lmax[iobs]
end

function find_obsfilt(sweepname,outdir,Tmin_max_obs,Tmax_min_obs)
    """
    filter for model runs consistent with observations  
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # load 4.5 μm brightness temperatures to check obs consistency 
    T_04p5_min = ncread(fnc,"T_4p5_min")
    T_04p5_max = ncread(fnc,"T_4p5_max")
    # load end state flags to ensure limit cycles 
    endstateflag = ncread(fnc,"end_state_flag")

    # filter min and max 4.5 μm brightness temperatures for consistency with inputted observation limits
    # when solutions are limit cycles 
    # true = consistent 
    # false = not consistent 
    (T_04p5_min .< Tmin_max_obs) .&& (T_04p5_max .> Tmax_min_obs) .&& (endstateflag .== 2)
end

function print_variedparam_ranges_obsfilt(sweepname,outdir;fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.)
    """
    print varied parameter ranges that are observationally consistent from parameter sweep 
    """

    model_type = get_modeltype(sweepname,outdir)

    cloud_param_name = if model_type == "rad_bal"
        "ΔTcloud [K]"
    else model_type == "const"
        "Tcloud [K]"
    end

    param_names = ["Π₁ [Pa⁻¹]",
                "Π₂ [ ]",
                "Π₃ [kg s⁻³ K⁻¹]",
                cloud_param_name,
                "β [ ]"]

    for nσ ∈ 1:3
        println("\nPARAMETER RANGES FOR $(nσ) σ CONSISTENCY")
        params = get_variedparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + nσ*σTmin_max_obs),Tmax_min_obs - nσ*σTmax_min_obs)
        for iparam ∈ 1:5
            println("\t$(param_names[iparam]):")
            if length(params[iparam])>0
                println("\t\tmin = $(minimum(params[iparam]))")
                println("\t\tmax = $(maximum(params[iparam]))")
            else
                println("\t\tno consistent runs")
            end
        end
    end
    nothing 
end

function get_properties(sweepname,outdir)
    # file name 
    fnc = outdir*sweepname*".nc"
    # load output properties from parameter sweep 
    endstateflag = ncread(fnc,"end_state_flag")
    P = ncread(fnc,"P")
    Tsurf_min = ncread(fnc,"T_surf_min")
    Tsurf_max = ncread(fnc,"T_surf_max")
    T_4p5_min = ncread(fnc,"T_4p5_min")
    T_4p5_max = ncread(fnc,"T_4p5_max")
    T_0p5_min = ncread(fnc,"T_0p5_min")
    T_0p5_max = ncread(fnc,"T_0p5_max")
    τSW_min = ncread(fnc,"tau_sw_min")
    τSW_max = ncread(fnc,"tau_sw_max")
    Lmin = ncread(fnc,"L_min")
    Lmax = ncread(fnc,"L_max")

    endstateflag,P,Tsurf_min,Tsurf_max,T_4p5_min,T_4p5_max,T_0p5_min,T_0p5_max,τSW_min,τSW_max,Lmin,Lmax
end

function print_property_ranges_obsfilt(sweepname,outdir;fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.)

    # set names for printing 
    prop_names = ["period [delay times]",
                "minimum surface T [K]",
                "maximum surface T [K]",
                "minimum LW brightness T [K]",
                "maximum LW brightness T [K]",
                "minimum SW brightness T [K]",
                "maximum SW brightness T [K]",
                "minimum SW τ [ ]",
                "maximum SW τ [ ]",
                "minimum L [J m⁻²]",
                "maximum L [J m⁻²]"]

    # iterate over number of standard deviations 
    for nσ ∈ 1:3
        println("\nRANGES FOR $(nσ) σ")
        props = get_properties_obsfilt(outdir,sweepname,fday*(Tmin_max_obs + nσ*σTmin_max_obs),Tmax_min_obs - nσ*σTmax_min_obs)
        for iprop ∈ 1:9
            println("\t$(prop_names[iprop]):")
            if length(props[iprop])>0
                println("\t\tmin = $(minimum(props[iprop]))")
                println("\t\tmax = $(maximum(props[iprop]))")
            else
                println("\t\tno consistent runs")
            end
        end
    end
end

function get_parameters(sweepname,outdir)
    """
    load parameters into p array used to run model 
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # get model type 
    model_type = ncgetatt(fnc,"Global","model_type")

    # load parameters from parameter sweep 
    # read in appropriate cloud parameter (ΔTcloud or Tcloud)
    # depending on model type 
    cloud_param = if model_type == "rad_bal"
        ncread(fnc,"Delta_T_cloud")
    else model_type == "const"
        ncread(fnc,"T_cloud")
    end
    Π1 = ncread(fnc,"Pi1")
    Π2 = ncread(fnc,"Pi2")
    Π3 = ncread(fnc,"Pi3")
    T_ref = ncread(fnc,"T_ref")
    p_ref = ncread(fnc,"p_ref")
    α = ncread(fnc,"alpha")
    S₀ = ncread(fnc,"S0")
    f = ncread(fnc,"f")
    β = ncread(fnc,"beta")
    Tsolidus = ncread(fnc,"T_solidus")
    hcat(Π1, Π2, Π3, T_ref, p_ref, α, S₀, f, β, cloud_param, Tsolidus)
end

function get_parameters_varied(sweepname,outdir)
    """
    load varied parameters 
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # get model type 
    model_type = ncgetatt(fnc,"Global","model_type")

    # load parameters from parameter sweep 
    # read in appropriate cloud parameter (ΔTcloud or Tcloud)
    # depending on model type 
    cloud_param = if model_type == "rad_bal"
        ncread(fnc,"Delta_T_cloud")
    else model_type == "const"
        ncread(fnc,"T_cloud")
    end
    Π1 = ncread(fnc,"Pi1")
    Π2 = ncread(fnc,"Pi2")
    Π3 = ncread(fnc,"Pi3")
    β = ncread(fnc,"beta")
    Π1, Π2, Π3,cloud_param,β
end

function check_param_sweep(sweepname,outdir;isplot=true,maxfig=50,maxparamprint=5,figdirbase="sfigs/",
    fday=1.1,Tmin_max_obs=873.,σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,
    nσ_plot=3,alg=KenCarp4(),reltol=1e-8,abstol=1e-10,t̂end=1e4)
    """
    check results of parameter sweep 
        + print end state flags 
        + print tests checking model behavior 
            > bad end state flags 
            > Tsurf / L integration switch misbehaviors when Tsurf = Tsolidus
        + make plots for problematic runs (up to maxfig)
        + print parameter ranges consistent with observations 
        + print solution properties for solutions consistent with observations
    """
    fnc = outdir*sweepname*".nc"
    Tsurf_min = ncread(fnc,"T_surf_min")
    T_solidus = ncread(fnc,"T_solidus")
    Lmin = ncread(fnc,"L_min")
    Lmax = ncread(fnc,"L_max")
    endstateflag = ncread(fnc,"end_state_flag")

    # get model type 
    model_type = ncgetatt(fnc,"Global","model_type")
    

    nsamp = length(T_solidus)

    # check model runs via end state flags 
    uni_flags = unique(endstateflag)
    println("end state flags encountered: $(uni_flags)")

    println("$(sum(endstateflag.==1))/$(nsamp) runs reach steady state")
    println("$(sum(endstateflag.==2))/$(nsamp) runs reach limit cycles")
    println("$(sum(endstateflag.==-1))/$(nsamp) runs fail to integrate")
    println("$(sum(endstateflag.==-2))/$(nsamp) runs have equilibrium Tsurf < Tsolidus")
    println("$(sum(endstateflag.==0))/$(nsamp) runs can't be classified")
    println("$(sum(endstateflag.==3))/$(nsamp) runs have erratic periods but are ending early")
    println("$(sum(endstateflag.==4))/$(nsamp) runs can't be classified and reach t̂end")
    println("$(sum(endstateflag.==5))/$(nsamp) runs reach steady state but at t̂end")
    println("$(sum(endstateflag.==6))/$(nsamp) runs reach limit cycles but at t̂end")
    println("$(sum(endstateflag.==7))/$(nsamp) runs have erratic periods and reach t̂end (possible chaotic regime)\n")

    if isplot
        # make fig directory 
        figdir = figdirbase*sweepname*"/"
        mkpath(figdir)
        ps = get_parameters(sweepname,outdir)
        # plot problems (note 7 is not a problem but want to see cases)
        flag_probs = [-1,0,3,4,5,6,7]
        for flag_prob ∈ flag_probs
            if flag_prob ∈ uni_flags
                p_flag = ps[endstateflag.==flag_prob,:]
                n_flag = size(p_flag)[1]
                println("plotting flag $flag_prob")
                for irun ∈ 1:(min(n_flag,maxfig))
                    runname = "$(round(Int(flag_prob)))_$(irun)"
                    if irun < maxparamprint
                        @show p_flag[irun,:]
                    end
                    if model_type=="rad_bal"
                        calcsolprop_radbal(p_flag[irun,:];isplot=true,figdir=figdir,runname=runname,reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
                        checksol_radbal(p_flag[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
                    elseif model_type=="const"
                        calcsolprop_const(p_flag[irun,:];isplot=true,figdir=figdir,runname=runname,reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
                        checksol_const(p_flag[irun,:],figdir,runname,reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
                    end
                end
                println("\n")
            end
        end
        # plot limit cycles 
        iobs = find_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + nσ_plot*σTmin_max_obs),Tmax_min_obs - nσ_plot*σTmax_min_obs)
        p_lc_obs = ps[iobs,:]
        n_lc_obs = size(p_lc_obs)[1]
        println("plotting limit cycles")
        for irun ∈ 1:(min(n_lc_obs,maxfig))
            if model_type=="rad_bal"
                checksol_radbal(p_lc_obs[irun,:],figdir,"lc_$(irun)";reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            elseif model_type=="const"
                checksol_const(p_lc_obs[irun,:],figdir,"lc_$(irun)";reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            end
        end
        println("\n")
    end
    

    # check T solidus 
    Tsolidus_check = Tsurf_min .< (T_solidus .- 1.)
    if any(Tsolidus_check) 
        println("some model solutions ($(sum(Tsolidus_check))/$(nsamp)) have minimum surface temperatures below magma solidus parameter")
        nprint = min(sum(Tsolidus_check),maxparamprint)
        p_freeze = ps[Tsolidus_check,:]
        for irun ∈ 1:nprint
            @show p_freeze[irun,:]
            runname = "freeze_$(irun)"
            if model_type=="rad_bal"
                checksol_radbal(p_freeze[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            elseif model_type=="const"
                checksol_const(p_freeze[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            end
        end
        println("\n")
    end

    # check Lmin 
    Lmin_check1 = (Lmin .< 0) .&& (abs.(Lmin)./Lmax .> 1e-6)
    if any(Lmin_check1) 
        println("some model solutions ($(sum(Lmin_check1))/$(nsamp)) have issues balancing latent heat")
        nprint = min(sum(Lmin_check1),maxparamprint)
        p_negL = ps[Lmin_check1,:]
        for irun ∈ 1:nprint
            @show p_negL[irun,:]
            runname = "negL1_$(irun)"
            if model_type=="rad_bal"
                checksol_radbal(p_negL[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            elseif model_type=="const"
                checksol_const(p_negL[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            end
        end
        println("\n")
    end
    Lmin_check2 = Lmin .< -1e-4
    if any(Lmin_check2) 
        println("some model solutions ($(sum(Lmin_check2))/$(nsamp)) have negative accumulated latent heat, which shouldn't happen by sign convention")
        nprint = min(sum(Lmin_check2),maxparamprint)
        p_negL = ps[Lmin_check2,:]
        for irun ∈ 1:nprint
            @show p_negL[irun,:]
            runname = "negL2_$(irun)"
            if model_type=="rad_bal"
                checksol_radbal(p_negL[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            elseif model_type=="const"
                checksol_const(p_negL[irun,:],figdir,runname;reltol=reltol,abstol=abstol,t̂end=t̂end,alg=alg)
            end
        end
    end

    # print parameter ranges consistent with obs
    print_variedparam_ranges_obsfilt(sweepname,outdir)

    println("\n")

    # print property ranges consistent with obs 
    print_property_ranges_obsfilt(sweepname,outdir)

    nothing

end

function compare_param_sweep(sweepname1,outdir1,sweepname2,outdir2;
    fday=1.1,Tmin_max_obs=873.,σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,nσ_plot=3)
    """
    compare results of two parameter sweeps 
    """
    fnc1 = outdir1*sweepname1*".nc"
    fnc2 = outdir2*sweepname2*".nc"

    iobs1 = find_obsfilt(sweepname1,outdir1,fday*(Tmin_max_obs + nσ_plot*σTmin_max_obs),Tmax_min_obs - nσ_plot*σTmax_min_obs)
    iobs2 = find_obsfilt(sweepname2,outdir2,fday*(Tmin_max_obs + nσ_plot*σTmin_max_obs),Tmax_min_obs - nσ_plot*σTmax_min_obs)

    endstateflag1,P1,Tsurf_min1,Tsurf_max1,T_4p5_min1,T_4p5_max1,T_0p5_min1,T_0p5_max1,τSW_min1,τSW_max1,Lmin1,Lmax1 = get_properties(sweepname1,outdir1)
    endstateflag2,P2,Tsurf_min2,Tsurf_max2,T_4p5_min2,T_4p5_max2,T_0p5_min2,T_0p5_max2,τSW_min2,τSW_max2,Lmin2,Lmax2 = get_properties(sweepname2,outdir2)


    if length(iobs1) == length(iobs2)
        println("COMPARE $(fnc1) vs $(fnc2)")
        if endstateflag1 == endstateflag2
            println("\nall end flags the same")
        else
            nflagdiff = sum(endstateflag1 .!= endstateflag2)
            nsamp = length(endstateflag1)
            println("\n$(nflagdiff)/$(nsamp) runs have different end flags")
        end

        maxrelerr_Tsurfmin = round(maximum(abs.((Tsurf_min1 .- Tsurf_min2)./Tsurf_min2)[endstateflag1.>-1]),sigdigits=3)
        println("maximum relative difference in min surface T = $(maxrelerr_Tsurfmin)")

        maxrelerr_Tsurfmax = round(maximum(abs.((Tsurf_max1 .- Tsurf_max2)./Tsurf_max2)[endstateflag1.>=-1]),sigdigits=3)
        println("maximum relative difference in max surface T = $(maxrelerr_Tsurfmax)")

        maxrelerr_T4p5_min = round(maximum(abs.((T_4p5_min1 .- T_4p5_min2)./T_4p5_min2)[endstateflag1.>=-1]),sigdigits=3)
        println("maximum relative difference in min 4.5 μm brightness T = $(maxrelerr_T4p5_min)")

        maxrelerr_T4p5_max = round(maximum(abs.((T_4p5_max1 .- T_4p5_max2)./T_4p5_max2)[endstateflag1.>=-1]),sigdigits=3)
        println("maximum relative difference in max 4.5 μm brightness T = $(maxrelerr_T4p5_max)")

        maxrelerr_T0p5_min = round(maximum(abs.((T_0p5_min1 .- T_0p5_min2)./T_0p5_min2)[endstateflag1.>=-1]),sigdigits=3)
        println("maximum relative difference in min 0.5 μm brightness T = $(maxrelerr_T0p5_min)")

        maxrelerr_T405_max = round(maximum(abs.((T_0p5_max1 .- T_0p5_max2)./T_0p5_max2)[endstateflag1.>=-1]),sigdigits=3)
        println("maximum relative difference in max 0.5 μm brightness T = $(maxrelerr_T405_max)")


        if iobs1 == iobs2
            println("\nall observationally consistent runs the same")
        else
            nobsdiff = sum(iobs1 .!= iobs2)
            nsamp = length(iobs1)
            println("\n$(nobsdiff)/$(nsamp) runs have different observational consistency")
        end
    else
        println("can't compare $(fnc1) vs $(fnc2)")
        println("they have different numbers of parameter combinations")
    end


end

# FUNCTIONS TO ANALYZE SOLUTIONS ##################################

###################################################################

# FUNCTIONS TO PLOT SOLUTIONS #####################################

function make_Π_Tcloudβ_3σ_figs_2model_poly(outdir_const,outdir_radbal,figdir,sweepname_const,sweepname_radbal;plotname="",fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,cmap=:lajolla,
    xtickrotation=π/4.,ngrid=25,lw=2,islogβ=false,figsize=(600,500),
    logΠ1min=nothing,logΠ1max=nothing,logΠ2min=nothing,logΠ2max=nothing,
    logΠ3min=nothing,logΠ3max=nothing,βmin=nothing,βmax=nothing,
    ΔTcloudmin=nothing,ΔTcloudmax=nothing,Tcloudmin=nothing,Tcloudmax=nothing,ΔTcloudticks=[0,100,200,300],
    ischeckβTcloud=true,βeff0=1e-2)

    # set labels 
    labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
    labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
    labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")
    labelTcloud = rich(rich("T",font=:italic),subscript("cloud")," [K]")
    labelΔTcloud = rich(rich("ΔT",font=:italic),subscript("cloud")," [K]")
    labelβ = if islogβ
        rich("log ",rich("β",font=:italic)," [–]")
    else
        rich(rich("β",font=:italic)," [–]")
    end

    # set colors 
    cobs = cgrad(cmap,5,categorical=true)[2:4]

    # make figdir if needed 
    mkpath(figdir)

    # set up figure and axes
    fig = Figure(size=figsize) 
    ax_21 = Axis(fig[1,1],xlabel=labelΠ1,ylabel=labelΠ2)
    ax_31 = Axis(fig[2,1],xlabel=labelΠ1,ylabel=labelΠ3)
    ax_32 = Axis(fig[2,2],xlabel=labelΠ2,ylabel=labelΠ3)


    ax_β1 = Axis(fig[3,1],xlabel=labelΠ1,ylabel=labelβ)
    ax_β2 = Axis(fig[3,2],xlabel=labelΠ2,ylabel=labelβ)
    ax_β3 = Axis(fig[3,3],xlabel=labelΠ3,ylabel=labelβ)

    ax_ΔT1 = Axis(fig[4,1],xlabel=labelΠ1,ylabel=labelΔTcloud,yticks=ΔTcloudticks)
    ax_ΔT2 = Axis(fig[4,2],xlabel=labelΠ2,ylabel=labelΔTcloud)
    ax_ΔT3 = Axis(fig[4,3],xlabel=labelΠ3,ylabel=labelΔTcloud)
    ax_ΔTβ = Axis(fig[4,4],xlabel=labelβ,ylabel=labelΔTcloud)

    ax_T1 = Axis(fig[5,1],xlabel=labelΠ1,ylabel=labelTcloud)
    ax_T2 = Axis(fig[5,2],xlabel=labelΠ2,ylabel=labelTcloud)
    ax_T3 = Axis(fig[5,3],xlabel=labelΠ3,ylabel=labelTcloud)
    ax_Tβ = Axis(fig[5,4],xlabel=labelβ,ylabel=labelTcloud)

    # set axis limits
    for ax ∈ [ax_21,ax_31,ax_β1,ax_ΔT1,ax_T1]
        xlims!(ax,logΠ1min,logΠ1max)
    end

    for ax ∈ [ax_32,ax_β2,ax_ΔT2,ax_T2]
        xlims!(ax,logΠ2min,logΠ2max)
    end

    for ax ∈ [ax_β3,ax_ΔT3,ax_T3]
        xlims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_ΔTβ,ax_Tβ]
        xlims!(ax,βmin,βmax)
    end


    ylims!(ax_21,logΠ2min,logΠ2max)


    for ax ∈ [ax_31,ax_32]
        ylims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_β1,ax_β2,ax_β3]
        ylims!(ax,βmin,βmax)
    end


    for ax ∈ [ax_ΔT1,ax_ΔT2,ax_ΔT3,ax_ΔTβ]
        ylims!(ax,ΔTcloudmin,ΔTcloudmax)
    end

    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ylims!(ax,Tcloudmin,Tcloudmax)
    end


    # do rad eq cloud 
    
    Π1_1σ,Π2_1σ,Π3_1σ,ΔTcloud_1σ,β_1σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + σTmin_max_obs),Tmax_min_obs - σTmax_min_obs)
    Π1_2σ,Π2_2σ,Π3_2σ,ΔTcloud_2σ,β_2σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + 2*σTmin_max_obs),Tmax_min_obs - 2*σTmax_min_obs)
    Π1_3σ,Π2_3σ,Π3_3σ,ΔTcloud_3σ,β_3σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = [log10.(Π1_1σ),log10.(Π1_2σ),log10.(Π1_3σ)]
    logΠ2_obs = [log10.(Π2_1σ),log10.(Π2_2σ),log10.(Π2_3σ)]
    logΠ3_obs = [log10.(Π3_1σ),log10.(Π3_2σ),log10.(Π3_3σ)]
    β_obs = if islogβ
        [log10.(β_1σ),log10.(β_2σ),log10.(β_3σ)]
    else
        [β_1σ,β_2σ,β_3σ]
    end
    ΔTcloud_obs = [ΔTcloud_1σ,ΔTcloud_2σ,ΔTcloud_3σ]

    ls = :solid

    for i ∈ reverse(1:3)
        plot2Dboundary_poly!(ax_21,logΠ1_obs[i],logΠ2_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_31,logΠ1_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_32,logΠ2_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_β1,logΠ1_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_β2,logΠ2_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_β3,logΠ3_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_ΔT1,logΠ1_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_ΔT2,logΠ2_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_ΔT3,logΠ3_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_ΔTβ,β_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls)
    end

    # do constant cloud 

    Π1_1σ,Π2_1σ,Π3_1σ,Tcloud_1σ,β_1σ = get_variedparam_obsfilt(sweepname_const,outdir_const,fday*(Tmin_max_obs + σTmin_max_obs),Tmax_min_obs - σTmax_min_obs)
    Π1_2σ,Π2_2σ,Π3_2σ,Tcloud_2σ,β_2σ = get_variedparam_obsfilt(sweepname_const,outdir_const,fday*(Tmin_max_obs + 2*σTmin_max_obs),Tmax_min_obs - 2*σTmax_min_obs)
    Π1_3σ,Π2_3σ,Π3_3σ,Tcloud_3σ,β_3σ = get_variedparam_obsfilt(sweepname_const,outdir_const,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = [log10.(Π1_1σ),log10.(Π1_2σ),log10.(Π1_3σ)]
    logΠ2_obs = [log10.(Π2_1σ),log10.(Π2_2σ),log10.(Π2_3σ)]
    logΠ3_obs = [log10.(Π3_1σ),log10.(Π3_2σ),log10.(Π3_3σ)]
    β_obs = if islogβ
        [log10.(β_1σ),log10.(β_2σ),log10.(β_3σ)]
    else
        [β_1σ,β_2σ,β_3σ]
    end
    Tcloud_obs = [Tcloud_1σ,Tcloud_2σ,Tcloud_3σ]

    ls = :dash
    
    for i ∈ reverse(1:3)
        plot2Dboundary!(ax_21,logΠ1_obs[i],logΠ2_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary!(ax_31,logΠ1_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary!(ax_32,logΠ2_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary!(ax_β1,logΠ1_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary!(ax_β2,logΠ2_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary!(ax_β3,logΠ3_obs[i],β_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_T1,logΠ1_obs[i],Tcloud_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_T2,logΠ2_obs[i],Tcloud_obs[i],ngrid,cobs[i],lw,ls)
        plot2Dboundary_poly!(ax_T3,logΠ3_obs[i],Tcloud_obs[i],ngrid,cobs[i],lw,ls)
        
        # represent very low β on Tcloud-β panel via a line if appropriate
        if i==3 && ischeckβTcloud
            ifilt_β = β_obs[i] .<= βeff0
            if sum(ifilt_β)>0
                Tcloud_obs_βeff0 = Tcloud_obs[i][ifilt_β]
                Tcloudmin4βeff0 = max(minimum(Tcloud_obs_βeff0),fday*(Tmin_max_obs + 3*σTmin_max_obs))
                Tcloudmax4βeff0 = maximum(Tcloud_obs_βeff0)
                filt_Tcloud = Tcloud_obs[i] .<= fday*(Tmin_max_obs + 3*σTmin_max_obs)
                plot2Dboundary_poly!(ax_Tβ,β_obs[i][filt_Tcloud],Tcloud_obs[i][filt_Tcloud],ngrid,cobs[i],lw,ls)
                linesegments!(ax_Tβ,[βeff0,βeff0],[Tcloudmin4βeff0,Tcloudmax4βeff0],linewidth=lw,linestyle=ls,color=cobs[i])
            else
                plot2Dboundary_poly!(ax_Tβ,β_obs[i],Tcloud_obs[i],ngrid,cobs[i],lw,ls)
            end

        else
            plot2Dboundary_poly!(ax_Tβ,β_obs[i],Tcloud_obs[i],ngrid,cobs[i],lw,ls)
        end
    end

     # exclude last row 
    for ax ∈ [ax_21,ax_31,ax_32,ax_β1,ax_β2,ax_β3,ax_ΔT1,ax_ΔT2,ax_ΔT3,ax_ΔTβ]
        hidexdecorations!(ax,ticks=false,grid=false)
    end

    # exclude first column 
    for ax ∈ [ax_32,ax_β2,ax_β3,ax_ΔT2,ax_ΔT3,ax_ΔTβ,ax_T2,ax_T3,ax_Tβ]
        hideydecorations!(ax,ticks=false,grid=false)
    end

    # rotate x ticks for last row 
    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ax.xticklabelrotation = xtickrotation
    end

    # add legend 
    Legend(fig[1:2,3:4],[[PolyElement(color=cobs[i]) for i ∈ 1:3],[LineElement(color=:black,linestyle=ls,linewidth=lw) for ls ∈ [:solid,:dash]]],
    [[rich("1 ",rich("σ",font=:italic)),rich("2 ",rich("σ",font=:italic)),rich("3 ",rich("σ",font=:italic))],
    ["radiative\nbalance","constant"]],["observation consistent to","cloud temperature model"],
    tellheight=false,tellwidth=false,nbanks=3)

    resize_to_layout!(fig)
    save(figdir*"obsvΠs+Tcloudβ_3σ_2modelpoly_$(sweepname_const)_$(sweepname_radbal)$(plotname).pdf",fig)

    nothing 
    
end

function make_Π_Tcloudβ_3σ_figs_radbal_nolw_poly(outdir_nolw,outdir_radbal,figdir,sweepname_nolw,sweepname_radbal;plotname="",fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,cmap=:lajolla,
    xtickrotation=π/4.,ngrid=25,lw=2,islogβ=true,figsize=(600,500),
    logΠ1min=nothing,logΠ1max=nothing,logΠ2min=nothing,logΠ2max=nothing,
    logΠ3min=nothing,logΠ3max=nothing,βmin=nothing,βmax=nothing,
    ΔTcloudmin=nothing,ΔTcloudmax=nothing,ΔTcloudticks=[0,100,200,300],
    αpoly=0.6,αline=1,lw_nolw=2,cobs=reverse(cgrad(cmap,5,categorical=true)[2:4]),clw=:black)

    # set labels 
    labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
    labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
    labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")
    labelΔTcloud = rich(rich("ΔT",font=:italic),subscript("cloud")," [K]")
    labelβ = if islogβ
        rich("log(",rich("β",font=:italic)," [–])")
    else
        rich(rich("β",font=:italic)," [–]")
    end
    

    # make figdir if needed 
    mkpath(figdir)

    # set up figure and axes
    fig = Figure(size=figsize) 
    ax_21 = Axis(fig[1,1],xlabel=labelΠ1,ylabel=labelΠ2)
    ax_31 = Axis(fig[2,1],xlabel=labelΠ1,ylabel=labelΠ3)
    ax_32 = Axis(fig[2,2],xlabel=labelΠ2,ylabel=labelΠ3)


    ax_β1 = Axis(fig[3,1],xlabel=labelΠ1,ylabel=labelβ)
    ax_β2 = Axis(fig[3,2],xlabel=labelΠ2,ylabel=labelβ)
    ax_β3 = Axis(fig[3,3],xlabel=labelΠ3,ylabel=labelβ)

    ax_ΔT1 = Axis(fig[4,1],xlabel=labelΠ1,ylabel=labelΔTcloud,yticks=ΔTcloudticks)
    ax_ΔT2 = Axis(fig[4,2],xlabel=labelΠ2,ylabel=labelΔTcloud)
    ax_ΔT3 = Axis(fig[4,3],xlabel=labelΠ3,ylabel=labelΔTcloud)
    ax_ΔTβ = Axis(fig[4,4],xlabel=labelβ,ylabel=labelΔTcloud)


    # set axis limits
    for ax ∈ [ax_21,ax_31,ax_β1,ax_ΔT1]
        xlims!(ax,logΠ1min,logΠ1max)
    end

    for ax ∈ [ax_32,ax_β2,ax_ΔT2]
        xlims!(ax,logΠ2min,logΠ2max)
    end

    for ax ∈ [ax_β3,ax_ΔT3]
        xlims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_ΔTβ]
        xlims!(ax,βmin,βmax)
    end


    ylims!(ax_21,logΠ2min,logΠ2max)


    for ax ∈ [ax_31,ax_32]
        ylims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_β1,ax_β2,ax_β3]
        ylims!(ax,βmin,βmax)
    end


    for ax ∈ [ax_ΔT1,ax_ΔT2,ax_ΔT3,ax_ΔTβ]
        ylims!(ax,ΔTcloudmin,ΔTcloudmax)
    end



    # do rad eq cloud 
    
    Π1_1σ,Π2_1σ,Π3_1σ,ΔTcloud_1σ,β_1σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + σTmin_max_obs),Tmax_min_obs - σTmax_min_obs)
    Π1_2σ,Π2_2σ,Π3_2σ,ΔTcloud_2σ,β_2σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + 2*σTmin_max_obs),Tmax_min_obs - 2*σTmax_min_obs)
    Π1_3σ,Π2_3σ,Π3_3σ,ΔTcloud_3σ,β_3σ = get_variedparam_obsfilt(sweepname_radbal,outdir_radbal,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = [log10.(Π1_1σ),log10.(Π1_2σ),log10.(Π1_3σ)]
    logΠ2_obs = [log10.(Π2_1σ),log10.(Π2_2σ),log10.(Π2_3σ)]
    logΠ3_obs = [log10.(Π3_1σ),log10.(Π3_2σ),log10.(Π3_3σ)]
    β_obs = if islogβ
        [log10.(β_1σ),log10.(β_2σ),log10.(β_3σ)]
    else
        [β_1σ,β_2σ,β_3σ]
    end
    ΔTcloud_obs = [ΔTcloud_1σ,ΔTcloud_2σ,ΔTcloud_3σ]

    ls = :solid

    for i ∈ reverse(1:3)
        plot2Dboundary_poly!(ax_21,logΠ1_obs[i],logΠ2_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_31,logΠ1_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_32,logΠ2_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_β1,logΠ1_obs[i],β_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_β2,logΠ2_obs[i],β_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_β3,logΠ3_obs[i],β_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_ΔT1,logΠ1_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_ΔT2,logΠ2_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_ΔT3,logΠ3_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
        plot2Dboundary_poly!(ax_ΔTβ,β_obs[i],ΔTcloud_obs[i],ngrid,cobs[i],lw,ls;αpoly=αpoly,αline=αline)
    end

    # do no LW effect cloud 

    Π1_3σ,Π2_3σ,Π3_3σ = get_Πparam_obsfilt(sweepname_nolw,outdir_nolw,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = log10.(Π1_3σ)
    logΠ2_obs = log10.(Π2_3σ)
    logΠ3_obs = log10.(Π3_3σ)


    ls = :dash

    plot2Dboundary!(ax_21,logΠ1_obs,logΠ2_obs,ngrid,clw,lw_nolw,ls)
    plot2Dboundary!(ax_31,logΠ1_obs,logΠ3_obs,ngrid,clw,lw_nolw,ls)
    plot2Dboundary!(ax_32,logΠ2_obs,logΠ3_obs,ngrid,clw,lw_nolw,ls)
    

     # exclude last row 
    for ax ∈ [ax_21,ax_31,ax_32,ax_β1,ax_β2,ax_β3]
        hidexdecorations!(ax,ticks=false,grid=false)
    end

    # exclude first column 
    for ax ∈ [ax_32,ax_β2,ax_β3,ax_ΔT2,ax_ΔT3,ax_ΔTβ]
        hideydecorations!(ax,ticks=false,grid=false)
    end

    # rotate x ticks for last row 
    for ax ∈ [ax_ΔT1,ax_ΔT2,ax_ΔT3,ax_ΔTβ]
        ax.xticklabelrotation = xtickrotation
    end

    # add legend 
    Legend(fig[1:2,3:4],[[PolyElement(color=cobs[i]) for i ∈ 1:3],[LineElement(color=:black,linestyle=:dash,linewidth=lw_nolw)]],
    [[rich("1 ",rich("σ",font=:italic)),rich("2 ",rich("σ",font=:italic)),rich("3 ",rich("σ",font=:italic))],
    [rich("3 ",rich("σ",font=:italic))]],["OBSERVATIONAL CONSISTENCY\n\nwith longwave cloud effects","without longwave cloud effects"],
    tellheight=false,tellwidth=false,nbanks=3)

    resize_to_layout!(fig)
    save(figdir*"obsvΠs+Tcloudβ_3σ_radbal_nolw_poly_$(sweepname_nolw)_$(sweepname_radbal)$(plotname).pdf",fig)

    nothing 
    
end

function find2Dboundary(param1,param2;ngrid=50,isadjustmidend=true)
    """
    grid in one direction and get min and max values 
    """
    grid1 = LinRange(minimum(param1),maximum(param1),ngrid+1)
    midgrid1 = collect(0.5 .* (grid1[1:(end-1)] .+ grid1[2:end]))
    if isadjustmidend
        midgrid1[1] = grid1[1]
        midgrid1[end] = grid1[end]
    end
    param2minmax = zeros(2,ngrid)
    for i1 ∈ 1:ngrid 
        # find values within gridding of param 1 
        filt = (param1 .>= grid1[i1]) .&& (param1 .<= grid1[i1+1])
        if sum(filt)>0 
            param2_grid1 = param2[filt]
            param2minmax[1,i1] = minimum(param2_grid1)
            param2minmax[2,i1] = maximum(param2_grid1)
        else
            param2minmax[:,i1] .= NaN 
        end 
    end
    midgrid1,param2minmax
end

function plot2Dboundary!(ax,paramx,paramy,ngrid,color,lw,ls)
    # calculate boundary 
    midgridx,paramyminmax = find2Dboundary(paramx,paramy;ngrid=ngrid)
    # remove regions of grid where no parameters (set to NaN)
    filtnan = .!isnan.(view(paramyminmax,1,:))
    midgridx = midgridx[filtnan]
    paramyminmax = paramyminmax[:,filtnan]
    # combine into 1 line 
    paramx_boundary = vcat(midgridx,reverse(midgridx),[midgridx[1]])
    paramy_boundary = vcat(view(paramyminmax,1,:),reverse(view(paramyminmax,2,:)),[paramyminmax[1,1]])
    # plot 
    lines!(ax,paramx_boundary,paramy_boundary,color=color,linewidth=lw,linestyle=ls)
    nothing 
end

function plot2Dboundary_poly!(ax,paramx,paramy,ngrid,color,lw,ls;αpoly=0.6,αline=1)
    # calculate boundary 
    midgridx,paramyminmax = find2Dboundary(paramx,paramy;ngrid=ngrid)
    # remove regions of grid where no parameters (set to NaN)
    filtnan = .!isnan.(view(paramyminmax,1,:))
    midgridx = midgridx[filtnan]
    paramyminmax = paramyminmax[:,filtnan]
    # combine into 1 line 
    paramx_boundary = vcat(midgridx,reverse(midgridx),[midgridx[1]])
    paramy_boundary = vcat(view(paramyminmax,1,:),reverse(view(paramyminmax,2,:)),[paramyminmax[1,1]])
    boundary = Point2f.(paramx_boundary,paramy_boundary)
    # plot 
    lines!(ax,paramx_boundary,paramy_boundary,color=(color,αline),linewidth=lw,linestyle=ls)
    poly!(ax,boundary,color=(color,αpoly))
    nothing 
end

function get_Πparam_obsfilt(sweepname,outdir,Tmin_max_obs,Tmax_min_obs)
    """
    load bulk parameter combinations consistent with observations  
    """
    # file name 
    fnc = outdir*sweepname*".nc"
    # load 4.5 μm brightness temperatures to check obs consistency 
    T_04p5_min = ncread(fnc,"T_4p5_min")
    T_04p5_max = ncread(fnc,"T_4p5_max")
    # load end state flags to ensure limit cycles 
    endstateflag = ncread(fnc,"end_state_flag")
    
    # filter min and max 4.5 μm brightness temperatures for consistency with inputted observation limits
    # also make sure results are limit cycles 
    iobs = (T_04p5_min .< Tmin_max_obs) .&& (T_04p5_max .> Tmax_min_obs) .&& (endstateflag .== 2)

    Π1 = ncread(fnc,"Pi1")[iobs]
    Π2 = ncread(fnc,"Pi2")[iobs]
    Π3 = ncread(fnc,"Pi3")[iobs]

    Π1,Π2,Π3
end

function make_Π_3σ_figs_poly(outdir,figdir,sweepname;plotname="",fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,cmap=:lajolla,
    xtickrotation=π/4.,marker=:rect,ms=3,lw=2,islogβ=false,figsize=(600,500),
    logΠ1min=nothing,logΠ1max=nothing,logΠ2min=nothing,logΠ2max=nothing,
    logΠ3min=nothing,logΠ3max=nothing,βmin=nothing,βmax=nothing,
    Tcloudmin=nothing,Tcloudmax=nothing,ngrid=25)


    # set labels 
    labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
    labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
    labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")


    # set colors 
    cobs = cgrad(cmap,5,categorical=true)[2:4]

    # make figdir if needed 
    mkpath(figdir)

    # set up figure and axes
    fig = Figure(size=figsize) 
    ax_21 = Axis(fig[1,1],xlabel=labelΠ1,ylabel=labelΠ2)
    ax_31 = Axis(fig[2,1],xlabel=labelΠ1,ylabel=labelΠ3)
    ax_32 = Axis(fig[2,2],xlabel=labelΠ2,ylabel=labelΠ3)

    # set axis limits
    for ax ∈ [ax_21,ax_31]
        xlims!(ax,logΠ1min,logΠ1max)
    end

    for ax ∈ [ax_31,ax_32]
        ylims!(ax,logΠ3min,logΠ3max)
    end

    xlims!(ax_32,logΠ2min,logΠ2max)
    ylims!(ax_21,logΠ2min,logΠ2max)

    
    Π1_1σ,Π2_1σ,Π3_1σ = get_Πparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + σTmin_max_obs),Tmax_min_obs - σTmax_min_obs)
    Π1_2σ,Π2_2σ,Π3_2σ = get_Πparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + 2*σTmin_max_obs),Tmax_min_obs - 2*σTmax_min_obs)
    Π1_3σ,Π2_3σ,Π3_3σ = get_Πparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = [log10.(Π1_1σ),log10.(Π1_2σ),log10.(Π1_3σ)]
    logΠ2_obs = [log10.(Π2_1σ),log10.(Π2_2σ),log10.(Π2_3σ)]
    logΠ3_obs = [log10.(Π3_1σ),log10.(Π3_2σ),log10.(Π3_3σ)]

    ls = :solid

    for i ∈ reverse(1:3)
        if length(logΠ1_obs[i]) > 0
            plot2Dboundary_poly!(ax_21,logΠ1_obs[i],logΠ2_obs[i],ngrid,cobs[i],lw,ls)
            plot2Dboundary_poly!(ax_31,logΠ1_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
            plot2Dboundary_poly!(ax_32,logΠ2_obs[i],logΠ3_obs[i],ngrid,cobs[i],lw,ls)
        end
    end

     # exclude last row 
     for ax ∈ [ax_21]
        hidexdecorations!(ax,ticks=false,grid=false)
    end

    # exclude first column 
    for ax ∈ [ax_32]
        hideydecorations!(ax,ticks=false,grid=false)
    end


    # add legend 
    Legend(fig[1,2],[PolyElement(color=cobs[i]) for i ∈ 1:3],
    [rich("1 ",rich("σ",font=:italic)),rich("2 ",rich("σ",font=:italic)),rich("3 ",rich("σ",font=:italic))],
    "observation consistent to",
    tellheight=false,tellwidth=false,nbanks=3)

    resize_to_layout!(fig)
    save(figdir*"obsvΠs_3σ_poly_$(sweepname)$(plotname).pdf",fig)

    nothing 
    
end

function make_Π_Tcloudβ_3σ_figs_scatter(outdir,figdir,sweepname;plotname="",fday=1.1,Tmin_max_obs=873.,
    σTmin_max_obs=167.,Tmax_min_obs=2816.,σTmax_min_obs=368.,cmap=:lajolla,
    xtickrotation=π/4.,marker=:rect,ms=3,lw=2,islogβ=false,figsize=(600,500),
    logΠ1min=nothing,logΠ1max=nothing,logΠ2min=nothing,logΠ2max=nothing,
    logΠ3min=nothing,logΠ3max=nothing,βmin=nothing,βmax=nothing,
    Tcloudmin=nothing,Tcloudmax=nothing)

    model_type = get_modeltype(sweepname,outdir)

    # set labels 
    labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
    labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
    labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")
    labelTcloud = if model_type=="const"
        rich(rich("T",font=:italic),subscript("cloud")," [K]")
    elseif model_type=="rad_bal"
        rich(rich("ΔT",font=:italic),subscript("cloud")," [K]")
    end
    labelβ = if islogβ
        rich("log ",rich("β",font=:italic)," [–]")
    else
        rich(rich("β",font=:italic)," [–]")
    end

    # set colors 
    cobs = cgrad(cmap,5,categorical=true)[2:4]

    # make figdir if needed 
    mkpath(figdir)

    # set up figure and axes
    fig = Figure(size=figsize) 
    ax_21 = Axis(fig[1,1],xlabel=labelΠ1,ylabel=labelΠ2)
    ax_31 = Axis(fig[2,1],xlabel=labelΠ1,ylabel=labelΠ3)
    ax_32 = Axis(fig[2,2],xlabel=labelΠ2,ylabel=labelΠ3)


    ax_β1 = Axis(fig[3,1],xlabel=labelΠ1,ylabel=labelβ)
    ax_β2 = Axis(fig[3,2],xlabel=labelΠ2,ylabel=labelβ)
    ax_β3 = Axis(fig[3,3],xlabel=labelΠ3,ylabel=labelβ)

    ax_T1 = Axis(fig[4,1],xlabel=labelΠ1,ylabel=labelTcloud)
    ax_T2 = Axis(fig[4,2],xlabel=labelΠ2,ylabel=labelTcloud)
    ax_T3 = Axis(fig[4,3],xlabel=labelΠ3,ylabel=labelTcloud)
    ax_Tβ = Axis(fig[4,4],xlabel=labelβ,ylabel=labelTcloud)

    # set axis limits
    for ax ∈ [ax_21,ax_31,ax_β1,ax_T1]
        xlims!(ax,logΠ1min,logΠ1max)
    end

    for ax ∈ [ax_32,ax_β2,ax_T2]
        xlims!(ax,logΠ2min,logΠ2max)
    end

    for ax ∈ [ax_β3,ax_T3]
        xlims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_Tβ]
        xlims!(ax,βmin,βmax)
    end


    ylims!(ax_21,logΠ2min,logΠ2max)


    for ax ∈ [ax_31,ax_32]
        ylims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_β1,ax_β2,ax_β3]
        ylims!(ax,βmin,βmax)
    end


    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ylims!(ax,Tcloudmin,Tcloudmax)
    end

    
    Π1_1σ,Π2_1σ,Π3_1σ,Tcloud_1σ,β_1σ = get_variedparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + σTmin_max_obs),Tmax_min_obs - σTmax_min_obs)
    Π1_2σ,Π2_2σ,Π3_2σ,Tcloud_2σ,β_2σ = get_variedparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + 2*σTmin_max_obs),Tmax_min_obs - 2*σTmax_min_obs)
    Π1_3σ,Π2_3σ,Π3_3σ,Tcloud_3σ,β_3σ = get_variedparam_obsfilt(sweepname,outdir,fday*(Tmin_max_obs + 3*σTmin_max_obs),Tmax_min_obs - 3*σTmax_min_obs)

    logΠ1_obs = [log10.(Π1_1σ),log10.(Π1_2σ),log10.(Π1_3σ)]
    logΠ2_obs = [log10.(Π2_1σ),log10.(Π2_2σ),log10.(Π2_3σ)]
    logΠ3_obs = [log10.(Π3_1σ),log10.(Π3_2σ),log10.(Π3_3σ)]
    β_obs = if islogβ
        [log10.(β_1σ),log10.(β_2σ),log10.(β_3σ)]
    else
        [β_1σ,β_2σ,β_3σ]
    end
    Tcloud_obs = [Tcloud_1σ,Tcloud_2σ,Tcloud_3σ]

    ls = :solid

    for i ∈ reverse(1:3)
        scatter!(ax_21,logΠ1_obs[i],logΠ2_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_31,logΠ1_obs[i],logΠ3_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_32,logΠ2_obs[i],logΠ3_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_β1,logΠ1_obs[i],β_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_β2,logΠ2_obs[i],β_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_β3,logΠ3_obs[i],β_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_T1,logΠ1_obs[i],Tcloud_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_T2,logΠ2_obs[i],Tcloud_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_T3,logΠ3_obs[i],Tcloud_obs[i],color=cobs[i],markersize=ms,marker=marker)
        scatter!(ax_Tβ,β_obs[i],Tcloud_obs[i],color=cobs[i],markersize=ms,marker=marker)
    end

     # exclude last row 
     for ax ∈ [ax_21,ax_31,ax_32,ax_β1,ax_β2,ax_β3]
        hidexdecorations!(ax,ticks=false,grid=false)
    end

    # exclude first column 
    for ax ∈ [ax_32,ax_β2,ax_β3,ax_T2,ax_T3,ax_Tβ]
        hideydecorations!(ax,ticks=false,grid=false)
    end

    # rotate x ticks for last row 
    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ax.xticklabelrotation = xtickrotation
    end

    # add legend 
    Legend(fig[1:2,3:4],[[PolyElement(color=cobs[i]) for i ∈ 1:3],[LineElement(color=:black,linestyle=ls,linewidth=lw) for ls ∈ [:solid,:dash]]],
    [[rich("1 ",rich("σ",font=:italic)),rich("2 ",rich("σ",font=:italic)),rich("3 ",rich("σ",font=:italic))],
    ["radiative\nbalance","const"]],["observation consistent to","cloud temperature model"],
    tellheight=false,tellwidth=false,nbanks=3)

    resize_to_layout!(fig)
    save(figdir*"obsvΠs+Tcloudβ_3σ_scatter_$(sweepname)$(plotname).png",fig,px_per_unit=5)

    nothing 
    
end

function make_Π_Tcloudβ_figs_scatter(outdir,figdir,sweepname,iplot;
    plotname="",cmap=:lajolla,
    xtickrotation=π/4.,marker=:rect,ms=3,lw=2,islogβ=false,figsize=(600,500),
    logΠ1min=nothing,logΠ1max=nothing,logΠ2min=nothing,logΠ2max=nothing,
    logΠ3min=nothing,logΠ3max=nothing,βmin=nothing,βmax=nothing,
    Tcloudmin=nothing,Tcloudmax=nothing,c=:black)

    model_type = get_modeltype(sweepname,outdir)

    # set labels 
    labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
    labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
    labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")
    labelTcloud = if model_type=="const"
        rich(rich("T",font=:italic),subscript("cloud")," [K]")
    elseif model_type=="rad_bal"
        rich(rich("ΔT",font=:italic),subscript("cloud")," [K]")
    end
    labelβ = if islogβ
        rich("log ",rich("β",font=:italic)," [–]")
    else
        rich(rich("β",font=:italic)," [–]")
    end

    # make figdir if needed 
    mkpath(figdir)

    # set up figure and axes
    fig = Figure(size=figsize) 
    ax_21 = Axis(fig[1,1],xlabel=labelΠ1,ylabel=labelΠ2)
    ax_31 = Axis(fig[2,1],xlabel=labelΠ1,ylabel=labelΠ3)
    ax_32 = Axis(fig[2,2],xlabel=labelΠ2,ylabel=labelΠ3)


    ax_β1 = Axis(fig[3,1],xlabel=labelΠ1,ylabel=labelβ)
    ax_β2 = Axis(fig[3,2],xlabel=labelΠ2,ylabel=labelβ)
    ax_β3 = Axis(fig[3,3],xlabel=labelΠ3,ylabel=labelβ)

    ax_T1 = Axis(fig[4,1],xlabel=labelΠ1,ylabel=labelTcloud)
    ax_T2 = Axis(fig[4,2],xlabel=labelΠ2,ylabel=labelTcloud)
    ax_T3 = Axis(fig[4,3],xlabel=labelΠ3,ylabel=labelTcloud)
    ax_Tβ = Axis(fig[4,4],xlabel=labelβ,ylabel=labelTcloud)

    # set axis limits
    for ax ∈ [ax_21,ax_31,ax_β1,ax_T1]
        xlims!(ax,logΠ1min,logΠ1max)
    end

    for ax ∈ [ax_32,ax_β2,ax_T2]
        xlims!(ax,logΠ2min,logΠ2max)
    end

    for ax ∈ [ax_β3,ax_T3]
        xlims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_Tβ]
        xlims!(ax,βmin,βmax)
    end


    ylims!(ax_21,logΠ2min,logΠ2max)


    for ax ∈ [ax_31,ax_32]
        ylims!(ax,logΠ3min,logΠ3max)
    end

    for ax ∈ [ax_β1,ax_β2,ax_β3]
        ylims!(ax,βmin,βmax)
    end


    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ylims!(ax,Tcloudmin,Tcloudmax)
    end

    
    Π1,Π2,Π3,Tcloud,β = get_parameters_varied(sweepname,outdir)

    logΠ1 = log10.(Π1)[iplot]
    logΠ2 = log10.(Π2)[iplot]
    logΠ3 = log10.(Π3)[iplot]
    β = if islogβ
        log10.(β)[iplot]
    else
        β[iplot]
    end
    Tcloud = Tcloud[iplot]

    
    scatter!(ax_21,logΠ1,logΠ2,color=c,markersize=ms,marker=marker)
    scatter!(ax_31,logΠ1,logΠ3,color=c,markersize=ms,marker=marker)
    scatter!(ax_32,logΠ2,logΠ3,color=c,markersize=ms,marker=marker)
    scatter!(ax_β1,logΠ1,β,color=c,markersize=ms,marker=marker)
    scatter!(ax_β2,logΠ2,β,color=c,markersize=ms,marker=marker)
    scatter!(ax_β3,logΠ3,β,color=c,markersize=ms,marker=marker)
    scatter!(ax_T1,logΠ1,Tcloud,color=c,markersize=ms,marker=marker)
    scatter!(ax_T2,logΠ2,Tcloud,color=c,markersize=ms,marker=marker)
    scatter!(ax_T3,logΠ3,Tcloud,color=c,markersize=ms,marker=marker)
    scatter!(ax_Tβ,β,Tcloud,color=c,markersize=ms,marker=marker)
   

     # exclude last row 
     for ax ∈ [ax_21,ax_31,ax_32,ax_β1,ax_β2,ax_β3]
        hidexdecorations!(ax,ticks=false,grid=false)
    end

    # exclude first column 
    for ax ∈ [ax_32,ax_β2,ax_β3,ax_T2,ax_T3,ax_Tβ]
        hideydecorations!(ax,ticks=false,grid=false)
    end

    # rotate x ticks for last row 
    for ax ∈ [ax_T1,ax_T2,ax_T3,ax_Tβ]
        ax.xticklabelrotation = xtickrotation
    end

    resize_to_layout!(fig)
    save(figdir*"obsvΠs+Tcloudβ_3σ_scatter_$(sweepname)$(plotname).png",fig,px_per_unit=5)

    nothing 
    
end


# FUNCTIONS TO PLOT SOLUTIONS #####################################

end # end module 