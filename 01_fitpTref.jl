using Pkg
Pkg.activate(".")
using CSV
using DataFrames
using CairoMakie
using LsqFit
"""
this script fits pref and Tref for gas species in equilbrium with a BSE magma ocean
"""

function calclnp(T,param)
    """
    calculate natural logarithm of partial pressure as a function of temperature 
    assuming Clausius-Clapeyron relationship 
    ln p = ln pref - Tref / T
    inputs:
        * T [K] - temperature 
        * param [Tuple] - parameters 
            (1) ln_pref [ln(Pa)] - reference natural logarithm of pressure 
            (2) Tref [K] - reference temperature 
    output:
        * ln p [ln(Pa)] - partial pressure 
    """
    ln_pref,Tref = param
    ln_pref .- Tref ./ T
end

function fit_pref_Tref(pps_bar,pp_names,name)
    # number of species 
    npp = size(pps_bar)[1]
    # arrays to store fitted parameters for each species 
    pref = zeros(npp)
    Tref = zeros(npp)
    for (i,pp_bar) ∈ enumerate(pps_bar)
        pp_Pa = pp_bar*1e5 # [convert bar to Pa]
    
        # fit model outputs to assumed p(T) functional form 
        fit = curve_fit(calclnp, Ts, log.(pp_Pa), [25.,300.]) 
        
        # store fitted parameters 
        pref[i] = exp(fit.param[1])
        Tref[i] = fit.param[2] 
    
        # print out fitted parameters 
        println("species: $(pp_names[i])")
        println("\tpref =  $(exp(fit.param[1])) Pa")
        println("\tTref = $(fit.param[2]) K\n")
    end
    # write out results to csv for easy reference 
    df = DataFrame(species=pp_names)
    df[:,Symbol("pref [Pa]")] = pref
    df[:,Symbol("Tref [K]")] = Tref
    CSV.write("data/magmaoc_p_v_T_fit_K16_SF09_$(name).csv", df)
    pref,Tref
end

function plot_pref_Tref_fits(pps_bar,pp_labels,name;nT=100)
    # number of species 
    npp = size(pps_bar)[1]
    # set up fit validation plot 
    fig = Figure()
    ax = Axis(fig[1,1],yscale=log10) 
    ccycle = cgrad(:roma, npp, categorical = true) # set up color cycle
    Tfits = LinRange(1400,3000,nT)
    pfits = zeros(nT)
    for (i,pp_bar) ∈ enumerate(pps_bar)
        pp_Pa = pp_bar*1e5 # [convert bar to Pa]
        # fit model outputs to assumed p(T) functional form 
        fit = curve_fit(calclnp, Ts, log.(pp_Pa), [25.,300.]) 
        # calculate fitted p for validation plot 
        for (j,Tfit) ∈ enumerate(Tfits)
            pfits[j] = exp(calclnp(Tfit,fit.param))
        end
        # plot model data  
        lines!(ax,Ts,pp_Pa,label=pp_labels[i],linewidth=3,color=(ccycle[i],0.5))
        # plot fitted expression 
        lines!(ax,Tfits,pfits,linestyle=:dash,linewidth=3,color=(ccycle[i],0.5))
    end
    # finish plot details 
    xlims!(ax,1400,3000)
    ylims!(ax,1e-9,1e4)
    ax.xlabel = "temperature [K]"
    ax.ylabel = "pressure [Pa]"
    Legend(fig[1,2],[[LineElement(color=:gray,linewidth=3),LineElement(color=:gray,linestyle=:dash,linewidth=3)],
    [PolyElement(color = c) for c ∈ ccycle]],
    [["data","fit"],[pp_label for pp_label ∈ pp_labels]],["","species",]) 
    mkpath("sfigs/")
    save("sfigs/magmaoc_K16_SF09_pvT_$(name).pdf",fig) # save out figure 
    nothing 
end


println("0% VAPORIZATION")
println("================")
# read in model data for 0% vaporization 
fname_0vap = "data/K16_SF09_0vap.csv"
df_0vap = CSV.read(fname_0vap,DataFrame; header=true, delim=',')
Ts = df_0vap[:,:T]
pO2 = df_0vap[:,:PO2]
pO = df_0vap[:,:PO]
pMg = df_0vap[:,:PMg]
pMgO = df_0vap[:,:PMgO]
pNa = df_0vap[:,:PNa]
pSiO = df_0vap[:,:PSiO]
pSiO2 = df_0vap[:,:PSiO2]
pFe = df_0vap[:,:PFe]
pFeO = df_0vap[:,:PFeO]
pNaO = df_0vap[:,:PNaO]

# do fitting 
pps_bar_0vap = [pO,pO2,pMg,pMgO,pSiO,pSiO2,pFe,pFeO,pNa,pNaO]
pp_names_0vap = ["O","O2","Mg","MgO","SiO","SiO2","Fe","FeO","Na","NaO"]
fit_pref_Tref(pps_bar_0vap,pp_names_0vap,"0vap")
# plot to check fit 
pp_labels_0vap = ["O","O₂","Mg","MgO","SiO","SiO₂","Fe","FeO","Na","NaO"]
plot_pref_Tref_fits(pps_bar_0vap,pp_labels_0vap,"0vap";nT=100)



# repeat for 20% vaporization
println("\n\n")
println("20% VAPORIZATION")
println("================")

# read in model data
fname = "data/K16_SF09_20vap.csv"
df = CSV.read(fname,DataFrame; header=true, delim=',')
Ts = df[:,:T]
pO2 = df[:,:PO2]
pO = df[:,:PO]
pMg = df[:,:PMg]
pMgO = df[:,:PMgO]
pNa = df[:,:PNa]
pSiO = df[:,:PSiO]
pSiO2 = df[:,:PSiO2]
pFe = df[:,:PFe]
pFeO = df[:,:PFeO]
pNaO = df[:,:PNaO]

# do fitting 
# note not all species included because poor fit for Clausius-Clapeyron relationship 
pps_bar_20vap = [pO,pO2,pMg,pMgO,pSiO,pSiO2]
pp_names_20vap = ["O","O2","Mg","MgO","SiO","SiO2"]
fit_pref_Tref(pps_bar_20vap,pp_names_20vap,"20vap")
# plot to check fit 
pp_labels_20vap = ["O","O₂","Mg","MgO","SiO","SiO₂"]
plot_pref_Tref_fits(pps_bar_20vap,pp_labels_20vap,"20vap";nT=100)
