using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
using Random 
using Roots 
using Interpolations
using NetCDF
"""
this script generates the data for figure S7 and plots it
"""

# set output directory 
outdir = "out/SI/"

# set figure directory 
figdir = "sfigs/"
mkpath(figdir)


# set model parameters the same as Figure S6
Π1_B = 1.7508340590339526 # [Pa⁻¹]
Π2_B = 377.1454124774199 # [ ]
Π3_B = 28.849960362230448 # [kg s⁻³ K⁻¹]
β_B = 3.3060284539486826 # [ ]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
ΔTcloud_B = 322.6134997840408 # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
pB = [Π1_B, Π2_B, Π3_B, T_ref, p_ref, α, S₀, f, β_B, ΔTcloud_B, Tsolidus]


# select which β values to plot from figure S6
nβ = 250
βs = LinRange(0.,1.5,nβ)
βplots = βs[[137,157,174,179,200,225]] 

# run parameters 
reltol = 1e-8
abstol = 1e-10

# set how long to integrate  
t̂end = 3e3


# do runs for various β values 
for (iβ,β) ∈ enumerate(βplots) 
    runname = "varyβ_$(iβ)"
    pB[9] = β
    outputrun_radbal2(pB,outdir,runname;maxiters=2e7,reltol=reltol,
        abstol=abstol,t̂end=t̂end)
end


# function to calculate Teq for rootfinding 
function calcTeq0_long(t,f,Teq0)
    f(t) - Teq0
end

# set up figure and plot

figdir = "sfigs/"
mkpath(figdir)

fig = Figure() 

nβplots = length(βplots)

cmap = Makie.to_colormap(:lajolla10)[1:8]
cβs = cgrad(cmap,nβplots,categorical=true)


ls = [:solid,:solid]
α = 0.8

ytick1s = [2700,2600,2500,2500,2400,2400]
ytick2 = 2800

for iβ ∈ 1:nβplots
    ax = Axis(fig[iβ,1],yticks=[ytick1s[iβ],ytick2])
    if iβ==nβplots
        ax.xlabel = rich(rich("t / d",font=:italic)," [-]")
    else
        hidexdecorations!(ax,ticks=false,grid=false)
    end
    xlims!(ax,0,8.116755100044452)

    runname = "varyβ_$(iβ)_inner"
    # load run 
    fnc = outdir*runname*".nc"
    Tsurf = ncread(fnc,"T_surf")
    ts = ncread(fnc,"t-hat")
    Tsurf_0 = ncread(fnc,"T_surf_0")[1]
    filt = ts .> (ts[end]-30)

    # find same start point 
    endts = ts[filt]
    endTs = Tsurf[filt]
    endΔTs = endTs .- Tsurf_0
    filt_pos_endΔT = endΔTs .> 0
    filt_neg_endΔT = endΔTs .< 0
    tstart = endts[filt_pos_endΔT][1]
    tend = endts[filt_neg_endΔT][end]
    interp_Tsurfvt = linear_interpolation(endts, endTs)
    calcTeq0(t) = calcTeq0_long(t,interp_Tsurfvt,Tsurf_0)
    tswhereTeq = find_zeros(calcTeq0,tstart,tend)

    istart = findmin(abs.(tswhereTeq[1].-endts))[2]
    
    lines!(ax,endts[istart:end].-endts[istart],endTs[istart:end],linewidth=2,label="$iβ",color=cβs[iβ])


end
rowgap!(fig.layout, 5)
Tsurflabel = rich(rich("T",font=:italic),subscript("surf ")," [K]")
Label(fig[1:nβplots, 0], Tsurflabel, rotation = pi/2)
Legend(fig[1:nβplots,2],[PolyElement(color=cβs[iβ]) for iβ ∈ 1:nβplots],["$(round(βplots[iβ],sigdigits=3))" for iβ ∈ 1:nβplots],rich(rich("β",font=:bold_italic)," [-]"))

save(figdir*"sfigure7.pdf",fig)

