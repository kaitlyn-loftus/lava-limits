using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
using Random 
using OrdinaryDiffEq
using NetCDF
"""
this script generates the data for SI figure 5 and plots it
"""

# set up model parameters for different numerical approaches

Π1 = 1.7508340590339526 # [Pa⁻¹]
Π2 = 377.1454124774199 # [ ]
Π3 = 28.849960362230448 # [kg s⁻³ K⁻¹]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
β = 3.3060284539486826 # [ ]
ΔTcloud = 322.6134997840408 # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
p = [Π1, Π2, Π3, T_ref, p_ref, α, S₀, f, β, ΔTcloud, Tsolidus]

# set default tolerances  
reltol = 1e-8
abstol = 1e-10

# set how long to integrate  
t̂end = 3e3


# set output file path directory 
outdir = "out/SI/"

# set figure directory 
figdir = "sfigs/"
mkpath(figdir)

# normal numerics 
runname = "erratic_normalnumerics"
outputrun_radbal2(p,outdir,runname;maxiters=2e7,reltol=reltol,
        abstol=abstol,t̂end=t̂end)


# lower tolerance 
runname = "erratic_lowertol"
outputrun_radbal2(p,outdir,runname;maxiters=2e7,reltol=reltol*1e-1,
                abstol=abstol,t̂end=t̂end)

# different numerical solver 
runname = "erratic_diffsolver"
alg = Kvaerno4()
outputrun_radbal2(p,outdir,runname;maxiters=2e7,reltol=reltol,
                abstol=abstol,t̂end=t̂end,alg=alg)



# plot results of different numerical approaches 

runnames = ["erratic_normalnumerics","erratic_lowertol","erratic_diffsolver"]

# set up random number seed because can't show all timesteps 
seed = 2478352
rng = Xoshiro(seed)

nphasespace = 1000

Tbrightlabel_0p5 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=0.5",font=:italic),"μm",rich(")",font=:italic)," [K]")
Tbrightlabel_4p5 = rich(rich("T",font=:italic),subscript("bright"),rich("(λ=4.5",font=:italic),"μm",rich(")",font=:italic)," [K]")
tlabel = rich(rich("t / d",font=:italic)," [-]")
Tsurflabel = rich(rich("T",font=:italic),subscript("surf")," [K]")
τSWlabel = rich(rich("τ",font=:italic),subscript("SW")," [ ]")
fig = Figure() 
ax1 = Axis(fig[1,1],xlabel=Tsurflabel,ylabel=τSWlabel,yscale=log10,yticks=LogTicks(-6:2:2),title="A",titlealign=:left)
ax2 = Axis(fig[2,1],xlabel=Tbrightlabel_4p5,ylabel=Tbrightlabel_0p5,title="B",titlealign=:left)
ax3 = Axis(fig[1,2],xlabel=tlabel,ylabel=Tsurflabel,title="C",titlealign=:left)
ax4 = Axis(fig[2,2],xlabel=tlabel,ylabel=τSWlabel,yscale=log10,yticks=LogTicks(-4:2:2),title="D",titlealign=:left)
# add twin axis for fixed points
Tsurfeqlabel = rich(rich("T",font=:italic),subscript("surf"),superscript("∗"))
τSWeqlabel = rich(rich("τ",font=:italic),subscript("SW"),superscript("∗"))
Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = p
Tsurfeq,τSWeq = DDEModel.findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)
ax3fixed = Axis(fig[1,2], yaxisposition=:right,yticks=([Tsurfeq],[Tsurfeqlabel]))
hidespines!(ax3fixed)
hidexdecorations!(ax3fixed)
ax4fixed = Axis(fig[2,2], yaxisposition=:right,yticks=([τSWeq],[τSWeqlabel]),yscale=log10)
hidespines!(ax4fixed)
hidexdecorations!(ax4fixed)


Δtplot = 5 
xlims!(ax3,0,Δtplot)
xlims!(ax4,0,Δtplot)

xlims!(ax3fixed,0,Δtplot)
xlims!(ax4fixed,0,Δtplot)

colors = Makie.wong_colors()
lss = [:solid,:dash,:dot]

α = 0.8

for (irun,runname) ∈ enumerate(runnames)
    
    fnc = outdir*runname*"_inner.nc"
    Tsurf = ncread(fnc,"T_surf")
    ts = ncread(fnc,"t-hat")
    Tsurf_0 = ncread(fnc,"T_surf_0")[1]
    τSW_0 = ncread(fnc,"tau_sw_0")[1]
    τSW = ncread(fnc,"tau_sw")
    T_4p5 = ncread(fnc,"T_4p5")
    T_0p5 = ncread(fnc,"T_0p5")

    filt_tplot = ts .> (ts[end] - Δtplot)

    ts_filt_tplot = ts[filt_tplot]
    ts_filt_tplot = ts_filt_tplot .- ts_filt_tplot[1]
    Tsurf_filt_tplot = Tsurf[filt_tplot]
    τSW_filt_tplot = τSW[filt_tplot]


    lines!(ax3,ts_filt_tplot,Tsurf_filt_tplot,color=(colors[irun],α),linewidth=1)
    lines!(ax4,ts_filt_tplot,τSW_filt_tplot,color=(colors[irun],α),linewidth=1)
    

     # outline 
    filt_boundary = ts .> (ts[end] - 2000)
    Tsurf_filt_boundary = Tsurf[filt_boundary]
    τSW_filt_boundary = τSW[filt_boundary]
    T_4p5_filt_boundary = T_4p5[filt_boundary]
    T_0p5_filt_boundary = T_0p5[filt_boundary]

    color = colors[irun]
    lw = 3
    ls = lss[irun]
    ngrid = 30

    iplot = rand(rng,1:sum(filt_boundary),nphasespace)

    scatter!(ax1,Tsurf_filt_boundary[iplot],τSW_filt_boundary[iplot],markersize=3,color=(colors[irun],α))
    scatter!(ax2,T_4p5_filt_boundary[iplot],T_0p5_filt_boundary[iplot],markersize=3,color=(colors[irun],α))

    DDEModel.plot2Dboundary!(ax1,Tsurf_filt_boundary,τSW_filt_boundary,ngrid,color,lw,ls)

    DDEModel.plot2Dboundary!(ax2,T_4p5_filt_boundary,T_0p5_filt_boundary,ngrid,color,lw,ls)
    
end
# adjust t column to be wider 
colsize!(fig.layout, 2, Relative(2/3))


ylims!(ax3fixed,ax3.yaxis.attributes.limits[])
ylims!(ax4fixed,ax4.yaxis.attributes.limits[])


axislegend(ax4,[PolyElement(color=colors[i]) for i ∈ 1:3],["$(i)" for i ∈ 1:3],
"numerical method",orientation=:horizontal,labelsize=8,titlesize=10,position=:lb,padding=4,patchsize=(8,8),titlegap=4)

save(figdir*"sfigure5.pdf",fig)

# not sure why this doesn't work the first time
# guess limits are only set upon saving...
ylims!(ax3fixed,ax3.yaxis.attributes.limits[])
ylims!(ax4fixed,ax4.yaxis.attributes.limits[])

rowgap!(fig.layout, 5)

save(figdir*"sfigure5.pdf",fig)
