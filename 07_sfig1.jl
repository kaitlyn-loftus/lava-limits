using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
using NetCDF 
"""
this script generates the data for SI figure 1 and plots it
"""

# set output file path details 
# outputs saved as 
# outdir*runname*"_inner.nc"
# outdir*runname*"_outer.nc"
outdir = "out/SI/"
runname = "sfig1"

# set model  parameters 
Π1 = 93.3 # [Pa⁻¹]
Π2 = 18.8 # [ ]
Π3 = 439. # [kg s⁻³ K⁻¹]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
β = 0.00236 # [ ]
ΔTcloud = 28.5 # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
p_sfig1 = [Π1, Π2, Π3, T_ref, p_ref, α, S₀, f, β, ΔTcloud, Tsolidus]

# run radiative balance model 
outputrun_radbal2(p_sfig1,outdir,runname)


# load outputs 
fnc = outdir*runname*"_inner.nc"
T_4p5s = ncread(fnc,"T_4p5")
T_0p5s = ncread(fnc,"T_0p5")
Tsurfs = ncread(fnc,"T_surf")
T_surf_0 = ncread(fnc,"T_surf_0")[1]
P = ncread(fnc,"P")[1]
ts = ncread(fnc,"t-hat")

# set figure directory 
figdir = "sfigs/"
mkpath(figdir)


# plot at end of simulation

filt = ts .>= max(0,ts[end]-20)


# start from where Tsurf is decreasing 
tfilt = ts[filt]

Tfilt = Tsurfs[filt]

filt_Tdecreasing = Tfilt .<= T_surf_0
tstartrevise = tfilt[filt_Tdecreasing][1]

filt2 = (ts .>= tstartrevise) .&& (ts .<= (tstartrevise + 3*P))


tfilt = ts[filt2]
tfilt = tfilt .- tfilt[1] 

Tbright_4p5 = T_4p5s[filt2]
Tbright_0p5 = T_0p5s[filt2]




lw = 3

fig = Figure(size=(400,600))

# phase diagram 
Tbrightlabel_0p5 = rich(rich("T",font=:italic),subscript("b,0.5")," [K]")
Tbrightlabel_4p5 = rich(rich("T",font=:italic),subscript("b,4.5")," [K]")
ax1 = Axis(fig[1,1],xlabel=Tbrightlabel_4p5,ylabel=Tbrightlabel_0p5,title="A",titlealign=:left)
lines!(ax1,Tbright_4p5,Tbright_0p5,color=:black,linewidth=lw)

# plot brightness temperatures vs time
c_4p5,c_0p5 = Makie.wong_colors()[1:2] 
α = 0.85

timelabel = rich(rich("t / d",font=:italic)," [-]")
axT0p5 = Axis(fig[2,1],xlabel=timelabel,ylabel=Tbrightlabel_0p5,yticklabelcolor=c_0p5,ylabelcolor=c_0p5,title="B",titlealign=:left) #,ytickcolor=c_0p5,ygridcolor=c_0p5)
axT4p5 = Axis(fig[2,1],ylabel=Tbrightlabel_4p5,yaxisposition=:right,yticklabelcolor=c_4p5,ylabelcolor=c_4p5) #,ytickcolor=c_4p5,ygridcolor=c_4p5)
hidexdecorations!(axT4p5)
# set xlims the same 
if tfilt[1]!=tfilt[end]
    xlims!(axT0p5,0,P)
    xlims!(axT4p5,0,P)
end


# set up gray shading for out of phase oscillation 
Tb0p5min = minimum(Tbright_0p5)
filtTb05min = abs.((Tbright_0p5.-Tb0p5min)./Tb0p5min) .< 1e-3
tfilt_Tb0p5min = tfilt[filtTb05min][1]

Tb4p5max = maximum(Tbright_4p5)
filtTb4p5max = abs.((Tbright_4p5.-Tb4p5max)./Tb4p5max) .< 1e-3
tfilt_Tb4p5max = tfilt[filtTb4p5max][1]

Tb0p5_at_T4p5max = Tbright_0p5[filtTb4p5max][1]

vspan!(axT4p5,tfilt_Tb0p5min,tfilt_Tb4p5max,color=(:gray,0.3))

# plot time series 
lines!(axT4p5,tfilt,Tbright_4p5,linewidth=lw,color=(c_4p5,α))
lines!(axT0p5,tfilt,Tbright_0p5,linewidth=lw,color=(c_0p5,α))

# adjust row gap
rowgap!(fig.layout, 8) 

save(figdir*"sfigure1.pdf",fig)