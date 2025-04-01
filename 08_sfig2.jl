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
this script generates the data for SI figure 2 and plots it
"""

# set output file path directory 
outdir = "out/SI/"


# set model  parameters 
Π1_fig2 = 9.47 # [Pa⁻¹]
Π2_fig2 = 74.2 # [ ]
Π3_fig2 = 115. # [ ]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
β = 0.143 # [ ]
ΔTcloud = 177. # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
p_fig2 = [Π1_fig2, Π2_fig2, Π3_fig2, T_ref, p_ref, α, S₀, f, β, ΔTcloud, Tsolidus]


# set numerical tolerances 
reltol=1e-8
abstol=1e-10

# vary Pi_1

Π1s = 10. .^ [-0.03,0.97,1.97]
runnamebase = "sfig2_varΠ1_"
for (i1,Π1) ∈ enumerate(Π1s)
    p_run = deepcopy(p_fig2)
    p_run[1] = Π1
    runname = runnamebase*"$(i1)"
    outputrun_radbal2(p_run,outdir,runname;reltol=reltol,abstol=abstol)
end

# vary Pi_2

Π2s = 10. .^ [-0.13,0.87,1.87]
runnamebase = "sfig2_varΠ2_"
for (i2,Π2) ∈ enumerate(Π2s)
    p_run = deepcopy(p_fig2)
    p_run[2] = Π2
    runname = runnamebase*"$(i2)"
    outputrun_radbal2(p_run,outdir,runname;reltol=reltol,abstol=abstol)
end

# vary Pi_3

Π3s = 10. .^ [1.06,2.06,3.06]

runnamebase = "sfig2_varΠ3_"
for (i3,Π3) ∈ enumerate(Π3s)
    p_run = deepcopy(p_fig2)
    p_run[3] = Π3
    runname = runnamebase*"$(i3)"
    outputrun_radbal2(p_run,outdir,runname;reltol=reltol,abstol=abstol)
end

# make plot 
figdir = "sfigs/"
mkpath(figdir)

labelΠ1 = rich("log(",rich("Π", subscript("1"),font=:italic)," [Pa",superscript("-1"),"])")
labelΠ2 = rich("log(",rich("Π", subscript("2"),font=:italic)," [–])")
labelΠ3 = rich("log(",rich("Π", subscript("3"),font=:italic)," [kg s",superscript("-3")," K",superscript("-1"),"])")

fig = Figure(size=(600,400)); 
ax1 = Axis(fig[1,1],xlabel=rich(rich("T",font=:italic),subscript("b,4.5")," [K]"),ylabel=rich(rich("T",font=:italic),subscript("b,0.5")," [K]"),title="A")
ax2 = Axis(fig[1,2],xlabel=rich(rich("T",font=:italic),subscript("b,4.5")," [K]"),ylabel=rich(rich("T",font=:italic),subscript("b,0.5")," [K]"),title="B")
ax3 = Axis(fig[1,3],xlabel=rich(rich("T",font=:italic),subscript("b,4.5")," [K]"),ylabel=rich(rich("T",font=:italic),subscript("b,0.5")," [K]"),title="C")

α = 0.9

for ax ∈ [ax1,ax2,ax3]
    ylims!(ax,2500,3600)
    xlims!(ax,1000,3000)
    ax.xticklabelrotation = π/2
    ax.titlealign = :left
end
# load and plot 

cmap = Makie.to_colormap(:devon10)[1:6]
cs = cgrad(cmap,3,categorical=true)

ls = [:solid,:dash,:solid]
lw = [2,4,2]

for i1 ∈ [2,1,3]
    runname = "sfig2_varΠ1_$(i1)"
    fnc = outdir*runname*"_inner.nc"
    T_4p5s = ncread(fnc,"T_4p5")
    T_0p5s = ncread(fnc,"T_0p5")
    P = ncread(fnc,"P")
    t̂s = ncread(fnc,"t-hat")
    filt = t̂s .> (t̂s[end] .- P*3.)
    lines!(ax1,T_4p5s[filt],T_0p5s[filt],color=(cs[i1],α),linewidth=lw[i1])
end

Colorbar(fig[2, 1], colormap=cs, vertical = false,flipaxis = false,label=labelΠ1,ticks=[-0.03,0.97,1.97],limits=(-0.03-0.5,1.97+0.5))




cs = cgrad(cmap,3,categorical=true)
ls = [:solid,:solid,:dash]
lw = [2,2,4]

for i2 ∈ [3,2,1]
    runname = "sfig2_varΠ2_$(i2)"
    fnc = outdir*runname*"_inner.nc"
    T_4p5s = ncread(fnc,"T_4p5")
    T_0p5s = ncread(fnc,"T_0p5")
    P = ncread(fnc,"P")
    t̂s = ncread(fnc,"t-hat")
    filt = t̂s .> (t̂s[end] .- P*3.)
    lines!(ax2,T_4p5s[filt],T_0p5s[filt],color=(cs[i2],α),linewidth=lw[i2])
end

Colorbar(fig[2, 2], colormap=cs, vertical = false,flipaxis = false,label=labelΠ2,ticks=[-0.13,0.87,1.87],limits=(-0.13-0.5,1.87+0.5))


cs = cgrad(cmap,3,categorical=true)
ls = [:solid,:dash,:solid]
lw = [2,4,2]
for i3 ∈ [2,1,3]
    runname = "sfig2_varΠ3_$(i3)"
    fnc = outdir*runname*"_inner.nc"
    T_4p5s = ncread(fnc,"T_4p5")
    T_0p5s = ncread(fnc,"T_0p5")
    P = ncread(fnc,"P")
    t̂s = ncread(fnc,"t-hat")
    filt = t̂s .> (t̂s[end] .- P*3.)
    lines!(ax3,T_4p5s[filt],T_0p5s[filt],color=(cs[i3],α),linewidth=lw[i3])
end
Colorbar(fig[2, 3], colormap=cs, vertical = false,flipaxis = false,label=labelΠ3,ticks=[1.06,2.06,3.06],limits=(1.06-0.5,3.06+0.5))


save(figdir*"sfigure2.pdf",fig)



