using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
"""
this script plots figure 3
"""

figdir = "figs/"

outdir_nolw = "out/"
sweepname_nolw= "fig3_zoomsweep_nolw"

outdir_radbal = "out/"
sweepname_radbal = "fig3_zoomsweep_radbal_logbeta"


c1σ = Makie.wong_colors()[1]
c2σ = Makie.wong_colors()[3]
c3σ = Makie.wong_colors()[2]

make_Π_Tcloudβ_3σ_figs_radbal_nolw_poly(outdir_nolw,outdir_radbal,figdir,sweepname_nolw,sweepname_radbal;
logΠ1min=max(log_min_Π1,log_max_Π1-5),logΠ1max=log_max_Π1,logΠ2min=max(log_min_Π2,log_max_Π2-5),logΠ2max=log_max_Π2,
logΠ3min=log_min_Π3,logΠ3max=min(log_max_Π3,log_min_Π3+5),βmin=-3,βmax=0.,
ΔTcloudmin=0.,ΔTcloudmax=350.,ngrid=25,xtickrotation=0.,
figsize=(600,500.),islogβ=true,αpoly=0.6,αline=1.,lw_nolw=2,cobs=[c1σ,c2σ,c3σ],clw=:black)

# rename figure from automatic name to figure 3 
mv(figdir*"obsvΠs+Tcloudβ_3σ_radbal_nolw_poly_$(sweepname_nolw)_$(sweepname_radbal).pdf",figdir*"figure3.pdf",force=true)
