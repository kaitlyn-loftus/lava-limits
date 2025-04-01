using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
"""
this script plots SI figure 4
"""

figdir = "sfigs/"

outdir_default = "out/"
sweepname_default= "fig3_zoomsweep_radbal_logbeta"

outdir_sensitivity = "out/SI/"
sweepname_sensitivity = "sfig2_zoomsweep_radbal_logbeta"


c1σ = Makie.wong_colors()[1]
c2σ = Makie.wong_colors()[3]
c3σ = Makie.wong_colors()[2]
cdef = Makie.wong_colors()[4]

make_Π_Tcloudβ_pTrefα_3σ_figs_radbal_nolw_poly(outdir_default,outdir_sensitivity,figdir,sweepname_default,sweepname_sensitivity;
logΠ1min=max(log_min_Π1,log_max_Π1-8),logΠ1max=log_max_Π1,logΠ2min=max(log_min_Π2,log_max_Π2-5),logΠ2max=log_max_Π2,
logΠ3min=log_min_Π3,logΠ3max=min(log_max_Π3,log_min_Π3+5),βmin=-3,βmax=log10(4),
ΔTcloudmin=0.,ΔTcloudmax=350.,ngrid=30,xtickrotation=π/2,
figsize=(800,700.),islogβ=true,αpoly=0.6,αline=1.,lw_nolw=2,cobs=[c1σ,c2σ,c3σ],clw=:black,
prefmin=10,prefmax=14,Trefmin=5,Trefmax=8,αmin=0.2,αmax=1.,cdef=cdef,lw_def=3)

# rename figure from automatic name to figure s4 
mv(figdir*"obsvΠs+Tcloudβ+pTrefα_3σ_radbal_nolw_poly_$(sweepname_default)_$(sweepname_sensitivity).pdf",figdir*"sfigure4.pdf",force=true)
