using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel

"""
this script generates the data for figure 2 
"""

# set output file path details 
# outputs saved as 
# outdir*runname*"_inner.nc"
# outdir*runname*"_outer.nc"
outdir = "out/"
runname = "fig2"

# set model  parameters 
Π1 = 9.47 # [Pa⁻¹]
Π2 = 74.2 # [ ]
Π3 = 115. # [ ]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
β = 0.143 # [ ]
ΔTcloud = 177. # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
p_fig2 = [Π1, Π2, Π3, T_ref, p_ref, α, S₀, f, β, ΔTcloud, Tsolidus]

# run radiative balance model 
outputrun_radbal(p_fig2,outdir,runname;reltol=1e-10,abstol=1e-12)