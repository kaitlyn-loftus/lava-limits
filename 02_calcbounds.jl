using Pkg
Pkg.activate(".")
using CSV
using DataFrames
"""
this script calculates the bounds on bulk parameters 
see Appendix Table 2
"""

# fixed parameters 
N_A = 6.02214076e23 # [mol⁻¹] Avogadro’s number
R = 8.31446261815324 # [J mol⁻¹ K⁻¹] universal gas constant
k_B = 1.380649e-23 # [J K⁻¹] Boltzmann constant
g_55Cnce = 22. # [m s⁻²] # 55 Cnc e gravitational acceleration

# min and max of physical parameters 

Mair_min = 0.028 # [kg mol⁻¹] CO/N₂
Mair_max = 0.044 # [kg mol⁻¹] CO₂

Mv_min = 0.044 # [kg mol⁻¹] molar mass of the cloud-forming vapor (for SiO)
Mv_max = 0.044 # [kg mol⁻¹] (for SiO)

ρc_min = 2.1e3 # [kg m⁻³] density of a cloud particle (for SiO)
ρc_max = 3.6e3 # [kg m⁻³] (for Mg2SiO4)


rc_min = 0.2e-6 # [log(m)] representative radius of cloud particles
rc_max = 30e-6 # [log(m)] 

T0_min = 675. # [K] typical atm temperature where cloud sedimenting
T0_max = 3000. # [K]

cp_min = 1090. # [J kg⁻¹ K⁻¹] specific heat capacity of magma (FeO melt)
cp_max = 2360. # [J kg⁻¹ K⁻¹] (MgO melt)


ρm_min = 2180. # [kg m⁻³] density of magma at surface (rhyolitic magma))
ρm_max = 2800. # [kg m⁻³] (basaltic magma)

η_min = 4.6e-5 # [kg m⁻¹ s⁻¹] dynamic viscosity of the background atmosphere (N₂ or CO at 1500 K)
η_max = 8e-5 # [kg m⁻¹ s⁻¹] (CO₂ at 3000 K)

log_Hmix_min = 0. # [log(m)] average thickness of the mixed layer at the surface magma ocean in one cycle
log_Hmix_max = 3. # [log(m)]

log_d_min = log10(3600.) # [s] timescale for upward vapor transport from the surface to the cloud layer 
log_d_max = log10(3600*24*10.) # [s]

log_k1_min = -3. # [ ] scale factor for the increase rate of cloud opacity
log_k1_max = 0. # [ ]

log_k2_min = -1. # [ ] scale factor for the residence time of cloud particles in the atmosphere
log_k2_max = 2. # [ ]

k1_min = 10. ^ log_k1_min
k1_max = 10. ^ log_k1_max

k2_min = 10. ^ log_k2_min
k2_max = 10. ^ log_k2_max

d_min = 3600. # [s]
d_max = 3600*24*10. # [s]

Hmix_min = 10. ^ log_Hmix_min # [m]
Hmix_max = 10. ^ log_Hmix_max # [m]


Π1_min = 3 * k1_min * R * Mv_min / (2 * k_B * N_A * g_55Cnce * Mair_max * ρc_max * rc_max) # [Pa⁻¹]
Π1_max = 3 * k1_max * R * Mv_max / (2 * k_B * N_A * g_55Cnce * Mair_min * ρc_min * rc_min) # [Pa⁻¹]

Π2_min = 2. .* k2_min .* g_55Cnce .^ 2 .* Mair_min .* ρc_min .* rc_min .^ 2 .* d_min ./(9. * R .* T0_max .* η_max) # dimensionless
Π2_max = 2. .* k2_max .* g_55Cnce .^ 2 .* Mair_max .* ρc_max .* rc_max .^ 2 .* d_max ./(9. * R .* T0_min .* η_min) # dimensionless

Π3_min = cp_min .* ρm_min .* Hmix_min ./ d_max # [kg s⁻³ K⁻¹]
Π3_max = cp_max .* ρm_max .* Hmix_max ./ d_min # [kg s⁻³ K⁻¹]

println("min Π1 = $(Π1_min) Pa⁻¹")
println("max Π1 = $(Π1_max) Pa⁻¹")
println("min Π2 = $(Π2_min)")
println("max Π2 = $(Π2_max)")
println("min Π3 = $(Π3_min) kg s⁻³ K⁻¹")
println("max Π3 = $(Π3_max) kg s⁻³ K⁻¹")

# save out bulk parameter bounds 
df = DataFrame()
df[:,Symbol("min Pi1 [1/Pa]")] = [Π1_min]
df[:,Symbol("max Pi1 [1/Pa]")] = [Π1_max]
df[:,Symbol("min Pi2 [ ]")] = [Π2_min]
df[:,Symbol("max Pi2 [ ]")] = [Π2_max]
df[:,Symbol("min Pi3 [kg/s^3/K]")] = [Π3_min]
df[:,Symbol("max Pi3 [kg/s^3/K]")] = [Π3_max]

CSV.write("data/bulk_parameter_bounds.csv", df)

# also save out log of bulk parameter bounds  
df = DataFrame()
df[:,Symbol("log(min Pi1 [1/Pa])")] = [log10(Π1_min)]
df[:,Symbol("log(max Pi1 [1/Pa])")] = [log10(Π1_max)]
df[:,Symbol("log(min Pi2 [ ])")] = [log10(Π2_min)]
df[:,Symbol("log(max Pi2 [ ])")] = [log10(Π2_max)]
df[:,Symbol("log(min Pi3 [kg/s^3/K])")] = [log10(Π3_min)]
df[:,Symbol("log(max Pi3 [kg/s^3/K])")] = [log10(Π3_max)]

CSV.write("data/log_bulk_parameter_bounds.csv", df)





