"""
this script calculates the bounds on bulk parameters for ODE 
see Table S1 and SI text 
"""

# fixed parameters 
N_A = 6.02214076e23 # [mol⁻¹] Avogadro’s number
R = 8.31446261815324 # [J mol⁻¹ K⁻¹] universal gas constant
k_B = 1.380649e-23 # [J K⁻¹] Boltzmann constant
g_55Cnce = 22. # [m s⁻²] # 55 Cnc e gravitational acceleration
τascent = 60. * 60. # [s] # ascent time scale 

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

log_k1_min = -3. # [ ] scale factor for the increase rate of cloud opacity
log_k1_max = 0. # [ ]

log_k2_min = -1. # [ ] scale factor for the residence time of cloud particles in the atmosphere
log_k2_max = 2. # [ ]

k1_min = 10. ^ log_k1_min
k1_max = 10. ^ log_k1_max

k2_min = 10. ^ log_k2_min
k2_max = 10. ^ log_k2_max

Hmix_min = 10. ^ log_Hmix_min # [m]
Hmix_max = 10. ^ log_Hmix_max # [m]


Π1_min = 3 * k1_min * R * Mv_min / (2 * k_B * N_A * g_55Cnce * Mair_max * ρc_max * rc_max * τascent) # [Pa⁻¹]
Π1_max = 3 * k1_max * R * Mv_max / (2 * k_B * N_A * g_55Cnce * Mair_min * ρc_min * rc_min * τascent) # [Pa⁻¹]

Π2_min = 2. .* k2_min .* g_55Cnce .^ 2 .* Mair_min .* ρc_min .* rc_min .^ 2 ./(9. * R .* T0_max .* η_max) # dimensionless
Π2_max = 2. .* k2_max .* g_55Cnce .^ 2 .* Mair_max .* ρc_max .* rc_max .^ 2 ./(9. * R .* T0_min .* η_min) # dimensionless

Π3_min = cp_min .* ρm_min .* Hmix_min # [kg s⁻³ K⁻¹]
Π3_max = cp_max .* ρm_max .* Hmix_max # [kg s⁻³ K⁻¹]

println("min Π1 = $(Π1_min) Pa⁻¹ s⁻¹")
println("max Π1 = $(Π1_max) Pa⁻¹ s⁻¹")
println("min Π2 = $(Π2_min) s⁻¹")
println("max Π2 = $(Π2_max) s⁻¹")
println("min Π3 = $(Π3_min) kg s⁻² K⁻¹")
println("max Π3 = $(Π3_max) kg s⁻² K⁻¹")

@show log10(Π1_min)
@show log10(Π1_max)
@show log10(Π2_min)
@show log10(Π2_max)
@show log10(Π3_min)
@show log10(Π3_max)





