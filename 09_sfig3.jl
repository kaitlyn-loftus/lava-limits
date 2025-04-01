using Pkg 
Pkg.activate(".")
using Dates
using CairoMakie

"""
this script plots JWST/NIRCam brightness temperatures from Patel et al. (2024) Table 1
[doi:10.1051/0004-6361/202450748]
for SI figure S3
"""

figdir = "sfigs/"
mkpath(figdir)


Ts = [873,1716,2078,2256,2016]
Ts_high = [167,230,172,330,137]
Ts_low = [187,315,342,188,179]

Ts_2p1 = [2417,1247,2234,2302,3138]
Ts_2p1_high = [335,190,86,413,107]
Ts_2p1_low = [287,245,88,807,107]


fig = Figure(size=(500,400),fontsize=20); 
ax = Axis(fig[1,1],ylabel="2.1 μm brightness temperature [K]",xlabel="4.5 μm brightness temperature [K]")
lines!(ax,Ts[1:4],Ts_2p1[1:4],color=:gray,linewidth=1,linestyle=:dash)
scatter!(ax,Ts,Ts_2p1,color=:black,markersize=25)
errorbars!(ax,Ts,Ts_2p1,Ts_low,Ts_high,color=:black,linewidth=3,direction=:x)
errorbars!(ax,Ts,Ts_2p1,Ts_2p1_low,Ts_2p1_high,color=:black,linewidth=3)
for i ∈ 1:5
    text!(Ts[i],Ts_2p1[i];text="$i",align=(:center,:center),color=:white)
end

save(figdir*"sfigure3.pdf",fig)