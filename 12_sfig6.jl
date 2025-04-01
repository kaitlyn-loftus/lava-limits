using Pkg
Pkg.activate(".")
using Revise
if "./src/" ∉ LOAD_PATH 
    push!(LOAD_PATH,"./src/")
end
using DDEModel
using CairoMakie
using Random 
using Interpolations
using Roots 
"""
this script generates the data for figure S6 and plots it
"""

# set up function to calculate times where τSW = τSW*
function findτSWeq0long(t̂,sol,τSWeq)

    exp(sol(t̂;idxs=2)) - τSWeq
end

# run parameters 
reltol = 1e-8
abstol = 1e-10

# set how long to integrate  
t̂end = 3e3

# how long to consider for τeq crossings 
Δt̂ = 50. 


# set figure directories
figdir_main = "sfigs/"
mkpath(figdir_main)
figdir_extra = "ssfigs/sfig6/"
mkpath(figdir_extra)

# set number of βs 
nβ = 250
βs = LinRange(0.,1.5,nβ)


# set base model parameters for panel A
Π1_A = 9.47 # [Pa⁻¹]
Π2_A = 74.2 # [ ]
Π3_A = 115. # [kg s⁻³ K⁻¹]
T_ref = T_ref_SiO_0vap # [K]
p_ref = p_ref_SiO_0vap # [Pa]
α = 0.5 # [ ]
S₀ = 3.4e6 # [W m⁻²]
f = 1.0 # [ ]
β_A = 0.143 # [ ]
ΔTcloud_A = 177. # [K]
Tsolidus = 1400.0 # [K]

# combine model parameters into input expected by model
pA = [Π1_A, Π2_A, Π3_A, T_ref, p_ref, α, S₀, f, β_A, ΔTcloud_A, Tsolidus]

# set base model parameters for panel B 
Π1_B = 1.7508340590339526 # [Pa⁻¹]
Π2_B = 377.1454124774199 # [ ]
Π3_B = 28.849960362230448 # [kg s⁻³ K⁻¹]
β_B = 3.3060284539486826 # [ ]
ΔTcloud_B = 322.6134997840408 # [K]

# combine model parameters into input expected by model
pB = [Π1_B, Π2_B, Π3_B, T_ref, p_ref, α, S₀, f, β_B, ΔTcloud_B, Tsolidus]


# set plot details 
βlabel = rich(rich("β",font=:italic)," [–]")
Tlabel = rich(rich("T",font=:italic),subscript("surf "),"(",rich("τ",font=:italic),subscript("SW")," = ",rich("τ",font=:italic),subscript("SW"),superscript("∗"),") [K]")
c_lc = Makie.wong_colors()[1]
c_stable = Makie.wong_colors()[5]
c_chaos = :white 
βstable1 = 0.3072289156626506
βstable2 = 0.7590361445783131
βlcfade = 1.
βchaos = 1.1
αback = 0.8
Tsurf_min_A = 1275
Tsurf_max_A = 3000
nβgrad = 2
fontsize_theme = Theme(fontsize=20)
set_theme!(fontsize_theme)

t̂end = 1e3

# set up plot and axes 
fig = Figure(size=(600, 600),figure_padding=20)
axA = Axis(fig[1,1],xlabel=βlabel,ylabel=Tlabel,title="A",titlealign=:left)
rowsize!(fig.layout, 1, Relative(1/3))
xlims!(axA,βs[1],βs[end])
ylims!(axA,Tsurf_min_A,Tsurf_max_A)
vspan!(axA,βs[1],βs[end],color=(c_lc,αback),rasterize=10)
text!(axA,βstable1/2,Tsurf_min_A;text=rich("limit cycle",font=:bold),align=(:center,:bottom),fontsize=16)

Tsurf_min_B = 2250
Tsurf_max_B = 3000
axB = Axis(fig[2,1],xlabel=βlabel,ylabel=Tlabel,title="B",titlealign=:left)
ylims!(Tsurf_min_B,Tsurf_max_B)
xlims!(axB,βs[1],βs[end])
vspan!(axB,0,βstable1,color=(c_lc,αback),rasterize=10)
vspan!(axB,βstable1,βstable2,color=(c_stable,αback))
band!(axB,LinRange(βstable2,βchaos,nβgrad),fill(Tsurf_min_B,nβgrad),fill(Tsurf_max_B,nβgrad),
colormap=Makie.cgrad([c_lc, c_chaos],alpha=αback),color=LinRange(0,1,nβgrad),rasterize=10)
vspan!(axB,βchaos,βs[end],color=(c_chaos,αback))

Tsurf_B_label = Tsurf_min_B 
text!(axB,βstable1/2,Tsurf_B_label;text=rich("limit cycle",font=:bold),align=(:center,:bottom),fontsize=16)
text!(axB,0.5*(βstable1+βstable2),Tsurf_B_label;text=rich("steady state",font=:bold),align=(:center,:bottom),fontsize=16)
text!(axB,0.5*(βstable2+βchaos),Tsurf_B_label;text=rich("limit cycle",font=:bold),align=(:center,:bottom),fontsize=16)
text!(axB,0.5*(βchaos+1.5),Tsurf_B_label;text=rich("chaos (?)",font=:bold),align=(:center,:bottom),fontsize=16)

for ax ∈ [axA,axB]
    ax.xticksize=8
    ax.yticksize=8
end

c = :black
α = 0.1
ΔTthres = 1.
Tsurfeqs_A = zeros(nβ)
Tsurfeqs_B = zeros(nβ)
endstateflags_A = zeros(nβ)
endstateflags_B = zeros(nβ)
ms = 4

# calculate equilibrium points for no delay 
for (iβ,β) ∈ enumerate(βs)
    pA[9] = β
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = pA
    Tsurfeqs_A[iβ] = DDEModel.findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)[1]

    pB[9] = β
    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = pB
    Tsurfeqs_B[iβ] = DDEModel.findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)[1]
end
# plot Tsurfeq 
lines!(axA,βs,Tsurfeqs_A,color=Makie.wong_colors()[2],linewidth=6)
lines!(axB,βs,Tsurfeqs_B,color=Makie.wong_colors()[2],linewidth=6)

# iterate over all β values for panel B 
for (iβ,β) ∈ enumerate(βs)
    runname = "iteratebeta_chaoticdiag_$(iβ)"
    pB[9] = β
    sol = DDEModel.checksol_radbal2(pB,figdir_extra,runname;reltol=reltol,abstol=abstol,t̂end=t̂end)
    filt = sol.t .>= (sol.t[end]-Δt̂) 
    if sol.t[end] == sol.t[end-1]
        filt[(end-1):end] .= false
    else
        filt[end] = false
    end
    endTsurfs = sol[1,filt]
    endlogτSWs = sol[2,filt]
    endτSWs = exp.(endlogτSWs)

    Tsurfmin = minimum(endTsurfs)
    Tsurfmax = maximum(endTsurfs)

    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = pB

    Tsurfeq,τSWeq = DDEModel.findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)

    Rstar=DDEModel.R55cncA
    aplanet=DDEModel.a55cnce
    Tstar=DDEModel.T55cncA
    ratA=1.

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax = DDEModel.checkendsol_radbal2(sol,ΔTthres,Tsurfeq,Δt̂,β,ΔTcloud,α,Rstar,aplanet,Tstar,ratA)

    endstateflags_B[iβ] = endstateflag


    # conditions used in isstoplong with additional endstateflag and plotting details 
    if (Tsurfmin <= (Tsurfeq + ΔTthres)) && (Tsurfmax>=(Tsurfeq - ΔTthres)) # check if T solution bounding Teq 
        if (Tsurfmin >= (Tsurfeq - ΔTthres)) && (Tsurfmax <= (Tsurfeq + ΔTthres)) # check if within threshold
            scatter!(axB,[β],[Tsurfeq],color=c,markersize=ms)
        else # check what oscillations (if any) present 
            findτSWeq0 = (t̂) -> findτSWeq0long(t̂,sol,τSWeq)
            endt̂s = sol.t[filt]
            t̂start = endt̂s[1]
            t̂end_search = endt̂s[end]
            t̂whereτeq = find_zeros(findτSWeq0,t̂start,t̂end_search;no_pts=100,k=50)
            filt2 = abs.(exp.(sol.(t̂whereτeq;idxs=2)).-τSWeq) .<= 1e-10
            t̂whereτeq = t̂whereτeq[filt2]
            Twhereτeq = sol.(t̂whereτeq;idxs=1)
            Twhereτeqmax,imax = findmax(Twhereτeq)
            scatter!(axB,fill(β,length(Twhereτeq)),Twhereτeq,color=(c,α),markersize=ms)
        end
    end
end

# iterate over all β values for panel A
for (iβ,β) ∈ enumerate(βs)
    runname = "iteratebeta_chaoticdiagA_$(iβ)"
    pA[9] = β
    sol = DDEModel.checksol_radbal2(pA,figdir_extra,runname;reltol=reltol,abstol=abstol,t̂end=t̂end)
    filt = sol.t .>= (sol.t[end]-Δt̂) 
    if sol.t[end] == sol.t[end-1]
        filt[(end-1):end] .= false
    else
        filt[end] = false
    end
    endTsurfs = sol[1,filt]
    endlogτSWs = sol[2,filt]
    endτSWs = exp.(endlogτSWs)

    Tsurfmin = minimum(endTsurfs)
    Tsurfmax = maximum(endTsurfs)

    Π₁,Π₂,Π₃,T_ref,p_ref,α,S₀,f,β,ΔTcloud,Tmagma_solidus = pA

    Tsurfeq,τSWeq = DDEModel.findTτeqnum_radbal(Π₁,Π₂,T_ref,p_ref,α,S₀,f,β,ΔTcloud)

    Rstar=DDEModel.R55cncA
    aplanet=DDEModel.a55cnce
    Tstar=DDEModel.T55cncA
    ratA=1.

    endstateflag,P,Tsurfmin,Tsurfmax,TbLWmin,TbLWmax,TbSWmin,TbSWmax,τSWmin,τSWmax,Lmin,Lmax = DDEModel.checkendsol_radbal2(sol,ΔTthres,Tsurfeq,Δt̂,β,ΔTcloud,α,Rstar,aplanet,Tstar,ratA)

    endstateflags_A[iβ] = endstateflag


    # conditions used in isstoplong with additional endstateflag and plotting details 
    if (Tsurfmin <= (Tsurfeq + ΔTthres)) && (Tsurfmax>=(Tsurfeq - ΔTthres)) # check if T solution bounding Teq 
        if (Tsurfmin >= (Tsurfeq - ΔTthres)) && (Tsurfmax <= (Tsurfeq + ΔTthres)) # check if within threshold
            scatter!(axA,[β],[Tsurfeq],color=c,markersize=ms)
        else # check what oscillations (if any) present 
            findτSWeq0 = (t̂) -> findτSWeq0long(t̂,sol,τSWeq)
            endt̂s = sol.t[filt]
            t̂start = endt̂s[1]
            t̂end_search = endt̂s[end]
            t̂whereτeq = find_zeros(findτSWeq0,t̂start,t̂end_search;no_pts=25,k=50)
            filt2 = abs.(exp.(sol.(t̂whereτeq;idxs=2)).-τSWeq) .<= 1e-10
            t̂whereτeq = t̂whereτeq[filt2]
            Twhereτeq = sol.(t̂whereτeq;idxs=1)
            Twhereτeqmax,imax = findmax(Twhereτeq)
            scatter!(axA,fill(β,length(Twhereτeq)),Twhereτeq,color=(c,α),markersize=ms)
        end
    end
end

# set limits 
xlims!(axA,βs[1],βs[end])
xlims!(axB,βs[1],βs[end])

# save figure 
save(figdir_main*"sfigure6.pdf",fig)

