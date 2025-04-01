## Extreme Weather Variability on Hot Rocky Exoplanet 55 Cancri e Explained by Magma Temperature-Cloud Feedback
Code associated with "Extreme Weather Variability on Hot Rocky Exoplanet 55 Cancri e Explained by Magma Temperature-Cloud Feedback" by Loftus*, Luo*, Fan, & Kite (2025). 

\* Note, these authors contributed equally. 

This code base includes all source code and scripts to reproduce the results of Loftus, Luo, et al. (2025).

Yangcheng Luo wrote scripts ``03b_fig2_plot.py`` and ``14_fig4_plot.py``.
Kaitlyn Loftus wrote all other code.

See ``src/DDEModel.jl`` for the model that solves the coupled differential delay equations for magma surface temperature and cloud optical depth.

Scripts are numbered in presumed order of execution. Outputs are saved as CSV or NetCDF files. The NetCDF files are too large to host on GitHub and are available [here](https://doi.org/10.5281/zenodo.15122200). To run plotting scripts using pre-generated output files, save them in ``./out/`` or ``./out/SI/``.

***

### Steps to reproduce calculations & visualizations:

1. Download this repository 

2. In downloaded repository directory, start [Julia](https://julialang.org/downloads/) (v1.10.2 used originally)

3. In Julia REPL, run
```
include("00_setuppkgs.jl")
include("01_fitpTref.jl")
include("02_calcbounds.jl")
include("03a_fig2_run.jl")
include("04a_fig3_fullsweep_radbal.jl")
include("04b_fig3_zoomsweep_radbal.jl")
include("04c_fig3_fullsweep_nolw.jl")
include("04d_fig3_zoomsweep_nolw.jl")
include("04e_fig3_plot.jl")
include("05a_convergencetest_default.jl")
include("05b_convergencetest_10pcttol.jl")
include("05c_convergencetest_table.jl")
include("06a_nolag_calcbounds.jl")
include("06b_nolag_fullsweep.jl")
include("07_sfig1.jl")
include("08_sfig2.jl")
include("09_sfig3.jl")
include("10a_sfig4_fullsweep_radbal.jl")
include("10b_sfig4_zoomsweep_radbal.jl")
include("10c_sfig4_plot.jl")
include("11_sfig5.jl")
include("12_sfig6.jl")
include("13_sfig7.jl")
```
Note, you will probably need to lower ``ncpus`` in parallelized scripts 4a-d, 5a-b, 6b, and 10a-b to run them on a laptop. It is likely impractical to run scripts 4b/d and 10b on a laptop. You should see a progress bar to estimate time until completion. (Note, the first calculation on each core will take disproportionately long, so wait about 15-30 s for an accurate estimate.)

4. In downloaded repository directory, set up a new python environment and install required packages
```
python -m pip install -r requirements.txt
```

5. Run  
```
python 03b_fig2_plot.py
python 14_fig4_plot.py
```
Note, stage annotations in Figure 2 were done externally to script and will not be reproduced. The Figure 4 script generates individual panels that were assembled into one figure externally to the script.

***




Please contact [Kaitlyn Loftus](mailto:kaitlyn.loftus@columbia.edu) with questions / issues / concerns.

