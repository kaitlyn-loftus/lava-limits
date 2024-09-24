## Extreme Weather Variability on Hot Rocky Exoplanet 55 Cancri e Explained by Magma Temperature-Cloud Feedback
Code associated with "Extreme Weather Variability on Hot Rocky Exoplanet 55 Cancri e Explained by Magma Temperature-Cloud Feedback" by Loftus*, Luo*, Fan, & Kite (preprint). 

\* Note, these authors contributed equally. 

This code base includes all source code and scripts to reproduce the results of Loftus, Luo, et al. 

Yangcheng Luo wrote script ``03b_fig2_plot.py``.
Kaitlyn Loftus wrote all other code.

See ``src/DDEModel.jl`` for the model that solves the coupled differential delay equations for magma surface temperature and cloud optical depth.

Scripts are numbered in presumed order of execution. All outputs are saved as CSV or NetCDF files. NetCDF files associated with scripts 2a, 4a-d are too large to host on GitHub and are available [here](https://doi.org/10.5281/zenodo.13829241). To run plotting scripts (2b, 4e) using pre-generated output files, save them in ``./out/``.

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
include("05_convergencetest.jl")
```
Note, you will potentially need to lower ``ncpus`` in scripts 4a, 4c, and 5. You will almost certainly need to lower ``ncpus`` in scripts 4b and 4d. It is likely impractical to run scripts 4b and 4d on a laptop. You should see a progress bar to estimate time until completion. (Note, the first calculation on each core will take disproportionately long, so wait about 15-30 s for an accurate estimate.)

4. In downloaded repository directory, set up a new python environment and install required packages
```
pip install -r requirements.txt
```

5. Run  
```
python 03b_fig2_plot.py
```
Note, stage annotations in Figure 2 were done externally to script and will not be reproduced.

***




Please contact [Kaitlyn Loftus](mailto:kaitlyn.loftus@columbia.edu) with questions / issues / concerns.

