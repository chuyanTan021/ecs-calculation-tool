# ecs-calculation-tool
A simple piece of code for calculating the Effective Climate Sensitivity (ECS or EffCS), ERF, and the radiative feedback parameter for GCM. Gregory method (Gregory et al, 2004) was used.


Tips:

I put the GCM data in my space at NWSC: '/glade/scratch/chuyan/CMIP6data/', and for some models i didn't download the data, I try to search the required data at NWSC glade collections speace: '/glade/collections/cmip/'.
If you want to run this code on your own computer (or other server), you need to change the file_loading path at file: "read_hs_file.py" at 43 line. Sorry for the inconvenience i caused.

Change the path for storing the result plot and data file at 'pth_plotting' (170 line) and 'pth_data' (174 line) in "calc_ECS.py". Or they will go to my place!!
