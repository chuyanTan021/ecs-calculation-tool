# ecs-calculation-tool
A simple piece of code for calculating the Effective Climate Sensitivity (ECS or EffCS), ERF, and the radiative feedback parameter for GCM. Gregory method (Gregory et al, 2004) was used.


Tips:

I put the GCM data in my space at NWSC: '/glade/scratch/chuyan/CMIP6data/'. For files I don't have in the scratch space, I then try to search the required file at NWSC glade collections space: '/glade/collections/cmip/'.
If you want to run this code on your own computer (or other server), simply change the file_loading path at file: "read_hs_file.py" at line 43 to path where you put your data. In this case, you can not use the 'glade/collections' as your substitude, but you don't need to change anything. Sorry for the inconvenience i caused.

Change the path for storing the result plot and data file at 'pth_plotting' (line 170) and 'pth_data' (line 174) in "calc_ECS.py". Or they will go to my place!!
