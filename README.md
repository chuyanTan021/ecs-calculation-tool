# ecs-calculation-tool
A simple piece of code for calculating the Effective Climate Sensitivity (ECS or EffCS), Effective Radiative Forcing (ERF), and the radiative feedback parameter (lambda) for GCM. Gregory method (Gregory et al, 2004) was used.

For using this tool, if you are in the NCAR-Wyoming Supercomputer (NWSC), simply put all the .py files under this project to your default folder, and type: "python calc_ECS.py 'name_of_GCM' " with the GCM name you want to calculate.

Tips:

I put GCM data in my storage space at NWSC: '/glade/scratch/chuyan/CMIP6data/'. For GCM data that I do not have in this space, the code will try to search the required file at NWSC glade collections space: '/glade/collections/cmip/', which save some of my storage space.

If you want to run this code on your own computer (or other server), simply change the file loading path at file: "read_hs_file.py" at line 43 to where you put your GCM data. In this case, you can not use the 'glade/collections' as your substitude, but that would not require any modification. Sorry for the misleading i may caused in the code.

Change the path for saving the result plots and data file to your folder at 'pth_plotting' (line 170) and 'pth_data' (line 174) in "calc_ECS.py". Or they will go to my place!!!

Read the data file (.npz) in the following way, which gives you the Effective Radiative Forcing for doubling CO2, feedback parameter, EffCS, and the model information.

<img width="937" alt="Screen Shot 2022-08-23 at 5 53 55 PM" src="https://user-images.githubusercontent.com/81000501/186295476-e7f22b55-d169-42d6-a43e-7eb33dc3a623.png">
