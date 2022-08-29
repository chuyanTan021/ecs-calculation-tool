# Main func for calculating the ECS for GCM(s) store in the NWSC and my scratch space (/glade/scratch/chuyan/CMIP6data):

# How many functionalities we may need to calc the regression of R to tas (/ plotting it)?

# Step1: Find the required monthly data on NWSC, read the .nc file as a dictionary;
# e.g., piControl = {'rsdt', 'rsut', 'rlut', 'tas'}; abr4x = {..}
# Step2: Calculate the annually mean and global (area-) mean of variables, store in another dict;
# Step3: R = rsdt - (rsut + rlut), tas = tas;
# Step4: Regress the R on tas, plotting;
# Step5: Save the result into a .npz file and save the regression plot.

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import sys

import glob
import pandas
import xarray
from copy import deepcopy
import statsmodels.formula.api as smf
from sklearn import linear_model
# below are self-defined functions

from read_hs_file import *
from get_ECSCMIP5data import * 
from get_ECSCMIP6data import *
from area_mean import *
from useful_func_cy import *


def annually_mean(data, times, label = 'mon'):
    # This function is for converting finer time scale data to annually mean data;
    # ..currently can only process: monthly data;
    # The default data shape is: (time, lat, lon).
    
    if label == 'mon':
        shape_yr = np.asarray(times).shape[0]// 12
        annually_array = np.zeros((shape_yr, np.asarray(data).shape[1], np.asarray(data).shape[2]))
        
        # Is the first month of data be 'January'? 
        if times[0, 1]== 1.0:  # start at January
            for i in range(shape_yr):
                annually_array[i,:,:] = np.nanmean(data[i*12:(i+1)*12, :,:], axis = 0)
        elif any(times[0,1]== x for x in [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]):  # not start at Jan
            for i in range(shape_yr):
                annually_array[i,:,:] = np.nanmean(data[i*12+(13-int(times[0,1])):(i+1)*12+(13-int(times[0,1])),:,:], axis = 0)
        else:
            print('wrong month value.')
    
    return annually_array


def area_mean(data, lats, lons):
    # This function is for processing the latitudinal weighting of 3-D array.
    '''..3D (time, lat, lon) would reduce to 1D (time)..
    '''
    Time_series = np.zeros(data.shape[0])
    for i in np.arange(data.shape[0]):
        
        S_time_step = data[i,:,:]
        # remove the NaN value within the 2-D array and squeeze to 1-D (indexing):
        ind1 = np.isnan(S_time_step)==False
        # weighted by cosine(lat):
        xlon, ylat = np.meshgrid(lons, lats)

        weighted_metrix1 = np.cos(np.deg2rad(ylat))  # lat matrix
        toc1 = np.sum(weighted_metrix1[ind1])   # total of cos(lat matrix) for the defined region of lat X lon

        S_weighted = S_time_step[ind1] * weighted_metrix1[ind1] / toc1
        
        Time_series[i] = np.sum(S_weighted)

    return Time_series


def PL_ECS(x, y, modn, pth, k=0, b=0):
    # Plotting function.
    # x, y are the time series of the X- and Y- axes, modn is the model's name, pth is path for saving the figure.
    # k and b is the coefficients of linear fit of 150 (abr4x) points, y_fit = k * x_fit + b;
    
    fig5, ax5 = plt.subplots(1, 1, figsize=(12, 9))  #(16.2, 9.3))
    
    parameters = {'axes.labelsize': 23, 'legend.fontsize': 18, 'axes.titlesize': 22, 'xtick.labelsize': 24, 'ytick.labelsize': 24}
    plt.rcParams.update(parameters)
    
    # plotting:
    plt.plot(x, y, 'b+', linewidth = 2.5, label = "regressing over every year of 150yrs abrupt-4xCO2 simulation")
    
    # add a fitting line:
    if all([k!=0, b!=0]):
        regr_plot = linear_model.LinearRegression()  # use sklearn calculating the coef and intecept for comparison.
        result_fit = regr_plot.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
        k2 = float(result_fit.coef_)
        b2 = float(result_fit.intercept_)
    
    xfit = np.linspace(0., 12.5, 100)
    yfit = k2 * xfit + b2
    print("/nPlotting : ")
    print("k, b from statsmodels are: ", k, b)
    print("k, b from sklearn are: ", k2, b2)
    
    plt.plot(xfit, yfit, 'k', linewidth = 2.4, linestyle = '-', label = 'fitting line for whole 150 points ')
    
    # plotting setting:s
    plt.xlim([0.00, 12.50])
    plt.ylim([0.00, 8.54])
    plt.ylabel('Change in TOA net downwelling radiative flux '+ r'$(W m^{-2})$')
    plt.xlabel('Change in surface air temperature '+ r'$(K)$')
    
    plt.title(" Radiation flux anomalies over GMT: "+ modn)
    plt.savefig(pth+modn+"_ECS calculation.jpg", dpi = 100)
    
    return None


def calc_ECS_metrics(**model_data):
    # This function is for step 1, 2, 3, 4, .
    
    # get the variable data (step 1):
    if model_data['cmip'] == 'cmip6':

        inputVar_pi, inputVar_abr = get_ECSCMIP6(**model_data)
        
    elif model_data['cmip'] == 'cmip5':
        
        inputVar_pi, inputVar_abr = get_ECSCMIP5(**model_data)
    else:
        print('not cmip6 & cmip5 data.')
    
    # calc the annually mean and global (area-) mean of all variables (step 2):
    anu_mean_pi = {}
    anu_mean_abr = {}
    anu_area_mean_pi = {}
    anu_area_mean_abr = {}
    variable_nas = ['rlut', 'rsdt', 'rsut', 'tas']
    
    # lat, lon, and time info
    lons = np.asarray(inputVar_pi['lon'][:])
    lats = np.asarray(inputVar_pi['lat'][:])

    times_abr = np.asarray(inputVar_abr['times'] * 1.)
    times_pi = np.asarray(inputVar_pi['times'] * 1.)
    
    # calculate the annually mean
    for i in range(len(variable_nas)):
        anu_mean_pi[variable_nas[i]] = annually_mean(inputVar_pi[variable_nas[i]], times_pi, label='mon')
        anu_mean_abr[variable_nas[i]] = annually_mean(inputVar_abr[variable_nas[i]], times_abr, label='mon')
    
    # calculate the area-mean (globally, weighted by the latitude)
    for i in range(len(variable_nas)):
        anu_area_mean_pi[variable_nas[i]] = area_mean(anu_mean_pi[variable_nas[i]], lats, lons)
        anu_area_mean_abr[variable_nas[i]] = area_mean(anu_mean_abr[variable_nas[i]], lats, lons)
    

    # calc R and tas (step 3):
    # R: TOA net downwelling radiative flux anomalies;
    R = (anu_area_mean_abr['rsdt'] - (anu_area_mean_abr['rsut']+anu_area_mean_abr['rlut'])) - np.mean(anu_area_mean_pi['rsdt'] - (anu_area_mean_pi['rsut']+anu_area_mean_pi['rlut']))
    # tas: glabally- and annually-mean surface air (2-m) temperature;
    tas = anu_area_mean_abr['tas'] - np.mean(anu_area_mean_pi['tas'])
    
    # do the Regression + Plotting (step 4):
    # construct a pandas DataFrame for statsmodels.formula.api to do ols regression
    data = pandas.DataFrame({'tas': tas, 'R': R})
    
    reg_model = smf.ols("R ~ tas", data).fit()
    # print the summary
    print(" R = F_4x + lambda * tas: ", ' ', reg_model.summary())

    result = {'F2x': reg_model._results.params[0]/2., 'lambda': reg_model._results.params[1], 'ECS': (-1.*reg_model._results.params[0]/reg_model._results.params[1])/2.}
    
    print(result)
    
    # Plotting
    pth_plotting = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/plot_file/ECS_calculation/'
    PL_ECS(tas, R, model_data['modn'], pth_plotting, k = reg_model._results.params[1], b = reg_model._results.params[0])  # k, b are optional arguments
    
    # Save the result (step 5):
    pth_data = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/data_file/ECS_calculation/'
    np.savez(pth_data+model_data['modn'], model = model_data, result = result)
    return None


def main():
    # This is the main function, people who use it to calc the ECS for specific 
    # gcm only need to tell the computer the name of the gcm.
    
    # CMIP6 model: # 32
    exp = 'piControl'  # Both 'piControl' and 'abrupt-4xCO2' is ok here
    AWICM11MR = {'modn': 'AWI-CM-1-1-MR', 'consort': 'AWI', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    BCCCSMCM2MR = {'modn': 'BCC-CSM2-MR', 'consort': 'BCC', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    BCCESM1 = {'modn': 'BCC-ESM1', 'consort': 'BCC', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CAMSCSM1 = {'modn': 'CAMS-CSM1-0', 'consort': 'CAMS', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CMCCCM2SR5 = {'modn': 'CMCC-CM2-SR5', 'consort': 'CMCC', 'cmip': 'cmip6', 
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2 = {'modn': 'CESM2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2FV2 = {'modn': 'CESM2-FV2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2WACCM = {'modn': 'CESM2-WACCM', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2WACCMFV2 = {'modn': 'CESM2-WACCM-FV2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}

    CNRMCM61 = {'modn': 'CNRM-CM6-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6', 
                   'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CNRMCM61HR = {'modn': 'CNRM-CM6-1-HR', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CNRMESM21 = {'modn': 'CNRM-ESM2-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6', 
                     'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CanESM5 = {'modn': 'CanESM5', 'consort': 'CCCma', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    E3SM10 = {'modn': 'E3SM-1-0', 'consort': 'E3SM-Project', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}

    ECEarth3 = {'modn': 'EC-Earth3', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    ECEarth3Veg = {'modn': 'EC-Earth3-Veg', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}

    FGOALSg3 = {'modn': 'FGOALS-g3', 'consort': 'CAS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE21G = {'modn': 'GISS-E2-1-G', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE21H = {'modn': 'GISS-E2-1-H', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE22G = {'modn': 'GISS-E2-2-G', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GFDLCM4 = {'modn': 'GFDL-CM4', 'consort': 'NOAA-GFDL', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
    HADGEM3 = {'modn': 'HadGEM3-GC31-LL', 'consort': 'MOHC', 'cmip': 'cmip6',
                'exper': 'piControl', 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    INM_CM48 = {'modn': 'INM-CM4-8', 'consort': 'INM', 'cmip': 'cmip6', 
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
    IPSLCM6ALR = {'modn': 'IPSL-CM6A-LR', 'consort': 'IPSL', 'cmip': 'cmip6',
                      'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    MIROCES2L = {'modn': 'MIROC-ES2L', 'consort': 'MIROC', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gn', "typevar": 'Amon'}
    MIROC6 = {'modn': 'MIROC6', 'consort': 'MIROC', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MPIESM12LR = {'modn': 'MPI-ESM1-2-LR', 'consort': 'MPI-M', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MRIESM20 = {'modn': 'MRI-ESM2-0', 'consort': 'MRI', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    NESM3 = {'modn': 'NESM3', 'consort': 'NUIST', 'cmip': 'cmip6',
                     'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    NorESM2MM = {'modn': 'NorESM2-MM', 'consort': 'NCC', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    SAM0 = {'modn': 'SAM0-UNICON', 'consort': 'SNU', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    TaiESM1 = {'modn': 'TaiESM1', 'consort': 'AS-RCEC', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    # CMIP5 model: # 14
    ACCESS10 = {'modn': 'ACCESS1-0', 'consort': 'CSIRO-BOM', 'cmip': 'cmip5',   # 2-d (145) and 3-d (146) variables have different lat shape
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    ACCESS13 = {'modn': 'ACCESS1-3', 'consort': 'CSIRO-BOM', 'cmip': 'cmip5',   # 2-d (145) and 3-d (146) variables have different lat shape
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    BNUESM = {'modn': 'BNU-ESM', 'consort': 'BNU', 'cmip': 'cmip5',
              'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    
    CCSM4 = {'modn': 'CCSM4', 'consort': 'NCAR', 'cmip': 'cmip5',
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CNRMCM5 = {'modn': 'CNRM-CM5', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CSIRO_Mk360 = {'modn': 'CSIRO-Mk3-6-0', 'consort': 'CSIRO-QCCCE', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CanESM2 = {'modn': 'CanESM2', 'consort': 'CCCma', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    FGOALSg2 = {'modn': 'FGOALS-g2', 'consort': 'LASG-CESS', 'cmip': 'cmip5',   # missing 'prw' in piControl
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    FGOALSs2 = {'modn': 'FGOALS-s2', 'consort': 'LASG-IAP', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GFDLCM3 = {'modn': 'GFDL-CM3', 'consort': 'NOAA-GFDL', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GISSE2H = {'modn': 'GISS-E2-H', 'consort': 'NASA-GISS', 'cmip': 'cmip5',
               'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GISSE2R = {'modn': 'GISS-E2-R', 'consort': 'NASA-GISS', 'cmip': 'cmip5',
               'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    IPSLCM5ALR = {'modn': 'IPSL-CM5A-LR', 'consort': 'IPSL', 'cmip': 'cmip5',
                   'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MIROC5 = {'modn': 'MIROC5', 'consort': 'MIROC', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MPIESMMR = {'modn': 'MPI-ESM-MR', 'consort': 'MPI-M', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    NorESM1M = {'modn': 'NorESM1-M', 'consort': 'NCC', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}

    deck = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM21, GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0, E3SM10, FGOALSg3, GFDLCM4, CAMSCSM1, INM_CM48, MPIESM12LR, AWICM11MR, BCCCSMCM2MR, CMCCCM2SR5, CESM2WACCMFV2, CNRMCM61, CNRMCM61HR, ECEarth3, ECEarth3Veg, GISSE22G, MIROCES2L, NESM3, NorESM2MM, TaiESM1, HADGEM3, BNUESM, CCSM4, CNRMCM5, CSIRO_Mk360, CanESM2, FGOALSg2, FGOALSs2, GFDLCM3, GISSE2H, GISSE2R, IPSLCM5ALR, MIROC5, MPIESMMR, NorESM1M]  #..current # 32 + 14 = 46

    # get the model's short name from input:
    Model_name = str(sys.argv[1])
    
    n_flag = 0
    for i in range(len(deck)):
        
        if Model_name == deck[i]['modn']:
            print(" model for calc: ", deck[i]['modn'])
            calc_ECS_metrics(**deck[i])  # pass the model's infomation to func for calculating ECS
            n_flag = n_flag +1
    else:
        if n_flag == 0:
            print(" I don't have this GCM currently, sorry.")

    
    return None

if __name__== "__main__":
    main()