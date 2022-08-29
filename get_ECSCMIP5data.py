# read CMIP5 monthly data for calculating ECS:

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import glob

from read_hs_file import read_var_mod

def get_ECSCMIP5(modn='', consort='', cmip='', exper='', ensmem='', typevar=''):
    
    
    #.. abrupt4xCO2
    exper = 'abrupt4xCO2'
    

    TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem, typevar=typevar, time1=[1,1,1], time2=[3349, 12, 31])[-1]
    time1=[int(min(TEST1_time[:,0])),1,1]
    time2=[int(min(TEST1_time[:,0]))+149, 12, 31]
        
    print("retrieve time: ", time1, ':', time2)

    tas_abr, [], lat_abr,lon_abr,times_abr = read_var_mod(modn=modn, consort=consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)  # no pressure level info need

    # P_abr = read_var_mod(modn=modn, consort=consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)[0]
    # E_abr = read_var_mod(modn=modn, consort=consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)[0]
    
    rlut_abr = read_var_mod(modn=modn, consort=consort, varnm='rlut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)[0]
    rsdt_abr = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)[0]
    rsut_abr = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= time1, time2= time2)[0]
    
    # print(tas_abr.shape)
    inputVar_abr = {'rsdt': rsdt_abr, 'rsut': rsut_abr, 'rlut': rlut_abr, 'tas': tas_abr, 'lat':lat_abr, 'lon':lon_abr, 'times':times_abr}
    
    
    
    #.. piControl
    exper = 'piControl'
    
    
    TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem, typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
    timep1=[int(min(TEST2_time[:,0])),1,1]   #.. max-799
    timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750
    
    print ("retrieve time: ", timep1, ':', timep2)
    
    tas_pi, [], lat_pi,lon_pi,times_pi = read_var_mod(modn= modn, consort= consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)  # no pressure level info needed
    #..2-m air Temperature, for 'gmt'

    
    # P_pi           = read_var_mod(modn= modn, consort= consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)[0]
    #..Precipitation, Units in kg m^-2 s^-1 = mm * s^-1
    # E_pi           = read_var_mod(modn= modn, consort= consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)[0]
    #..Evaporations, Units also in kg m^-2 s^-1 = mm * s^-1
    
    rlut_pi = read_var_mod(modn=modn, consort=consort, varnm='rlut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)[0]
    rsdt_pi = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)[0]
    rsut_pi = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, read_p=False, time1= timep1, time2= timep2)[0]
    
    # print(tas_pi.shape)
    
    inputVar_pi = {'rsdt': rsdt_pi, 'rsut': rsut_pi, 'rlut': rlut_pi, 'tas': tas_pi, 'lat':lat_pi, 'lon':lon_pi, 'times': times_pi}
    
    return inputVar_pi, inputVar_abr