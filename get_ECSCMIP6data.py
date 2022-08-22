# read CMIP6 monthly data for calculating ECS:

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import glob

from read_hs_file import read_var_mod

def get_ECSCMIP6(modn='', consort='', cmip='', exper='', ensmem='', gg='', typevar=''):
    # e.g, modn='IPSL-CM5A-LR'; consort='IPSL'; cmip='cmip5'; exper='amip'; ensmem='r1i1p1'; gg=''; typevar='Amon'

    #.. abrupt4xCO2
    exper = 'abrupt-4xCO2'
    
    if modn == 'HadGEM3-GC31-LL':
        ensmem = 'r1i1p1f3'
        TEST1_time = read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,15], time2=[3349, 12, 15])[-1]
        time1=[int(min(TEST1_time[:,0])),1,15]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 15]

    elif modn == 'EC-Earth3':
        ensmem = 'r3i1p1f1'
        TEST1_time = read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[3349, 12, 31])[-1]
        time1=[int(min(TEST1_time[:,0])),1,1]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 31]

    
    else:
        
        TEST1_time = read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1],time2=[3349, 12, 31])[-1]
        time1=[int(min(TEST1_time[:,0])),1,1]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 31]

    print("retrieve time: ", time1, ':', time2)
    
    tas_abr, [], lat_abr, lon_abr, times_abr = read_var_mod(modn=modn, consort=consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)  # no pressure level info needed
    #. 2-m air Temperature, as 'GMT'

    # P_abr           = read_var_mod(modn=modn, consort=consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    # E_abr           = read_var_mod(modn=modn, consort=consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    rlut_abr = read_var_mod(modn=modn, consort=consort, varnm='rlut', cmip=cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2 = time2)[0]
    rsdt_abr = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    rsut_abr = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    # rsutcs_abr = read_var_mod(modn=modn, consort=consort, varnm='rsutcs', cmip= cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1 = time1, time2= time2)[0]

    # save 'abrupt-4xCO2' data into dictionary:
    inputVar_abr = {'rsdt': rsdt_abr, 'rsut': rsut_abr, 'rlut': rlut_abr, 'tas': tas_abr, 'lat':lat_abr, 'lon':lon_abr, 'times':times_abr}
    
    
    
    #.. piControl
    exper = 'piControl'
    
    if modn == 'HadGEM3-GC31-LL':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,15], time2=[8000,12,15])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1,15]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 15]  #.. max-750

    elif modn == 'EC-Earth3':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

    elif modn == 'NESM3':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ta',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar, read_p= True, time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750
    
    elif modn == 'CNRM-CM6-1':
        ensmem = 'r1i1p1f2'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='evspsbl',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

    else:
        
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])),1,1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

    print ("retrieve time: ", timep1, ':', timep2)

    tas_pi, [], lat_pi, lon_pi, times_pi = read_var_mod(modn= modn, consort= consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)  # no pressure level info needed
    #. 2-m air Temperature, as 'GMT'

    # P_pi           = read_var_mod(modn= modn, consort= consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    # E_pi           = read_var_mod(modn= modn, consort= consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    
    rlut_pi = read_var_mod(modn=modn, consort=consort, varnm='rlut', cmip=cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2 =timep2)[0]
    rsdt_pi = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    rsut_pi  = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    # rsutcs_pi = read_var_mod(modn=modn, consort=consort, varnm='rsutcs', cmip= cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1 = timep1, time2= timep2)[0]

    # save 'piControl' data into dictionary:
    inputVar_pi = {'rsdt': rsdt_pi, 'rsut': rsut_pi, 'rlut': rlut_pi, 'tas': tas_pi, 'lat':lat_pi, 'lon':lon_pi, 'times': times_pi}

    return inputVar_pi, inputVar_abr
