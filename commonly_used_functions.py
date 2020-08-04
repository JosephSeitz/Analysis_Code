# -*- coding: utf-8 -*-
"""
Commonly Used Functions

Created by: Joey Seitz

These are a few of the functions that I have used in multiple projects and 
I have put them in here to be loaded in to save room in my wokring projects.
"""
##############################################################################
def burn_dict():
    """
    This fuction returns the dictionaty of the visually inspected pre-burn 
    periods of each burn ("burn"[0]) and the index where the first sonic hits
    a temperature of +5sigma ("burn"[1]).

    Returns
    -------
    dct : The dictionary containing the index of pre-fire and first sonic to 
          to reach a +sigma temperature.
    """
    
    dct = {'01': [21000 ,22289], '02': [14000, 14972], '03': [20000, 20296], \
           '04': [30000, 34979], '05': [15000, 15974], '06': [6000, 6544], \
           '07': [25000, 27822], '08': [18000, 19810], '09': [20000, 25047], \
           '11': [30000,30681],  '12': [60000, 62937], '13': [50000, 55131], \
           '18': [185000, 192322], '19': [45000, 48831], '20': [24000, 25860],\
           '21': [25000, 27822], '22': [27000, 29484], '23': [45000, 48846], \
           '24': [20000, 23015], '25': [25000, 26797], '26': [25000, 26909], \
           '27': [35000, 36916], '28': [27000, 28673], '29': [25000, 28166], \
           '30': [12000, 13954], '31': [30000, 31046], '32': [25000, 26294], \
           '33': [20000, 23293], '34': [40000,43702]}
    
    return dct

##############################################################################    
def text_finder(path, file_type = ".txt"):
    """
    This function takes a path and looks for the desired file type and returns
    a list of the files loacted in that directory.

    Parameters
    ----------
    path : path of the desired directory
    
    file_type: type of files wanted, default is a .txt files

    Returns
    -------
    txt_files : Alphabetized list of files in said directory

    """
    import os   #used for directory work
    txt_files=[]
    all_files = os.listdir(path) #tells what directory the files are in
    for i in all_files: 
        if i[-4:] == str(file_type):
            txt_files.append(i)
    txt_files.sort() #alphabetizes them
    
    return txt_files

##############################################################################
def file_to_df(path, fillnans = False):
    import numpy as np
    import pandas as pd
    """
    This function takes a path of the file and loads it in as a pandas 
    DataFrame

    Parameters
    ----------
    path : str
           Path of desired file 
    
    fillnans : Default is False, it will leave the NaN values in the DF
                IF True: it will replace the NaNs with np.nan

    Returns
    -------
    df : Dataframe of desired file
    """
    
    df= pd.read_csv(path,na_values = ['NAN', "00nan", "NaN"], sep=" ")
    if fillnans ==True:
        df.fillna(value=np.nan, inplace=True)
        
    return df

##############################################################################
def fire_start(df, n, sig = 5):
    """
    This function takes a burn dataframe and finds the index where the desired 
    temperature sigma value is reached 

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame of file with a column of tempurature "T".
    n : int
        The pre-defined pre-burn averaging period. Used to narrow the index 
        where the fire is after the index.
    sig : float, optional
        The sig value defining the fire period. The default is 5.

    Returns
    -------
    x_fire : int
        The index of the fire start, defined by the given sigma value.
        If DF doesn't reach desired sigma value, returns 9e10

    """
    import numpy as np
    ### Finding the start of fire
    x_fire, fire_sig = 9e10, np.nanmean(df["T"][:n]) + sig*np.std(df["T"][:n]) 
    start = n-9000
    if n-9000< 0:
        start = 0
        
    for i in range(start, len(df["T"])):
        if df["T"][i] > fire_sig:
            x_fire = i
            break
    
    return x_fire

##############################################################################
def lin_rb(freq, power, n_bins, new_n_bins):
    """
    Linear Redistribution function 

    Parameters
    ----------
    freq : list
        List of spectra frequencies (x_values from spectra)
    power : list
        List of the power values corresponding to the spectral frequencies 
        (y_values from spectra).
    n_bins : int
        Original length of the value list before spectra.
    new_n_bins : int
        The desired length of the rebined list.

    Returns
    -------
    new_freq : list
         The list of the new frequencies of the spectra (x_values).
    new_df : float
        DESCRIPTION.
    lin_rb_psd : list
        The list of the new power of the spectra (y_values).

    """
    import scipy as sc
    import numpy as np
    
    lin_rb_psd, f_bin_edges, something = sc.stats.binned_statistic(freq, \
                                power, statistic='mean',bins=new_n_bins)

    new_df = np.median(np.diff(f_bin_edges))
    new_freq = f_bin_edges[0:-1] + 0.5 * new_df  # so that the freq is mid-bin

    return new_freq, new_df, lin_rb_psd

##############################################################################
def fast_fourier_spectra(var_lst,binn = 100, T= 0.1, redist = True):
    """
    This function creates a fast fourier spectrum of desired list

    Parameters
    ----------
    var_lst : list
        List of the variable wanted to do a spectral analysis.
    binn : int, optional
        The wanted length of the binned spectral list, averages the spectral 
        anaylsis and data is 'smoothed'. The default is 100.
    T : float, optional
        The period of data (.1 = 10hz). The default is 0.1.
    redist : bool, optional
        If True, the data will be rebinned and if false the data will be the 
        raw spectral output. The default is True.

    Returns
    -------
    
     new_x: list
         If the length of the array is less than the rebinned number (default 
         is 100), then a blank list is returned.If it is longer than the rebinn
         the frequency of the spectra binned or unbinned is returned.
    powery: list
        The list of the power from the spectra. If the length of the input is 
        less than the rebin number, a blank list is returned.If it is longer 
        than the rebinn the power (y-values) of the spectra binned or unbinned
        is returned.
         

    """
    from scipy.fft import fft
    import numpy as np
    
    N = len(var_lst)
    if N < binn:
        return [np.nan],[np.nan]
    
    # Number of sample points
    N = len(var_lst)
    
    # sample spacing
    yf = fft(np.array(var_lst))
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yfft =  np.abs(yf[0:N//2])
    
    power = np.multiply(yfft, np.conj(yfft))
    if redist == False:
        return xf, power
    
    if redist == True:
        new_x, df_stat, powery = lin_rb(xf,power,N,binn)
    
    return  new_x, powery
