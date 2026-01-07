import numpy as np
import scipy.constants as scip
from scipy.optimize import curve_fit
import h5py
import os
import math
import scipy.signal as sps
from scipy.signal import savgol_filter
import scipy.interpolate as interpol
from scipy.signal import find_peaks

### extra imports so new code for sweeps can work with VNA
#from Instr_drivers import \
#    vna as vna_, yokogawa
#import numpy as np
import datetime
import sys
#from Helper_functions import save_script, get_temperature, vna_tools, plotting_tools
#import os
import matplotlib.pyplot as plt
import time


def s21_bas_2(f,A0,B0,t0,phi):
    indx = int(len(f)/2)
    f1 = f[:indx]
    S21 = (A0+2*np.pi*f1*B0)*np.exp(1j*2*np.pi*f1*t0+1j*phi)
    return np.append(S21.real,S21.imag)

def s21_bas_eval_3(f,popt):
    A, B, C, D, t0, phi = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    S21 = (A+B*2*np.pi*f+C*(2*np.pi*f)**2+D*(2*np.pi*f)**3)*np.exp(1j*2*np.pi*f*t0+1j*phi)
    return S21

def s21_res_notch(f,fr,Ql,Qc_mod,phi):
    ind = int(len(f)/2)
    f1 = f[:ind]
    S21 = 1-(Ql*np.exp(1j*phi)/Qc_mod)/(1+2*1j*Ql*(f1/fr-1))
    return np.append(S21.real, S21.imag)

def s21_res_notch_eval(f,popt):
    fr,Ql,Qc_mod,phi = popt[0],popt[1],popt[2],popt[3]
    S21 = 1-(Ql*np.exp(1j*phi)/Qc_mod)/(1+2*1j*Ql*(f/fr-1))
    return S21


def s21_bas_eval_2(f, popt):

    A0 = popt[0]
    B0 = popt[1]
    t0 = popt[2]
    phi = popt[3]
    S21 = (A0+2*np.pi*f*B0)*np.exp(1j*2*np.pi*f*t0+1j*phi)
    return S21

def s21_renorm(freqs_data, complex_data, popt, m):
    if m == 2:
        S21_bas_fit = s21_bas_eval_2(np.asarray(freqs_data), popt)
    elif m == 3:
        S21_bas_fit = s21_bas_eval_3(np.asarray(freqs_data), popt)
    S21_res = []
    for ind, ii in enumerate(freqs_data):
        S21_res.append(complex_data[ind] / S21_bas_fit[ind])
    S21_res = np.asarray(S21_res)
    return freqs_data, S21_res

def s21_res_fit(freqs_data, S21_res):
    fr_ind = np.argmin(20 * np.log10(np.abs(S21_res)))
    N = len(freqs_data)
    fr_guess = freqs_data[fr_ind]
    FWHM = ((np.mean(np.abs(S21_res[:int(N/20)])) + np.mean(np.abs(S21_res[-int(N/20):]))) * 0.5 + np.min(np.abs(S21_res))) * 0.5

    ind_left = np.argmin(np.abs(np.abs(S21_res[:fr_ind]) - FWHM))
    ind_right = np.argmin(np.abs(np.abs(S21_res[fr_ind:]) - FWHM))
    Ql_guess = fr_guess / (freqs_data[fr_ind + ind_right] - freqs_data[ind_left])

    xc = 0.5 * (S21_res[np.argmax(S21_res.real)].real + S21_res[np.argmin(S21_res.real)].real)
    yc = 0.5 * (S21_res[np.argmax(S21_res.imag)].imag + S21_res[np.argmin(S21_res.imag)].imag)
    r0 = 0.25 * (S21_res[np.argmax(S21_res.real)].real - S21_res[np.argmin(S21_res.real)].real + S21_res[
        np.argmax(S21_res.imag)].imag - S21_res[np.argmin(S21_res.imag)].imag)


    if xc <= 1:
        phi_guess = -np.arcsin(yc / r0)
    else:
        if yc > 0:
            phi_guess = -np.pi/2 - np.arccos(yc/r0)
        else:
            phi_guess = np.pi / 2 + np.arccos(abs(yc) / r0)

    #phi_guess = -np.arcsin(yc / r0)
    Qc_mod_guess = Ql_guess / 2 / r0
    #### Fit the data
    try:
        popt, pcov = curve_fit(s21_res_notch, np.append(freqs_data, freqs_data), np.append(S21_res.real, S21_res.imag),
                            p0=[fr_guess, Ql_guess, Qc_mod_guess, phi_guess])
        S21_result = s21_res_notch_eval(freqs_data, popt)
        Qc_complex = popt[2] * np.exp(-1j * popt[3])

        Qi = 1 / (1 / popt[1] - (1 / Qc_complex).real)
        Qc = popt[2]/np.cos(popt[3])

        fr_err = np.sqrt(np.diag(pcov))[0]
        Ql_err = np.sqrt(np.diag(pcov))[1]
        Qc_mod_err = np.sqrt(np.diag(pcov))[2]
        phi_err = np.sqrt(np.diag(pcov))[3]
        Qc_err = (abs(Qc_mod_err*np.cos(popt[3]))+abs(popt[2]*np.sin(popt[3])*phi_err))/np.cos(popt[3])**2
        Qi_err = (abs(Ql_err*popt[2]**2)+abs(Ql_err*popt[2]*popt[1]*np.cos(popt[3]))+abs(np.cos(popt[3])*popt[1]**2*Qc_mod_err)+ \
                abs(popt[1]**2*popt[2]*np.sin(popt[3])*phi_err)+abs(popt[1]*popt[2]*np.cos(popt[3])*Ql_err))/(popt[2]-np.cos(popt[3]*popt[1]))**2
        flag = 1
        ## Goodness of the fit R2 real number: when the fit is perfect it should be 1.
        SS_res, SS_tot, S_mean = 0, 0, np.mean(S21_res)
        for ff in range(len(freqs_data)):
            SS_res = SS_res + abs(S21_res[ff] - S21_result[ff])**2
            SS_tot = SS_tot + abs(S21_res[ff] - S_mean)**2
        R2 = 1 - SS_res/SS_tot
    except: popt, Qi, Qc, S21_result, fr_err, Qi_err, Qc_err, flag, R2 = 0, 0, 0, 0, 0, 0, 0, 0, 0

    return popt, Qi, Qc, S21_result, fr_err, Qi_err, Qc_err, flag, R2


def s21_res_fit_raw(freqs_data, complex_data, popt_2, popt):
    A0_guess = popt_2[0]
    B0_guess = popt_2[1]
    t0_guess = popt_2[2]
    phi_guess = popt_2[3]
    fr_guess = popt[0]
    Ql_guess = popt[1]
    Qc_mod_guess = popt[2]
    phi0_guess = popt[3]
    #### Fit the data
    try:
        popt_val_raw, pcov_val_raw = curve_fit(s21_res_notch_raw, np.append(freqs_data, freqs_data),\
            np.append(complex_data.real, complex_data.imag),\
            p0=[A0_guess, B0_guess, t0_guess, phi_guess, fr_guess, Ql_guess, Qc_mod_guess, phi0_guess])
        S21_result_val_raw = s21_res_notch_eval_raw(freqs_data, popt_val_raw[:4], popt_val_raw[4:])
        A0_val_raw = popt_val_raw[0]
        B0_val_raw = popt_val_raw[1]
        t0_val_raw = popt_val_raw[2]
        phi_val_raw = popt_val_raw[3]
        fr_val_raw = popt_val_raw[4]
        Ql_val_raw = popt_val_raw[5]
        absQc_val_raw = popt_val_raw[6]
        phi0_val_raw = popt_val_raw[7]
        Qc_complex = absQc_val_raw * np.exp(-1j * phi0_val_raw)
        Qi_val_raw = 1 / (1 / Ql_val_raw - (1 / Qc_complex).real)

        fr_err_raw = np.sqrt(np.diag(pcov_val_raw))[4]
        Ql_err_raw = np.sqrt(np.diag(pcov_val_raw))[5]
        Qc_mod_err_raw = np.sqrt(np.diag(pcov_val_raw))[6]
        phi_err_raw = np.sqrt(np.diag(pcov_val_raw))[7]
        Qc_err_raw = (abs(Qc_mod_err_raw*np.cos(popt_val_raw[7]))+abs(popt_val_raw[6]*np.sin(popt_val_raw[7])*phi_err_raw))/np.cos(popt_val_raw[7])**2
        Qi_err_raw = (abs(Ql_err_raw*popt_val_raw[6]**2)+abs(Ql_err_raw*popt_val_raw[6]*popt_val_raw[5]*np.cos(popt_val_raw[7]))+\
                  abs(np.cos(popt_val_raw[7])*popt_val_raw[5]**2*Qc_mod_err_raw)+\
                abs(popt_val_raw[5]**2*popt_val_raw[6]*np.sin(popt_val_raw[7])*phi_err_raw)+\
                  abs(popt_val_raw[5]*popt_val_raw[6]*np.cos(popt_val_raw[7])*Ql_err_raw))/(popt_val_raw[6]-np.cos(popt_val_raw[7]*popt_val_raw[5]))**2
        flag = 1
        ## Goodness of the fit R2 real number: when the fit is perfect it should be 1.
        SS_res, SS_tot, S_mean = 0, 0, np.mean(complex_data)
        for ff in range(len(freqs_data)):
            SS_res = SS_res + abs(complex_data[ff] - S21_result_val_raw[ff])**2
            SS_tot = SS_tot + abs(complex_data[ff] - S_mean)**2
        R2_val_raw = 1 - SS_res/SS_tot
    except: fr_val_raw, Qi_val_raw, Ql_val_raw, absQc_val_raw, phi0_val_raw, A0_val_raw, B0_val_raw, t0_val_raw, phi_val_raw, S21_result_val_raw,\
        flag, fr_err_raw, Qc_err_raw, Qi_err_raw,  R2_val_raw = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0 ,0 ,0

    return fr_val_raw, Qi_val_raw, Ql_val_raw, absQc_val_raw, phi0_val_raw, A0_val_raw, B0_val_raw, t0_val_raw, phi_val_raw,\
        S21_result_val_raw, flag, fr_err_raw, Qc_err_raw, Qi_err_raw, R2_val_raw

def s21_res_notch_raw(f,A0,B0,t0,phi,fr,Ql,Qc_mod,phi0):
    ind = int(len(f)/2)
    f1 = f[:ind]
    S21 = (A0+2*np.pi*f1*B0)*np.exp(1j*2*np.pi*f1*t0+1j*phi)*(1-(Ql*np.exp(1j*phi0)/Qc_mod)/(1+2*1j*Ql*(f1/fr-1)))
    return np.append(S21.real, S21.imag)

def s21_res_notch_eval_raw(f,bas_fit_param,popt):
    A0, B0, t0, phi = bas_fit_param[0],bas_fit_param[1],bas_fit_param[2],bas_fit_param[3]
    fr,Ql,Qc_mod,phi0 = popt[0],popt[1],popt[2],popt[3]
    S21 = (A0+2*np.pi*f*B0)*np.exp(1j*2*np.pi*f*t0+1j*phi)*(1-(Ql*np.exp(1j*phi0)/Qc_mod)/(1+2*1j*Ql*(f/fr-1)))
    return S21

def av_phot_number(p_in_dB, offset, Ql, fr, Qc):
    #have offset + attenuation from function created, give it calibration data
    p_in = 10**(-3)*10**((p_in_dB+offset)/10)
    return 2*Ql**2*p_in/(scip.hbar*(2*scip.pi*fr)**2*Qc)

def data_import(filename):

    freqs = []
    S21_complex = []
    powers = []
    with h5py.File(filename, 'r') as file:
        for ind, key in enumerate(file.keys()):
            freqs.append(np.asarray(file[str(key)]['ResSpec']['raw_data']['freqs'][()]))
            S21_complex.append(np.asarray(file[str(key)]['ResSpec']['raw_data']['complex'][()]))
            powers.append(file[str(key)]['ResSpec']['params']['Powers'][()])
    return freqs, S21_complex, powers

def merge_two_data_sets_in_same_power_range(file_name, file_name_2, new_file):

    with h5py.File(file_name, 'r') as f: #calling data from the first file
        keys_list_1 = []                       #making empty lists to append data into
        freqs_centers_list = []
        freqs_span_list = []
        complex_list = []
        freqs_list = []
        for key in f.keys():                    #appending data from files into new hdf5 file
            keys_list_1.append(key)
            #separate for ones below as they have variable data
            freqs_centers_list.append(f[key]['ResSpec']['params']['Frequency Centre'][()])
            freqs_span_list.append(f[key]['ResSpec']['params']['Frequency Span'][()])
            complex_list.append(f[key]['ResSpec']['raw_data']['complex'][()])
            freqs_list.append(f[key]['ResSpec']['raw_data']['freqs'][()])

    with h5py.File(file_name_2, 'r') as f:        #calling data from second file
        keys_list_2 = []
        freqs_centers_list_2 = []
        freqs_span_list_2 = []
        complex_list_2 = []
        freqs_list_2 = []
        for key in f.keys():
            keys_list_2.append(key)
            freqs_centers_list_2.append(f[key]['ResSpec']['params']['Frequency Centre'][()])
            freqs_span_list_2.append(f[key]['ResSpec']['params']['Frequency Span'][()])
            complex_list_2.append(f[key]['ResSpec']['raw_data']['complex'][()])
            freqs_list_2.append(f[key]['ResSpec']['raw_data']['freqs'][()])

    #creation of file
    with h5py.File(new_file, 'w') as f:
        for ii, key in enumerate(keys_list_1):
            top_group = f.create_group(key)
            sub1_group = top_group.create_group('ResSpec')
            sub2_group = sub1_group.create_group('params')    #creation of params folder
            sub2_group.create_dataset('Attenuation', data=[40]) # for low power 40, for high power 0
            sub2_group.create_dataset('Averages', data=[1600, 800, 400, 200, 100, 50, 25, 12, 6, 3, 1])
            #for low power 1600, 800, 400, 200, 100, 50, 25, 12, 6, 3, 1]) , for high power [1,1,1,1,1,1,1,1] (8 x 1)
            sub2_group.create_dataset('Frequency Centre', data=freqs_centers_list[ii])  #adding in data dependent upon resonator
            sub2_group.create_dataset('Frequency Span', data=freqs_span_list[ii])
            sub2_group.create_dataset('IF Bandwidth', data=[1000.0])
            sub2_group.create_dataset('Powers', data=[[-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0]])
            #for low power [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0], for high power [-40, -35, -30, -25, -20, -15, -10, -5]
            sub2_group.create_dataset('SParam', data=[b'S21'])
            sub3_group = sub1_group.create_group('raw_data')   #creation of raw_data folder
            sub3_group.create_dataset('complex', data=complex_list[ii])
            sub3_group.create_dataset('freqs', data=freqs_list[ii])

    #adding in data from the second file- 'a'
    with h5py.File(new_file, 'a') as f:
        for ii, key in enumerate(keys_list_2):
            top_group = f.create_group(key)
            sub1_group = top_group.create_group('ResSpec')
            sub2_group = sub1_group.create_group('params')
            sub2_group.create_dataset('Attenuation', data=[40])
            sub2_group.create_dataset('Averages', data=[1600, 800, 400, 200, 100, 50, 25, 12, 6, 3, 1])
            sub2_group.create_dataset('Frequency Centre', data=freqs_centers_list_2[ii])
            sub2_group.create_dataset('Frequency Span', data=freqs_span_list_2[ii])
            sub2_group.create_dataset('IF Bandwidth', data=[1000.0])
            sub2_group.create_dataset('Powers', data=[[-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0]])
            sub2_group.create_dataset('SParam', data=[b'S21'])
            sub3_group = sub1_group.create_group('raw_data')
            sub3_group.create_dataset('complex', data=complex_list_2[ii])
            sub3_group.create_dataset('freqs', data=freqs_list_2[ii])

def merge_two_data_sets_in_same_power_range_v2(file_name, file_name_2, new_file):

    with h5py.File(file_name, 'r') as f: #calling data from the first file
        keys_list_1 = []                       #making empty lists to append data into
        freqs_centers_list = []
        freqs_span_list = []
        complex_list = []
        freqs_list = []
        for key in f.keys():                    #appending data from files into new hdf5 file
            keys_list_1.append(key)
            #separate for ones below as they have variable data
            freqs_centers_list.append(f[key]['params']['Frequency Centre'][()])
            freqs_span_list.append(f[key]['params']['Frequency Span'][()])
            complex_list.append(f[key]['raw_data']['complex'][()])
            freqs_list.append(f[key]['raw_data']['freqs'][()])

    with h5py.File(file_name_2, 'r') as f:        #calling data from second file
        keys_list_2 = []
        freqs_centers_list_2 = []
        freqs_span_list_2 = []
        complex_list_2 = []
        freqs_list_2 = []
        for key in f.keys():
            keys_list_2.append(key)
            freqs_centers_list_2.append(f[key]['params']['Frequency Centre'][()])
            freqs_span_list_2.append(f[key]['params']['Frequency Span'][()])
            complex_list_2.append(f[key]['raw_data']['complex'][()])
            freqs_list_2.append(f[key]['raw_data']['freqs'][()])

    #creation of file
    with h5py.File(new_file, 'w') as f:
        for ii, key in enumerate(keys_list_1):
            top_group = f.create_group(key)
            sub1_group = top_group.create_group('ResSpec')
            sub2_group = sub1_group.create_group('params')    #creation of params folder
            sub2_group.create_dataset('Attenuation', data=[40]) # for low power 40, for high power 0
            sub2_group.create_dataset('Averages', data=[2048, 1024, 512, 256, 128,  64, 32, 16, 8, 4, 2, 2,2])
            #for low power 1600, 800, 400, 200, 100, 50, 25, 12, 6, 3, 1]) , for high power [1,1,1,1,1,1,1,1] (8 x 1)
            sub2_group.create_dataset('Frequency Centre', data=freqs_centers_list[ii])  #adding in data dependent upon resonator
            sub2_group.create_dataset('Frequency Span', data=freqs_span_list[ii])
            sub2_group.create_dataset('IF Bandwidth', data=[1000.0])
            sub2_group.create_dataset('Powers', data=[[-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10,  -5, 0]])
            #for low power [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0], for high power [-40, -35, -30, -25, -20, -15, -10, -5]
            sub2_group.create_dataset('SParam', data=[b'S21'])
            sub3_group = sub1_group.create_group('raw_data')   #creation of raw_data folder
            sub3_group.create_dataset('complex', data=complex_list[ii])
            sub3_group.create_dataset('freqs', data=freqs_list[ii])

    #adding in data from the second file- 'a'
    with h5py.File(new_file, 'a') as f:
        for ii, key in enumerate(keys_list_2):
            top_group = f.create_group(key)
            sub1_group = top_group.create_group('ResSpec')
            sub2_group = sub1_group.create_group('params')
            sub2_group.create_dataset('Attenuation', data=[40])
            sub2_group.create_dataset('Averages', data=[2048, 1024, 512, 256, 128,  64, 32, 16, 8, 4, 2, 2,2])
            sub2_group.create_dataset('Frequency Centre', data=freqs_centers_list_2[ii])
            sub2_group.create_dataset('Frequency Span', data=freqs_span_list_2[ii])
            sub2_group.create_dataset('IF Bandwidth', data=[1000.0])
            sub2_group.create_dataset('Powers', data=[[-60, -55,-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0]])
            sub2_group.create_dataset('SParam', data=[b'S21'])
            sub3_group = sub1_group.create_group('raw_data')
            sub3_group.create_dataset('complex', data=complex_list_2[ii])
            sub3_group.create_dataset('freqs', data=freqs_list_2[ii])



### Baseline fitting
def baseline_corr(freqs_data,complex_data):
    freqs_data_init = freqs_data
    fr_ind = np.argmin(20 * np.log10(np.abs(complex_data)))
    freqs_data = np.append(freqs_data[:fr_ind-100],freqs_data[fr_ind+100:])
    complex_data = np.append(complex_data[:fr_ind-100],complex_data[fr_ind+100:])
    # freqs_data = np.append(freqs_data[:fr_ind - 10], freqs_data[fr_ind + 10:])
    # complex_data = np.append(complex_data[:fr_ind - 10], complex_data[fr_ind + 10:])

    B0_guess_2 = (abs(complex_data)[-1] - abs(complex_data)[0]) / (freqs_data[-1] - freqs_data[0]) / 2 / np.pi
    A0_guess_2 = abs(complex_data)[0] - B0_guess_2 * 2 * np.pi * freqs_data[0]
    # When measure in broader frequency range use 3 lines below:
    t0_guess = (np.unwrap(np.angle(complex_data))[-1] - np.unwrap(np.angle(complex_data))[0]) / (
                freqs_data[-1] - freqs_data[0]) / 2 / np.pi
    phi_guess = np.unwrap(np.angle(complex_data))[0] - 2 * np.pi * freqs_data[0] * t0_guess
    # t0_guess = (np.angle(complex_data)[-1] - np.angle(complex_data[0]))/(freqs_data[-1] - freqs_data[0]) / 2 / np.pi
    # phi_guess = np.angle(complex_data)[-1] - 2*np.pi*freqs_data[-1]*t0_guess

    # N = len(freqs_data)
    # freqs_data = np.append(freqs_data[:int(N / 10)], freqs_data[-int(N / 10):])
    # complex_data = np.append(complex_data[:int(N / 10)], complex_data[-int(N / 10):])
    real_data = complex_data.real
    imag_data = complex_data.imag

    popt_2, pcov_2 = curve_fit(s21_bas_2, np.append(freqs_data, freqs_data), np.append(real_data, imag_data),
                               p0=[A0_guess_2, B0_guess_2, t0_guess, phi_guess])
    S21_bas_fit_2 = s21_bas_eval_2(np.asarray(freqs_data_init), popt_2)

    return popt_2, S21_bas_fit_2


def plot_baseline_corr(freqs_data, complex_data, S21_bas_fit, power, offset):
    fig20, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freqs_data / 1e9, complex_data.real)
    ax[0].plot(freqs_data / 1e9, S21_bas_fit.real)
    ax[0].plot(freqs_data / 1e9, complex_data.imag)
    ax[0].plot(freqs_data / 1e9, S21_bas_fit.imag)
    ax[0].set_xlabel(r'Freq [GHz]')
    ax[0].set_ylabel(r'Ampl')

    ax[1].plot(freqs_data / 1e9, abs(complex_data), 'b')
    ax[1].plot(freqs_data / 1e9, np.abs(S21_bas_fit))
    ax[1].set_ylabel(r'Ampl', color='b')
    ax1 = ax[1].twinx()
    ax1.plot(freqs_data / 1e9, np.angle(complex_data), 'r')
    ax1.plot(freqs_data / 1e9, np.angle(S21_bas_fit))
    ax1.set_ylabel(r'Phase [rad]', color='red')

    ax[2].plot(complex_data.real, complex_data.imag)
    ax[2].set_xlabel(r'$\Re[S_{21}]$')
    ax[2].set_ylabel(r'$\Im[S_{21}]$')
    ax[2].plot(S21_bas_fit.real, S21_bas_fit.imag)
    fig20.suptitle(f'Before the baseline correction at power = {power+offset} dBm')
    fig20.tight_layout()

def plot_fitted_circle(freqs_data, S21_res, S21_result, popt, Qi, power, offset, R2):
    S21_res = np.asarray(S21_res)
    from matplotlib.patches import Circle
    center = (0, 0)
    radius = 1
    circle = Circle(center, radius, fill=False, edgecolor='k', linestyle='--')

    fig50, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freqs_data / 1e9, S21_res.real, label=r'real')
    ax[0].plot(freqs_data / 1e9, S21_res.imag, label=r'imag')
    ax[0].plot(freqs_data / 1e9, S21_result.real,linewidth=0.7)
    ax[0].plot(freqs_data / 1e9, S21_result.imag,linewidth=0.7)
    ax[0].set_xlabel(r'Freq [GHz]')
    ax[0].set_ylabel(r'Ampl')
    ax[0].legend()

    ax[1].plot(freqs_data / 1e9, 20 * np.log10(abs(S21_res)), 'b')
    ax[1].plot(freqs_data / 1e9, 20 * np.log10(abs(S21_result)))
    ax[1].set_xlabel(r'Freq [GHz] ')
    ax[1].set_ylabel(r'$\vert S_{21}\vert$ [dB]', color='blue')
    ax1 = ax[1].twinx()
    ax1.plot(freqs_data / 1e9, np.angle(S21_res), 'r')
    ax1.plot(freqs_data / 1e9, np.angle(S21_result))
    ax1.set_ylabel(r'Phase [rad]', color='red')

    ax[2].plot(S21_res.real, S21_res.imag)
    ax[2].set_xlabel(r'$\Re[S_{21}]$')
    ax[2].set_ylabel(r'$\Im[S_{21}]$')
    ax[2].plot(S21_res[0].real, S21_res[0].imag, 'or', label='start')
    ax[2].plot(S21_res[-1].real, S21_res[-1].imag, 'ob', label='end')
    ax[2].plot(S21_result.real, S21_result.imag)
    ax[2].add_patch(circle)
    ax[2].legend(loc = 'lower left')
    ax[2].set_xlim([-1.2,1.2])
    ax[2].set_ylim([-1.2,1.2])
    ax[2].text(-1,0.75,f'Qi = {round(Qi/1e6,3)} M\nQc = {round(popt[2]/np.cos(popt[3])/1e6,3)} M\nf_r = {round(popt[0]/1e9,3)} GHz')
    ax[2].text(0.2,0.75, f'R2 = {round(R2,3)}')
    fig50.suptitle(f'Fitting the resonance at power = {power+offset} dBm')
    fig50.tight_layout()

def process_one_resonator_all_powers_raw(res, freqs, powers, S21_complex, offset, room_temp_freqs, room_temp_S21_complex):
    freqs_data = freqs[res]

    bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n, fr_err, Qi_err,\
        Qc_err, R2_raw = \
        [], [], [], [], [], [], [], [], [], [], [], [], []


    for ii, pow in enumerate(powers[res][0]):
        complex_data = S21_complex[res][ii]

        popt_2, S21_bas_fit_2  = baseline_corr(freqs_data,complex_data)
        bas_fit_param.append(popt_2)
        bas_fit_2_evaluation.append(S21_bas_fit_2)
        #plot_baseline_corr(freqs_data, complex_data, S21_bas_fit_2, pow, offset) # Plots the baseline experimental data and fitting done on it

        freqs_data, S21_res = s21_renorm(freqs_data, complex_data, popt_2, 2) # Renormalization of resonator data with the baseline correction
        popt, Qi_val, Qc_val, S21_result, fr_err_val, Qi_err_val, Qc_err_val, flag, R2_val = s21_res_fit(freqs_data, S21_res) # Fitting the baseline corrected data and returning the Qi and the fitted circle


        if type(popt) != int:
            fr_val_raw, Qi_val_raw, Ql_val_raw, absQc_val_raw, phi0_val_raw, A0_val_raw, B0_val_raw, t0_val_raw, phi_val_raw, \
            S21_result_val_raw, flag, fr_err_raw, Qc_err_raw, Qi_err_raw, R2_val_raw\
            = s21_res_fit_raw(freqs_data, complex_data, popt_2, popt)


            if flag == 1 and Qi_val_raw >0:
                xpoints, smoother_ypoints = S21_SG_function_data(room_temp_freqs, room_temp_S21_complex)
                atten_value = offset_estimate(fr_val_raw, xpoints, smoother_ypoints)

                freqs_splitter, S21_complex_splitter, powers_splitter = data_import(r'G:\Shared drives\MSDE_Trade_Secrets\Projects\cryo_materials\cpw_resonators\all_attenuation_data\splitter_atten.hdf5')
                xpoints_splitter, smoother_ypoints_splitter = S21_SG_function_data(freqs_splitter, S21_complex_splitter)
                splitter_atten = offset_estimate(fr_val_raw, xpoints_splitter, smoother_ypoints_splitter)
                
                offset_total = offset + atten_value + splitter_atten
                n = av_phot_number(pow, offset_total, Ql_val_raw, fr_val_raw, absQc_val_raw / np.cos(phi0_val_raw))
                av_n.append(n)
                fr.append(fr_val_raw)
                Ql.append(Ql_val_raw)
                Qi.append(Qi_val_raw)
                Qc.append(absQc_val_raw / np.cos(phi0_val_raw))
                S21_result_evaluation_raw.append(S21_result_val_raw)
                S21_res_renorm.append(S21_res)
                fr_err.append(fr_err_raw)
                Qc_err.append(Qc_err_raw)
                Qi_err.append(Qi_err_raw)
                R2_raw.append(R2_val_raw)
            else:
                av_n.append(np.nan)
                fr.append(np.nan)
                Ql.append(np.nan)
                Qi.append(np.nan)
                Qc.append(np.nan)
                S21_result_evaluation_raw.append(np.nan)
                S21_res_renorm.append(np.nan)
                fr_err.append(np.nan)
                Qc_err.append(np.nan)
                Qi_err.append(np.nan)
                R2_raw.append(np.nan)
        else:
            av_n.append(np.nan)
            fr.append(np.nan)
            Ql.append(np.nan)
            Qi.append(np.nan)
            Qc.append(np.nan)
            S21_result_evaluation_raw.append(np.nan)
            S21_res_renorm.append(np.nan)
            fr_err.append(np.nan)
            Qc_err.append(np.nan)
            Qi_err.append(np.nan)
            R2_raw.append(np.nan)

    return bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n, fr_err, Qc_err, Qi_err, R2_raw


def process_all_resonators_all_powers_raw(freqs, S21_complex, powers, offset, room_temp_freqs, room_temp_S21_complex):
    bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n,\
        fr_err_raw, Qi_err_raw, Qc_err_raw, R2_raw = \
        [], [], [], [], [], [], [], [], [], [], [], [], []
    for res in range(len(freqs)):
        # Process all resonators for all powers
        bas_fit_param_values, bas_fit_2_evaluation_values, fr_values, Ql_values, Qi_values, Qc_values, \
        S21_result_evaluation_raw_values, S21_res_renorm_values, av_n_values, fr_err_values, Qi_err_values, Qc_err_values, R2_values = \
        process_one_resonator_all_powers_raw(res, freqs, powers, S21_complex, offset, room_temp_freqs, room_temp_S21_complex)

        bas_fit_param.append(bas_fit_param_values)
        bas_fit_2_evaluation.append(bas_fit_2_evaluation_values)
        fr.append(fr_values)
        Ql.append(Ql_values)
        Qi.append(Qi_values)
        Qc.append(Qc_values)
        S21_result_evaluation_raw.append(S21_result_evaluation_raw_values)
        S21_res_renorm.append(S21_res_renorm_values)
        av_n.append(av_n_values)
        fr_err_raw.append(fr_err_values)
        Qi_err_raw.append(Qi_err_values)
        Qc_err_raw.append(Qc_err_values)
        R2_raw.append(R2_values)

    return bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n, fr_err_raw,\
        Qi_err_raw, Qc_err_raw, R2_raw

def save_results_to_hdf5(file_for_saving, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, geom = None):
    if not os.path.exists(file_for_saving):
        file = h5py.File(file_for_saving, 'w')
        try:
            file.create_dataset('res_index', data=list(range(len(powers))))
            file.create_dataset('powers', data=powers)
            file.create_dataset('res_freqs', data=fr)
            file.create_dataset('Qis', data=Qi)
            file.create_dataset('Qcs', data=Qc)
            file.create_dataset('av_photon', data=av_n)
            file.create_dataset('fr_err', data=fr_err)
            file.create_dataset('Qi_err', data=Qi_err)
            file.create_dataset('Qc_err', data=Qc_err)
            #if geom == None:
            #    geom_arr = []
            #    for res in range(len(powers)):
            #        frar = np.array(fr[res])[~np.isnan(np.array(fr[res]))]
            #        if (4.29 <= np.mean(frar)/1e9 <= 4.4) or (6.4 <= np.mean(frar)/1e9 <= 6.72):
            #            geom_arr.append(7.25)
            #       if (4.4 <= np.mean(frar)/1e9 <= 4.6) or (5.80 <= np.mean(frar)/1e9 <= 6.10) or (6.72 <= np.mean(frar)/1e9 <= 7.00):
            #            geom_arr.append(29.00)
            #        if (5.00 <= np.mean(frar)/1e9 <= 5.40) or (6.10 <= np.mean(frar)/1e9 <= 6.40) or (7.00 <= np.mean(frar)/1e9 <= 7.30):
            #            geom_arr.append(21.75)
            #        if (5.4 <= np.mean(frar)/1e9 <= 5.70) or (7.30 <= np.mean(frar)/1e9 <= 7.55):
            #            geom_arr.append(14.50)
            #    file.create_dataset('geom', data=list(geom_arr))
            #else:
            #    file.create_dataset('geom', data=list(geom))
            file.create_dataset('R2', data=list(R2))
            file.close()
            add_participation_ratios_hdf5_v1(file_for_saving)
        except:
            file.close()
            print('Could not write into the file!')
    else:
        print('File already exists! Delete the old file.')

def read_data_from_hdf5_file(file_to_read_from):
    try:
        with h5py.File(file_to_read_from) as file:
            res_index = np.asarray(file['res_index'])
            powers = np.asarray(file['powers'][()])
            #geom = np.asarray(file['geom'][()])
            fr = np.asarray(file['res_freqs'][()])
            Qi = np.asarray(file['Qis'][()])
            Qc = np.asarray(file['Qcs'][()])
            av_n = np.asarray(file['av_photon'][()])
            fr_err = np.asarray(file['fr_err'][()])
            Qi_err = np.asarray(file['Qi_err'][()])
            Qc_err = np.asarray(file['Qc_err'][()])
            R2 = np.asarray(file['R2'][()])
            p_surf = np.asarray(file['p_surf'][()])
            p_MA = np.asarray(file['p_MA'][()])
            p_SA = np.asarray(file['p_SA'][()])
            p_MS = np.asarray(file['p_MS'][()])
    except:
        print('File could not be found!')
    return res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS #Katie has removed geom from this return


def read_data_from_hdf5_file_v1(file_to_read_from):
    try:
        with h5py.File(file_to_read_from) as file:
            res_index = np.asarray(file['res_index'])
            powers = np.asarray(file['powers'][()])
            fr = np.asarray(file['res_freqs'][()])
            Qi = np.asarray(file['Qis'][()])
            Qc = np.asarray(file['Qcs'][()])
            av_n = np.asarray(file['av_photon'][()])
            fr_err = np.asarray(file['fr_err'][()])
            Qi_err = np.asarray(file['Qi_err'][()])
            Qc_err = np.asarray(file['Qc_err'][()])
            R2 = np.asarray(file['R2'][()])
            p_surf = np.asarray(file['p_surf'][()])
            p_MA = np.asarray(file['p_MA'][()])
            p_SA = np.asarray(file['p_SA'][()])
            p_MS = np.asarray(file['p_MS'][()])
    except:
        print('File could not be found!')
    return res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS


def plot_Qi_vs_n_all_resonators_all_powers(file_to_read_from, file_for_saving, plot_save):
    #res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, geom, R2, p_surf, p_MA, p_SA, p_MS = read_data_from_hdf5_file(file_to_read_from)
    res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS = read_data_from_hdf5_file_v1(file_to_read_from)

    av_n, Qi, Qc, Qi_err, Qc_err =filter_negative_Q_or_n(av_n, Qi, Qc, Qi_err, Qc_err)

    fig3, ax = plt.subplots(1, 2, figsize=(10, 5))
    for ii in range(len(av_n)):
        frar = np.array(fr[ii])[~np.isnan(np.array(fr[ii]))]
        ax[0].errorbar(av_n[ii], np.asarray(Qi[ii]) / 1e6, yerr=np.asarray(Qi_err[ii])/1e6, fmt='o', markerfacecolor='none', label=f'{round(np.mean(frar, 0) / 1e9, 3)} GHz') #- {geom[ii]}')
        ax[1].errorbar(av_n[ii], np.asarray(Qc[ii]) / 1e6, yerr=np.asarray(Qc_err[ii])/1e6, fmt='o', markerfacecolor='none', label=f'{round(np.mean(frar, 0) / 1e9, 3)} GHz') #- {geom[ii]}')

    ax[0].set_xlabel(r'$\langle n\rangle$')
    ax[0].set_ylabel(r'$Q_i$ [M]')
    ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_yscale('log') #changed from linear
    ax[1].set_xlabel(r'$\langle n\rangle$')
    ax[1].set_ylabel(r'$Q_c$ [M]')
    ax[1].legend()
    ax[1].set_xscale('log')
    ax[1].set_yscale('log') #changed from linear

    idx_s = file_to_read_from.rfind('\\')
    idx_e = file_to_read_from.rfind('.')
    fig3.suptitle('Log Qi scale for sample ' + file_to_read_from[idx_s + 1:idx_e])
    fig3.tight_layout()

    if plot_save == True:
        fig3.savefig(file_for_saving + '.png')

def fit_Qi_vs_n_all_resonators_v1(file_to_read_from,not_include_high_power,save_plot):
    #res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, geom, R2, p_surf, p_MA, p_SA, p_MS = \
        #read_data_from_hdf5_file(file_to_read_from)
    res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS = \
        read_data_from_hdf5_file_v1(file_to_read_from)
    fig101, ax = plt.subplots(1, 1, figsize=(5, 5))
    C1_ar, C2_ar, C3_ar, C4_ar, Qi_evaluation_ar, av_n_evaluation_ar, C1_err_ar, C2_err_ar, C3_err_ar, C4_err_ar = [], [], [], [], [], [], [], [], [], []
    for res in res_index:
        fr_clean = [x for x in fr[res][:-not_include_high_power] if math.isnan(x) == False]
        Qi_clean = [x for x in Qi[res][:-not_include_high_power] if math.isnan(x) == False]
        av_n_clean = [x for x in av_n[res][:-not_include_high_power] if math.isnan(x) == False]
        inv_Qi = np.divide(1, Qi_clean)
        try:
            popt, pcov = curve_fit(TLS_loss, av_n_clean, inv_Qi,
                                   p0=[np.mean(inv_Qi[:4]) - np.mean(inv_Qi[-4:]), 1, 0.5, np.mean(inv_Qi[-4:])])
            inv_Qi_evaluation = TLS_loss(av_n_clean, *popt)
            Qi_evaluation = np.divide(1, inv_Qi_evaluation)

            C1_ar.append(popt[0])
            C2_ar.append(popt[1])
            C3_ar.append(popt[2])
            C4_ar.append(popt[3])
            Qi_evaluation_ar.append(Qi_evaluation)
            av_n_evaluation_ar.append(av_n[res][:-not_include_high_power])
            C1_err_ar.append(np.sqrt(np.diag(pcov))[0])
            C2_err_ar.append(np.sqrt(np.diag(pcov))[1])
            C3_err_ar.append(np.sqrt(np.diag(pcov))[2])
            C4_err_ar.append(np.sqrt(np.diag(pcov))[3])

            ax.loglog(av_n[res][:-not_include_high_power], np.divide(Qi[res][:-not_include_high_power], 1e6), 'o',
                      markerfacecolor='none',
                      label=f'{round(np.mean(fr_clean) / 1e9, 3)} GHz')
            ax.loglog(av_n_clean, np.divide(Qi_evaluation, 1e6), '-')
        except:
            print(f'Could not fit res {res}')
            C1_ar.append(np.nan)
            C2_ar.append(np.nan)
            C3_ar.append(np.nan)
            C4_ar.append(np.nan)
            Qi_evaluation_ar.append(np.nan)
            av_n_evaluation_ar.append(np.nan)
            C1_err_ar.append(np.nan)
            C2_err_ar.append(np.nan)
            C3_err_ar.append(np.nan)
            C4_err_ar.append(np.nan)
            continue


    ax.legend()
    ax.set_xlabel(r'$\bar{n}$')
    ax.set_ylabel('$Q_i$ [M]')
    idx_s = file_to_read_from.rfind('\\')
    fig101.suptitle(r'Qi vs $\bar{n}$ for ' + file_to_read_from[idx_s + 1:idx_s + 11])
    fig101.tight_layout()
    if save_plot == True:
        fig101.savefig(file_to_read_from[:idx_s+1]+'Qi_vs_n_' + file_to_read_from[idx_s + 1:idx_s +11] + '.png')

    return C1_ar, C2_ar, C3_ar, C4_ar, Qi_evaluation_ar, av_n_evaluation_ar, C1_err_ar, C2_err_ar, C3_err_ar, C4_err_ar

def filter_negative_Q_or_n(av_n, Qi, Qc, Qi_err, Qc_err):
    for ii in range(len(av_n)):
        for jj in range(len(av_n[ii])):
            if av_n[ii][jj] < 0 or Qi[ii][jj] < 0 or Qc[ii][jj] < 0:
                av_n[ii][jj] = np.nan
                Qi[ii][jj] = np.nan
                Qc[ii][jj] = np.nan
                Qi_err[ii][jj] = np.nan
                Qc_err[ii][jj] = np.nan
    return av_n, Qi, Qc, Qi_err, Qc_err


def add_participation_ratios_hdf5(file_for_saving):
    with h5py.File(file_for_saving,'a') as file:
        if 'res_freqs' in file.keys():
            p_surf_ar, p_MA_ar, p_SA_ar, p_MS_ar = [], [], [], []
            for res in range(len(file['res_freqs'])):
                res_freq_list_clean = [x for x in file['res_freqs'][res] if math.isnan(x) == False]
                res_fr = np.mean(res_freq_list_clean)
                if res_fr / 1e9 < 4.4 and res_fr / 1e9 > 4.2:
                    p_surf_ar.append(np.nan)
                    p_MA_ar.append(np.nan)
                    p_SA_ar.append(np.nan)
                    p_MS_ar.append(np.nan)
                if res_fr / 1e9 < 4.6 and res_fr / 1e9 > 4.4:
                    p_surf_ar.append(14.683 * 1e-4)
                    p_MA_ar.append(0.163 * 1e-4)
                    p_SA_ar.append(10.714 * 1e-4)
                    p_MS_ar.append(3.806 * 1e-4)
                if res_fr / 1e9 < 5.4 and res_fr / 1e9 > 5.1:
                    p_surf_ar.append(21.033 * 1e-4)
                    p_MA_ar.append(0.256 * 1e-4)
                    p_SA_ar.append(15.28 * 1e-4)
                    p_MS_ar.append(5.497 * 1e-4)
                if res_fr / 1e9 < 5.7 and res_fr / 1e9 > 5.4:
                    p_surf_ar.append(29.195 * 1e-4)
                    p_MA_ar.append(0.311 * 1e-4)
                    p_SA_ar.append(22.377 * 1e-4)
                    p_MS_ar.append(6.506 * 1e-4)
                if res_fr / 1e9 < 6.1 and res_fr / 1e9 > 5.8:
                    p_surf_ar.append(14.343 * 1e-4)
                    p_MA_ar.append(0.15 * 1e-4)
                    p_SA_ar.append(10.549 * 1e-4)
                    p_MS_ar.append(3.643 * 1e-4)
                if res_fr / 1e9 < 6.4 and res_fr / 1e9 > 6.1:
                    p_surf_ar.append(20.915 * 1e-4)
                    p_MA_ar.append(0.253 * 1e-4)
                    p_SA_ar.append(15.228 * 1e-4)
                    p_MS_ar.append(5.433 * 1e-4)
                if res_fr / 1e9 < 6.72 and res_fr / 1e9 > 6.4:
                    p_surf_ar.append(np.nan)
                    p_MA_ar.append(np.nan)
                    p_SA_ar.append(np.nan)
                    p_MS_ar.append(np.nan)
                if res_fr / 1e9 < 7.00 and res_fr / 1e9 > 6.72:
                    p_surf_ar.append(14.989 * 1e-4)
                    p_MA_ar.append(0.17 * 1e-4)
                    p_SA_ar.append(10.883 * 1e-4)
                    p_MS_ar.append(3.936 * 1e-4)
                if res_fr / 1e9 < 7.3 and res_fr / 1e9 > 7.1:
                    p_surf_ar.append(20.773 * 1e-4)
                    p_MA_ar.append(0.246 * 1e-4)
                    p_SA_ar.append(15.176 * 1e-4)
                    p_MS_ar.append(5.351 * 1e-4)
                if res_fr / 1e9 < 7.5 and res_fr / 1e9 > 7.3:
                    p_surf_ar.append(32.708 * 1e-4)
                    p_MA_ar.append(0.446 * 1e-4)
                    p_SA_ar.append(23.87 * 1e-4)
                    p_MS_ar.append(8.391 * 1e-4)
            file.create_dataset('p_surf', data=p_surf_ar)
            file.create_dataset('p_MA', data=p_MA_ar)
            file.create_dataset('p_SA', data=p_SA_ar)
            file.create_dataset('p_MS', data=p_MS_ar)
        else: print('"res_freqs" is not in the file. Cannot add the surface participation ratio.')

def add_participation_ratios_hdf5_v1(file_for_saving):
    with h5py.File(file_for_saving,'a') as file:
        if 'res_freqs' in file.keys():
            p_surf_ar, p_MA_ar, p_SA_ar, p_MS_ar = [], [], [], []
            for res in range(len(file['res_freqs'])):
                res_freq_list_clean = [x for x in file['res_freqs'][res] if math.isnan(x) == False]
                res_fr = np.mean(res_freq_list_clean)
                if res_fr / 1e9 < 4.4 and res_fr / 1e9 > 4.2:
                    p_surf_ar.append(np.nan)
                    p_MA_ar.append(np.nan)
                    p_SA_ar.append(np.nan)
                    p_MS_ar.append(np.nan)
                elif res_fr / 1e9 < 4.6 and res_fr / 1e9 > 4.4:
                    p_surf_ar.append(14.683 * 1e-4)
                    p_MA_ar.append(0.163 * 1e-4)
                    p_SA_ar.append(10.714 * 1e-4)
                    p_MS_ar.append(3.806 * 1e-4)
                elif res_fr / 1e9 < 5.4 and res_fr / 1e9 > 5.1:
                    p_surf_ar.append(21.033 * 1e-4)
                    p_MA_ar.append(0.256 * 1e-4)
                    p_SA_ar.append(15.28 * 1e-4)
                    p_MS_ar.append(5.497 * 1e-4)
                elif res_fr / 1e9 < 5.7 and res_fr / 1e9 > 5.4:
                    p_surf_ar.append(29.195 * 1e-4)
                    p_MA_ar.append(0.311 * 1e-4)
                    p_SA_ar.append(22.377 * 1e-4)
                    p_MS_ar.append(6.506 * 1e-4)
                elif res_fr / 1e9 < 6.1 and res_fr / 1e9 > 5.8:
                    p_surf_ar.append(14.343 * 1e-4)
                    p_MA_ar.append(0.15 * 1e-4)
                    p_SA_ar.append(10.549 * 1e-4)
                    p_MS_ar.append(3.643 * 1e-4)
                elif res_fr / 1e9 < 6.4 and res_fr / 1e9 > 6.1:
                    p_surf_ar.append(20.915 * 1e-4)
                    p_MA_ar.append(0.253 * 1e-4)
                    p_SA_ar.append(15.228 * 1e-4)
                    p_MS_ar.append(5.433 * 1e-4)
                elif res_fr / 1e9 < 6.72 and res_fr / 1e9 > 6.4:
                    p_surf_ar.append(np.nan)
                    p_MA_ar.append(np.nan)
                    p_SA_ar.append(np.nan)
                    p_MS_ar.append(np.nan)
                elif res_fr / 1e9 < 7.00 and res_fr / 1e9 > 6.72:
                    p_surf_ar.append(14.989 * 1e-4)
                    p_MA_ar.append(0.17 * 1e-4)
                    p_SA_ar.append(10.883 * 1e-4)
                    p_MS_ar.append(3.936 * 1e-4)
                elif res_fr / 1e9 < 7.3 and res_fr / 1e9 > 7.1:
                    p_surf_ar.append(20.773 * 1e-4)
                    p_MA_ar.append(0.246 * 1e-4)
                    p_SA_ar.append(15.176 * 1e-4)
                    p_MS_ar.append(5.351 * 1e-4)
                elif res_fr / 1e9 < 7.5 and res_fr / 1e9 > 7.3:
                    p_surf_ar.append(32.708 * 1e-4)
                    p_MA_ar.append(0.446 * 1e-4)
                    p_SA_ar.append(23.87 * 1e-4)
                    p_MS_ar.append(8.391 * 1e-4)
                else :
                    p_surf_ar.append(np.nan)
                    p_MA_ar.append(np.nan)
                    p_SA_ar.append(np.nan)
                    p_MS_ar.append(np.nan)
            file.create_dataset('p_surf', data=p_surf_ar)
            file.create_dataset('p_MA', data=p_MA_ar)
            file.create_dataset('p_SA', data=p_SA_ar)
            file.create_dataset('p_MS', data=p_MS_ar)
        else: print('"res_freqs" is not in the file. Cannot add the surface participation ratio.')



def merge_two_power_ranges(file_1, file_2, offset_1, offset_2):
    res_index_1, powers_1, fr_1, Qi_1, Qc_1, av_n_1, fr_err_1, Qi_err_1, Qc_err_1, R2_1, p_surf_1, p_MA_1, p_SA_1, p_MS_1 = read_data_from_hdf5_file(file_1)
    res_index_2, powers_2, fr_2, Qi_2, Qc_2, av_n_2, fr_err_2, Qi_err_2, Qc_err_2, R2_2, p_surf_2, p_MA_2, p_SA_2, p_MS_2 = read_data_from_hdf5_file(file_2)

    fr_1_dict, fr_2_dict, Qi_1_dict, Qi_2_dict, Qc_1_dict, Qc_2_dict, av_n_1_dict, av_n_2_dict, powers_1_dict, powers_2_dict,\
     fr_err_1_dict, fr_err_2_dict, Qi_err_1_dict, Qi_err_2_dict, Qc_err_1_dict, Qc_err_2_dict, R2_1_dict, R2_2_dict \
    = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), \
    dict()

    for res in res_index_1:
        frar_1 = np.array(fr_1[res])[~np.isnan(np.array(fr_1[res]))]
        frar_2 = np.array(fr_2[res])[~np.isnan(np.array(fr_2[res]))]
        powers_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(powers_1[res][0] + offset_1)
        fr_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(fr_1[res])
        Qi_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(Qi_1[res])
        Qc_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(Qc_1[res])
        av_n_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(av_n_1[res])
        # geom_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = geom_1[res]
        fr_err_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(fr_err_1[res])
        Qi_err_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(Qi_err_1[res])
        Qc_err_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(Qc_err_1[res])
        R2_1_dict[str(round(np.mean(frar_1) / 1e9, 2))] = list(R2_1[res])

        powers_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(powers_2[res][0] + offset_2)
        fr_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(fr_2[res])
        Qi_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(Qi_2[res])
        Qc_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(Qc_2[res])
        av_n_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(av_n_2[res])
        # geom_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = geom_2[res]
        fr_err_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(fr_err_2[res])
        Qi_err_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(Qi_err_2[res])
        Qc_err_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(Qc_err_2[res])
        R2_2_dict[str(round(np.mean(frar_2) / 1e9, 2))] = list(R2_2[res])

    merged_powers, merged_Qi, merged_Qc, merged_av_n, merged_fr, merged_geom, merged_fr_err, merged_Qi_err, merged_Qc_err, merged_R2 \
    = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    for key in fr_1_dict.keys():
        merged_powers[key] = powers_1_dict[key] + powers_2_dict[key]
        merged_Qi[key] = Qi_1_dict[key] + Qi_2_dict[key]
        merged_fr[key] = fr_1_dict[key] + fr_2_dict[key]
        merged_Qc[key] = Qc_1_dict[key] + Qc_2_dict[key]
        merged_av_n[key] = av_n_1_dict[key] + av_n_2_dict[key]
        merged_fr_err[key] = fr_err_1_dict[key] + fr_err_2_dict[key]
        merged_Qi_err[key] = Qi_err_1_dict[key] + Qi_err_2_dict[key]
        merged_Qc_err[key] = Qc_err_1_dict[key] + Qc_err_2_dict[key]
        merged_R2[key] = R2_1_dict[key] + R2_2_dict[key]
        # if geom_1_dict[key] == geom_2_dict[key]:
            # merged_geom[key] = geom_1_dict[key]
        # else: print('Not equal geom values!')

    merged_powers_ar, merged_fr_ar, merged_Qi_ar, merged_Qc_ar, merged_av_n_ar, merged_geom_ar, merged_fr_err_ar, \
    merged_Qi_err_ar, merged_Qc_err_ar, merged_R2_ar = \
        [], [], [], [], [], [], [], [], [], []
    for key in sorted(merged_powers):
        merged_powers_ar.append(merged_powers[key])
        merged_fr_ar.append(merged_fr[key])
        merged_Qi_ar.append(merged_Qi[key])
        merged_Qc_ar.append(merged_Qc[key])
        merged_av_n_ar.append(merged_av_n[key])
        # merged_geom_ar.append(merged_geom[key])
        merged_fr_err_ar.append(merged_fr_err[key])
        merged_Qi_err_ar.append(merged_Qi_err[key])
        merged_Qc_err_ar.append(merged_Qc_err[key])
        merged_R2_ar.append(merged_R2[key])

    return merged_powers_ar, merged_fr_ar, merged_Qi_ar, merged_Qc_ar, merged_av_n_ar, merged_fr_err_ar, merged_Qi_err_ar, \
        merged_Qc_err_ar, merged_R2_ar #, merged_geom_ar

def save_results_to_hdf5_v1(file_for_saving, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2):
    if not os.path.exists(file_for_saving):
        file = h5py.File(file_for_saving, 'w')
        try:
            file.create_dataset('res_index', data=list(range(len(powers))))
            file.create_dataset('powers', data=powers)
            file.create_dataset('res_freqs', data=fr)
            file.create_dataset('Qis', data=Qi)
            file.create_dataset('Qcs', data=Qc)
            file.create_dataset('av_photon', data=av_n)
            file.create_dataset('fr_err', data=fr_err)
            file.create_dataset('Qi_err', data=Qi_err)
            file.create_dataset('Qc_err', data=Qc_err)
            file.create_dataset('R2', data=list(R2))
            file.close()
            add_participation_ratios_hdf5_v1(file_for_saving)
        except:
            file.close()
            print('Could not write into the file!')
    else:
        print('File already exists! Delete the old file.')

def filter_data_based_on_R2(file_to_look_into, R2_threshold, plot_save):
    if os.path.exists(file_to_look_into):
        res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS = \
            read_data_from_hdf5_file(file_to_look_into)

        xpoints = np.array(av_n)
        ypoints = np.array(R2)
        graph = plt.figure()
        plt.axhline(y=R2_threshold, color='r', linestyle='--', label='y= R2_threshold')
        plt.suptitle('R2 vs n')
        plt.xlabel('<n>')
        plt.ylabel('R2')
        plt.xscale('log')
        file_name = os.path.basename(file_to_look_into)
        wafer_name = os.path.splitext(file_name)[0]
        graph.suptitle(f"{R2_threshold} R2 filtering for {wafer_name}")
        plt.scatter(xpoints[ypoints <= R2_threshold], ypoints[ypoints <= R2_threshold], label=wafer_name + ' removed points',
                    facecolor='none', edgecolor='r', marker='o')
        plt.scatter(xpoints[ypoints > R2_threshold], ypoints[ypoints > R2_threshold],
                    label=wafer_name , facecolor='none', edgecolor='k', marker='o')
        #plt.xlim(1, 1e10)
        plt.legend()
        plt.tight_layout()
        plt.show()

        file = h5py.File(file_to_look_into, 'r+')

        for res in res_index:
            for p in range(len(powers[res])):
                if R2[res][p] <= R2_threshold:
                    fr[res][p] = np.nan
                    Qi[res][p] = np.nan
                    Qc[res][p] = np.nan
                    av_n[res][p] = np.nan
                    fr_err[res][p] = np.nan
                    Qi_err[res][p] = np.nan
                    Qc_err[res][p] = np.nan


        del file['res_freqs']
        file.create_dataset('res_freqs', data=fr)
        del file['fr_err']
        file.create_dataset('fr_err', data=fr_err)
        del file['Qcs']
        file.create_dataset('Qcs', data=Qc)
        del file['Qc_err']
        file.create_dataset('Qc_err', data=Qc_err)
        del file['Qis']
        file.create_dataset('Qis', data=Qi)
        del file['Qi_err']
        file.create_dataset('Qi_err', data=Qi_err)
        del file['av_photon']
        file.create_dataset('av_photon', data=av_n)
        file.close()
    else:
        print('File does not exist. Create it first.')
    if plot_save == True:
        graph.savefig(file_to_look_into + '.png')

def Qi_1p_from_exper(file_to_use):
    import scipy.interpolate as interpol
    with h5py.File(file_to_use, 'r') as file:
        for key in file.keys():
            try:
                if key == 'Qis':
                    Qis = file[str(key)][()]
                if key == 'av_photon':
                    av_n = file[str(key)][()]
            except: print('Qis or av_photon field is not found in the file!')
    if len(Qis) == len(av_n):
        Qi_1p = []
        for ii in range(len(Qis)):
            av_n_clean = [x for x in av_n[ii] if math.isnan(x) == False]
            Qis_clean = [x for x in Qis[ii] if math.isnan(x) == False]
            Qis_sorted = [x for _,x in sorted(zip(av_n_clean,Qis_clean))]
            av_n_sorted = sorted(av_n_clean)
            # cs = interpol.CubicSpline(av_n_sorted,Qis_sorted)
            f = interpol.interp1d(av_n_sorted,Qis_sorted)
            if av_n_sorted[0] < 1:
                Qi_1p.append(float(f(1)))
            else: Qi_1p.append(np.nan)
    return Qi_1p, Qis, av_n

def Qi_1Mp_from_exper(file_to_use):
    import scipy.interpolate as interpol
    with h5py.File(file_to_use, 'r') as file:
        for key in file.keys():
            try:
                if key == 'Qis':
                    Qis = file[str(key)][()]
                if key == 'av_photon':
                    av_n = file[str(key)][()]
            except: print('Qis or av_photon field is not found in the file!')
    if len(Qis) == len(av_n):
        Qi_1Mp = []
        for ii in range(len(Qis)):
            av_n_clean = [x for x in av_n[ii] if math.isnan(x) == False]
            Qis_clean = [x for x in Qis[ii] if math.isnan(x) == False]
            Qis_sorted = [x for _,x in sorted(zip(av_n_clean,Qis_clean))]
            av_n_sorted = sorted(av_n_clean)
            f = interpol.interp1d(av_n_sorted,Qis_sorted)
            if av_n_sorted[-1] > 1e6:
                Qi_1Mp.append(float(f(1e6)))
            else: Qi_1Mp.append(Qis_sorted[-1])
    return Qi_1Mp, Qis, av_n

def S21_SG_function_data(room_temp_freqs, room_temp_S21_complex):
    xpoints = np.array(room_temp_freqs[0]) #value might be 1 if there are two parts of the np.shape- check by plot
    ypoints = 20 * np.log10(np.array(abs(room_temp_S21_complex[0][0])))
     #dervied from VNA giving voltage out but we want power

    smoother_ypoints = savgol_filter(ypoints, window_length=51, polyorder=5)
    #found values to work well for three sets of data- can alter if plot isn't giving a good fit
    return xpoints, smoother_ypoints

def S21_SG_function_data_plot(room_temp_freqs, room_temp_S21_complex):
    xpoints = np.array(room_temp_freqs[0]) #value might be 1 if there are two parts of the np.shape- check by plot
    ypoints = 20 * np.log10(np.array(abs(room_temp_S21_complex[0][0])))
 #dervied from VNA giving voltage out but we want power

    smoother_ypoints = savgol_filter(ypoints, window_length=51, polyorder=5)
    #found values to work well for three sets of data- can alter if plot isn't giving a good fit

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(xpoints, ypoints, label='original data')
    ax.plot(xpoints, smoother_ypoints, '-g', label='SG smoothed data')
    ax.set_xlabel('freqs [Hz]')
    ax.set_ylabel(r'$S_{21}$ [dB]')
    ax.legend()

def offset_estimate(res_freqs,freqs_cal,atten_cal):
    itp_signal = interpol.interp1d(freqs_cal, atten_cal)
    return itp_signal(res_freqs)

def phase_res_freq_finder(freq_total, phases_total, N_ranges, folder, filename):
    graph, ax = plt.subplots(1,1,figsize=(7,5))
    res_freqs =[]
    for idx in range(N_ranges):
        freqs_ghz = np.divide(freq_total[idx], 1e9)
        phase_flat = np.unwrap(phases_total[idx][0], discont=None)
        phase_derivative = np.gradient(phase_flat)
        ax.set_xlabel('Freqs [GHz]')
        ax.set_ylabel('Phase [Rads]')
        ax.plot(freqs_ghz, phase_derivative)
        mean = np.mean(phase_derivative)
        stan_dev = np.std(phase_derivative)
        peaks, propoerties = find_peaks(phase_derivative, height=mean + 10*stan_dev, threshold=None, distance=None, prominence=0.15, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        if len(freqs_ghz[peaks])> 0:
            res_freqs.extend(freqs_ghz[peaks].tolist())
    graph.savefig(folder + '/' + filename.replace('.hdf5','.png'))
    return res_freqs 

def resonator_spectroscopy_2(vna, fcentre=10e9, fspan=2e9, powers=[], Attenuation=20, IF=1e3,
                                  n_pnts=2001, avgs=10, s_param='s21', plot=True, live_plot=False, save_hdf5=False,
                                  data_folder=r'C:\calibrations\testing',filename='testy.hdf5', name_tag=''):
    # vna.inst.write_str('SYSt:PRES')
    # trace = 'Trc1'
    # s = s_param
    # vna.inst.write_str("CALC:PAR:SDEF '" + str(trace) + "', '" + str(s) + "'")
    # s_index = 1
    # vna.inst.write_str("DISP:WIND:TRAC" + str(s_index + 1) + ":FEED '" + str(trace) + "'")
    vna.reset()
    vna.set_measurement_type(s=s_param, trace='Trc1')
    vna.set_band_if(IF)
    start, stop = fcentre - fspan / 2, fcentre + fspan / 2
    attenuation = Attenuation



    vna.set_start(start, channel=1)
    vna.set_stop(stop, channel=1)
    # vna.set_active_channel(1)
    vna.set_average_state(1)
    if len(avgs)==1:
        vna.set_averages(avgs[0])
    vna.set_points(n_pnts)

    params={}
    params['Frequency Centre'] = fcentre
    params['Frequency Span'] = fspan
    params['IF Bandwidth'] = IF
    params['SParam'] = s_param
    params['Powers'] = powers
    params['Averages'] = avgs
    params['Attenuation'] = Attenuation

    for param_key, param_val in params.items():
        if params[param_key] is list:
            params[param_key] = param_val
        else:
            params[param_key] = [param_val]

    start = time.time()
    live_traces = np.zeros((len(powers), n_pnts))
    runs= len(powers)
    i=0
    res=[]
    ims=[]
    amps=[]
    phases=[]
    meas_time=[]
    for indx, power in enumerate(powers):
        vna.set_power(power)
        if len(avgs) == len(powers):
            vna.set_averages(avgs[indx])
        elif len(avgs)>1:
            print(f'Please use a a static number of averages a list of averages equal in length to list of pwoers')
            return
        data = vna.get_trace()
        res.append(data[1])
        ims.append(data[2])

        amp, phase = vna_tools.complex_to_dbphase(np.asarray(data[1]) + 1j * np.asarray(data[2]))
        amps.append(amp)
        phases.append(phase)
        freq = data[0]
        if live_plot:
            i, meas_time = plotting_tools.meas_counter(i, start, runs, meas_time=meas_time)
            live_traces = plotting_tools.live_plotting_heatmap_trace(live_traces, indx, freq, powers, amp, np.zeros(np.shape(amp)),
                                                       y_label='Power [dBm]', x_label='Freq [GHz]', cmap_cols='Reds')
        if plot:


            plt.plot(np.divide(freq, 1e9), amp, label=fr'$power = {power:.1f} $ dBm')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('S21 (dB)')
            plt.legend()
            plt.show()



    if save_hdf5:
        save_script.save_sweep_trace_hdf5(freq, res, ims, params, hdf5_file=os.path.join(data_folder,filename), name_tag=os.path.join(name_tag,'ResSpec'))
        print(name_tag)

    return freq, amps, phases

def multi_small_freq_sweep(N_ranges, powers, avgs, f_start, f_stop, n_points, vna, folder, filename_0, name_tag):
    fspan = (f_stop-f_start)/N_ranges # frequency span in one sweep
    fcentre_list = np.zeros(len(range(N_ranges)))
    for ii in range(N_ranges):
        fcentre_list[ii] = f_start + ii*fspan
    fcentre_list = np.add(fcentre_list,fspan/2)
    freq_total = []
    amps_total = []
    phases_total = []
    for ind, fcentre in enumerate(fcentre_list):
        freq, amps, phases = resonator_spectroscopy_2(vna, fcentre=fcentre, fspan=fspan, powers=powers, Attenuation=0, IF=1e3,
                                      n_pnts=n_points, avgs=avgs, s_param='S21', plot=True, save_hdf5=True,
                                  data_folder=folder,filename=filename_0.replace('.hdf5','') + f'range_{ind}' + '.hdf5', name_tag=name_tag)
        freq_total.append(freq)
        amps_total.append(amps)
        phases_total.append(phases)
    return freq_total, amps_total, phases_total

def phase_sweep_and_analysis(N_ranges, powers, avgs, f_start, f_stop, n_points, vna, folder, filename_0, name_tag):
    freq_total, amps_total, phases_total = multi_small_freq_sweep(N_ranges, powers, avgs, f_start, f_stop, n_points, vna, folder, filename_0, name_tag)
    res_freqs = phase_res_freq_finder(freq_total, phases_total, N_ranges, folder, filename_0)

    return freq_total, amps_total, phases_total, res_freqs

def save_results_to_hdf5_attenuation(file_for_saving, xpoints, ypoints):
    if not os.path.exists(file_for_saving):
        file = h5py.File(file_for_saving, 'w')
        try:
            file.create_dataset('xpoints', data=xpoints)
            file.create_dataset('ypoints', data=ypoints)
            file.close()
        except:
            file.close()
            print('Could not write into the file!')
    else:
        print('File already exists! Delete the old file.')

def refining_fr_and_fspans(QT, vna, res_freqs, fspan = 10e6, powers = np.linspace(-20,-20,1),
      Attenuation=0, IF = 1e3, npoints = 2001, avgs= [1], s_param = 'S21', plot = False, save_hdf5 = False, folder =
       r'/home/jovyan/calibrations/2025/MX2_CD4_CPW/cryo_tests',
       filename = 'f_span_searching', name_tag = 'fspan_search', filename_at = r'/home/jovyan/calibrations/2025/MX2_CD4_CPW/RT_tests/in_54_attenuation.hdf5',
                           offset = 0, R2_threshold = 99.5, max_deviation = 2):

    res_freqs_hz = np.multiply(res_freqs, 1e9)
    fspans = []
    fr_final = []

    for fr_phase in res_freqs_hz:
        Ql_arr = []
        Qi_arr = []
        Qc_arr = []
        fr_arr = []
        fr_current = fr_phase
        k = 0
        while True:
            freq, amps, phases = resonator_spectroscopy_2(vna, fcentre=fr_current, fspan=fspan, powers=powers,
                                                               Attenuation=Attenuation, IF=IF,
                                                               n_pnts=npoints, avgs=avgs, s_param=s_param,  plot=plot,
                                                               save_hdf5=save_hdf5,
                                                               data_folder=folder, filename=filename,
                                                               name_tag=name_tag)
            amps = amps[0]
            phases = phases[0]

            lin_amps = 10 ** (amps / 20)
            S21_complex = lin_amps * np.exp(1j * phases)
            S21_complex = [[S21_complex]]
            powers_new = [[list(powers)]]
            freqs = [freq]
            res = 0

            room_temp_freqs, room_temp_S21_complex, room_temp_powers = data_import(filename_at)

            bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n, \
                fr_err, Qc_err, Qi_err, R2_raw = process_one_resonator_all_powers_raw(res, freqs, powers_new,
                                                                                           S21_complex, offset,
                                                                                           room_temp_freqs,
                                                                                           room_temp_S21_complex)

            fr_arr.append(fr[0])
            Ql_arr.append(Ql[0])
            Qi_arr.append(Qi[0])
            Qc_arr.append(Qc[0])


            fr_current = fr_arr[-1]
            if R2_raw[0] * 100 < R2_threshold:
                fspan = 0.9 * fspan

            else:
                if 20 * fr[0] / Ql[0] > 10e6:
                    pass
                else: fspan = 20 * fr[0] / Ql[0]
            if k > 3:
                Qi_dev = max(abs(Qi_arr[-3:] - np.mean(Qi_arr[-3:]))) * 100 / np.mean(Qi_arr[-3:])
                Qc_dev = max(abs(Qc_arr[-3:] - np.mean(Qc_arr[-3:]))) * 100 / np.mean(Qc_arr[-3:])
                fr_dev = max(abs(fr_arr[-3:] - np.mean(fr_arr[-3:]))) * 100 / np.mean(fr_arr[-3:])
                Ql_dev = max(abs(Ql_arr[-3:] - np.mean(Ql_arr[-3:]))) * 100 / np.mean(Ql_arr[-3:])
                if max(Qi_dev, Qc_dev, fr_dev, Ql_dev) < max_deviation:
                    break
            k = k + 1

            fig51, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].plot(np.asarray(fr_arr) / 1e9, 'ob')
            ax[0].set_xlabel('iteration #')
            ax[0].set_ylabel('Res frequency [GHz]')
            ax[1].plot(np.asarray(Ql_arr) / 1e6, 'ob')
            ax[1].set_xlabel('iteration #')
            ax[1].set_ylabel('Ql [M]')
            ax[2].plot(np.asarray(Qi_arr) / 1e6, 'ob')
            ax[2].set_xlabel('iteration #')
            ax[2].set_ylabel('Qi [M]')
            ax[3].plot(np.asarray(Qc_arr) / 1e6, 'ob')
            ax[3].set_xlabel('iteration #')
            ax[3].set_ylabel('Qc [M]')
            fig51.tight_layout()
            plt.show(fig51)
            plt.close

        fspans.append(fspan)
        fr_final.append(fr_arr[-1])
    return fspans, fr_final, Ql_arr, Qi_arr, Qc_arr


### Qi vs n fitting
def TLS_loss(n, C1, C2, C3, C4):
    return C1 / np.sqrt(1 + (n / C2) ** C3) + C4

def TLS_loss_v1(n, C1, C2, C3, C4):
    return C1 / (1 + n / C2)**C3 + C4

def fit_Qi_vs_n_all_resonators(file_to_read_from,not_include_high_power,save_plot):
    #res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, geom, R2, p_surf, p_MA, p_SA, p_MS = \
        #read_data_from_hdf5_file(file_to_read_from)
    res_index, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, p_surf, p_MA, p_SA, p_MS = \
        read_data_from_hdf5_file_v1(file_to_read_from)
    fig101, ax = plt.subplots(1, 1, figsize=(5, 5))
    C1_ar, C2_ar, C3_ar, C4_ar, Qi_evaluation_ar, av_n_evaluation_ar, C1_err_ar, C2_err_ar, C3_err_ar, C4_err_ar = [], [], [], [], [], [], [], [], [], []
    for res in res_index:
        fr_clean = [x for x in fr[res][:-not_include_high_power] if math.isnan(x) == False]
        Qi_clean = [x for x in Qi[res][:-not_include_high_power] if math.isnan(x) == False]
        av_n_clean = [x for x in av_n[res][:-not_include_high_power] if math.isnan(x) == False]
        inv_Qi = np.divide(1, Qi_clean)
        try:
            popt, pcov = curve_fit(TLS_loss, av_n_clean, inv_Qi,
                                   p0=[np.mean(inv_Qi[:4]) - np.mean(inv_Qi[-4:]), 1, 0.5, np.mean(inv_Qi[-4:])])
            inv_Qi_evaluation = TLS_loss(av_n_clean, *popt)
            Qi_evaluation = np.divide(1, inv_Qi_evaluation)

            C1_ar.append(popt[0])
            C2_ar.append(popt[1])
            C3_ar.append(popt[2])
            C4_ar.append(popt[3])
            Qi_evaluation_ar.append(Qi_evaluation)
            av_n_evaluation_ar.append(av_n[res][:-not_include_high_power])
            C1_err_ar.append(np.sqrt(np.diag(pcov))[0])
            C2_err_ar.append(np.sqrt(np.diag(pcov))[1])
            C3_err_ar.append(np.sqrt(np.diag(pcov))[2])
            C4_err_ar.append(np.sqrt(np.diag(pcov))[3])

            ax.loglog(av_n[res][:-not_include_high_power], np.divide(Qi[res][:-not_include_high_power], 1e6), 'o',
                      markerfacecolor='none',
                      label=f'{round(np.mean(fr_clean) / 1e9, 3)} GHz - {geom[res]} $\mu$m')
            ax.loglog(av_n_clean, np.divide(Qi_evaluation, 1e6), '-')
        except:
            print(f'Could not fit res {res}')
            C1_ar.append(np.nan)
            C2_ar.append(np.nan)
            C3_ar.append(np.nan)
            C4_ar.append(np.nan)
            Qi_evaluation_ar.append(np.nan)
            av_n_evaluation_ar.append(np.nan)
            C1_err_ar.append(np.nan)
            C2_err_ar.append(np.nan)
            C3_err_ar.append(np.nan)
            C4_err_ar.append(np.nan)
            continue


    ax.legend()
    ax.set_xlabel('$\overline{n}$')
    ax.set_ylabel('$Q_i$ [M]')
    idx_s = file_to_read_from.rfind('\\')
    fig101.suptitle('Qi vs $\overline{n}$ for ' + file_to_read_from[idx_s + 1:idx_s + 9])
    fig101.tight_layout()
    if save_plot == True:
        fig101.savefig(file_to_read_from[:idx_s+1]+'Qi_vs_n_' + file_to_read_from[idx_s + 1:idx_s + 9] + '.png')

    return C1_ar, C2_ar, C3_ar, C4_ar, Qi_evaluation_ar, av_n_evaluation_ar, C1_err_ar, C2_err_ar, C3_err_ar, C4_err_ar


def data_import_old_data(filename):

    freqs = []
    S21_complex = []
    powers = []
    with h5py.File(filename, 'r') as file:
        for ind, key in enumerate(file.keys()):
            freqs.append(np.asarray(file[str(key)]['raw_data']['freqs'][()]))
            S21_complex.append(np.asarray(file[str(key)]['raw_data']['complex'][()]))
            powers.append(file[str(key)]['params']['Powers'][()])
    return freqs, S21_complex, powers

def atten_data(file_path_at):

    room_temp_freqs_50 = room_temp_S21_complex_50 = room_temp_powers = None
    room_temp_freqs_51 = room_temp_S21_complex_51 = room_temp_powers = None
    room_temp_freqs_52 = room_temp_S21_complex_52 = room_temp_powers = None
    room_temp_freqs_53 = room_temp_S21_complex_53 = room_temp_powers = None
    room_temp_freqs_54 = room_temp_S21_complex_54 = room_temp_powers = None
    room_temp_freqs_55 = room_temp_S21_complex_55 = room_temp_powers = None
    for filename in sorted(os.listdir(folder_path_at),reverse=True):
        file_path_at = os.path.join(folder_path_at, filename)
        if os.path.isfile(file_path_at):
            if '50' in filename:
                room_temp_freqs_50, room_temp_S21_complex_50, room_temp_powers = resa.data_import(file_path_at)
            elif '51' in filename:
                room_temp_freqs_51, room_temp_S21_complex_51, room_temp_powers = resa.data_import(file_path_at)
            elif '52' in filename:
                room_temp_freqs_52, room_temp_S21_complex_52, room_temp_powers = resa.data_import(file_path_at)
            elif '53' in filename:
                room_temp_freqs_53, room_temp_S21_complex_53, room_temp_powers = resa.data_import(file_path_at)
            elif '54' in filename:
                room_temp_freqs_54, room_temp_S21_complex_54, room_temp_powers = resa.data_import(file_path_at)
            elif '55' in filename:
                room_temp_freqs_55, room_temp_S21_complex_55, room_temp_powers = resa.data_import(file_path_at)
    return (room_temp_freqs_50, room_temp_S21_complex_50, room_temp_freqs_51, room_temp_S21_complex_51, room_temp_freqs_52, \
            room_temp_S21_complex_52, room_temp_freqs_53, room_temp_S21_complex_53, room_temp_freqs_54, room_temp_S21_complex_54, \
            room_temp_freqs_55, room_temp_S21_complex_55)


def processing_all_data_high_and_low(file_path_at, data_folder_path, folder_for_saving_processed_data):
    room_temp_freqs_50, room_temp_S21_complex_50, room_temp_freqs_51, room_temp_S21_complex_51, room_temp_freqs_52, \
        room_temp_S21_complex_52, room_temp_freqs_53, room_temp_S21_complex_53, room_temp_freqs_54, room_temp_S21_complex_54, \
        room_temp_freqs_55, room_temp_S21_complex_55 = atten_data(file_path_at)

    room_temp_freqs_50, room_temp_S21_complex_50, room_temp_freqs_51, room_temp_S21_complex_51, room_temp_freqs_52, \
        room_temp_S21_complex_52, room_temp_freqs_53, room_temp_S21_complex_53, room_temp_freqs_54, room_temp_S21_complex_54, \
        room_temp_freqs_55, room_temp_S21_complex_55 = resa.atten_data(folder_path_at)

    for filename in sorted(os.listdir(data_folder_path), reverse=True):
        file_path = os.path.join(data_folder_path, filename)
        if os.path.isfile(file_path):
            if filename.startswith('W') and 'low' in filename:
                if '50' in filename:
                    room_temp_freqs = room_temp_freqs_50
                    room_temp_S21_complex = room_temp_S21_complex_50
                if '51' in filename:
                    room_temp_freqs = room_temp_freqs_51
                    room_temp_S21_complex = room_temp_S21_complex_51
                if '52' in filename:
                    room_temp_freqs = room_temp_freqs_52
                    room_temp_S21_complex = room_temp_S21_complex_52
                if '53' in filename:
                    room_temp_freqs = room_temp_freqs_53
                    room_temp_S21_complex = room_temp_S21_complex_53
                if '54' in filename:
                    room_temp_freqs = room_temp_freqs_54
                    room_temp_S21_complex = room_temp_S21_complex_54
                if '55' in filename:
                    room_temp_freqs = room_temp_freqs_55
                    room_temp_S21_complex = room_temp_S21_complex_55
                with h5py.File(file_path, 'r') as f:
                    first_keys = list(f.keys())
                    if first_keys:  # make sure it's not empty
                        first_key = first_keys[0]
                        if 'ResSpec' in f[first_key]:
                            print("  ResSpec found ")
                            freqs, S21_complex, powers = resa.data_import(file_path)
                        else:
                            print("  ResSpec not found ")
                            freqs, S21_complex, powers = resa.data_import_2(file_path)
                offset = -40
                bas_fit_param, bas_fit_2_evaluation, fr_raw, Ql_raw, Qi_raw, Qc_raw, S21_result_evaluation_raw, S21_res_renorm, av_n_raw, fr_err_raw, \
                    Qi_err_raw, Qc_err_raw, R2_raw = resa.process_all_resonators_all_powers_raw(freqs, S21_complex,
                                                                                                powers, offset,
                                                                                                room_temp_freqs,
                                                                                                room_temp_S21_complex)
                # bas_fit_param, bas_fit_2_evaluation, fr_raw, Ql_raw, Qc_raw, S21_result_evaluation_raw, S21_res_renorm, av_n_raw, fr_err_raw, Qi_err_raw, Qc_err_raw, R2_raw = resa.process_all_resonators_all_powers_raw(freqs, S21_complex, powers, offset, room_temp_freqs, room_temp_S21_complex)
                file_for_saving = folder_for_saving_processed_data + filename + '_processed' + '.hdf5'
                resa.save_results_to_hdf5(file_for_saving, powers, fr_raw, Qi_raw, Qc_raw, av_n_raw, fr_err_raw,
                                          Qi_err_raw, Qc_err_raw, R2_raw)
            else:
                if '50' in filename:
                    room_temp_freqs = room_temp_freqs_50
                    room_temp_S21_complex = room_temp_S21_complex_50
                if '51' in filename:
                    room_temp_freqs = room_temp_freqs_51
                    room_temp_S21_complex = room_temp_S21_complex_51
                if '52' in filename:
                    room_temp_freqs = room_temp_freqs_52
                    room_temp_S21_complex = room_temp_S21_complex_52
                if '53' in filename:
                    room_temp_freqs = room_temp_freqs_53
                    room_temp_S21_complex = room_temp_S21_complex_53
                if '54' in filename:
                    room_temp_freqs = room_temp_freqs_54
                    room_temp_S21_complex = room_temp_S21_complex_54
                if '55' in filename:
                    room_temp_freqs = room_temp_freqs_55
                    room_temp_S21_complex = room_temp_S21_complex_55
                with h5py.File(file_path, 'r') as f:
                    first_keys = list(f.keys())
                    if first_keys:  # make sure it's not empty
                        first_key = first_keys[0]
                        if 'ResSpec' in f[first_key]:
                            print("  ResSpec found ")
                            freqs, S21_complex, powers = resa.data_import(file_path)
                        else:
                            print("  ResSpec not found ")
                            freqs, S21_complex, powers = resa.data_import_2(file_path)
                offset = 0
                bas_fit_param, bas_fit_2_evaluation, fr_raw, Ql_raw, Qi_raw, Qc_raw, S21_result_evaluation_raw, S21_res_renorm, av_n_raw, fr_err_raw, \
                    Qi_err_raw, Qc_err_raw, R2_raw = resa.process_all_resonators_all_powers_raw(freqs, S21_complex,
                                                                                                powers, offset,
                                                                                                room_temp_freqs,
                                                                                                room_temp_S21_complex)
                file_for_saving = folder_for_saving_processed_data + filename + '_processed' + '.hdf5'
                resa.save_results_to_hdf5(file_for_saving, powers, fr_raw, Qi_raw, Qc_raw, av_n_raw, fr_err_raw,
                                          Qi_err_raw, Qc_err_raw, R2_raw)


def add_amat_participation_ratios_hdf5(file_for_saving, amat_participation_ratios, variation):
    with h5py.File(file_for_saving, 'a') as file: 
        if 'res_freqs' in file.keys():
            p_surf_ar, p_MA_ar, p_SA_ar, p_MS_ar = [], [], [], []
            for res in range(len(file['res_freqs'])):
                res_freq_list_clean = [x for x in file['res_freqs'][res] if math.isnan(x) == False]
                res_fr = np.mean(res_freq_list_clean)
                res_fr = np.divide(res_fr, 1e9)
                for row in amat_participation_ratios: 
                    if abs(row[0] - res_fr) <= 0.2:
                        p_surf_ar.append(row[1] + row[2] + row[3])
                        p_MA_ar.append(row[1])
                        p_SA_ar.append(row[2])
                        p_MS_ar.append(row[3])
            file.create_dataset('p_surf', data= p_surf_ar)
            file.create_dataset('p_MA', data=p_MA_ar)
            file.create_dataset('p_SA', data=p_SA_ar)
            file.create_dataset('p_MS', data=p_MS_ar)

def data_import_2(filename):

    freqs = []
    S21_complex = []
    powers = []
    with h5py.File(filename, 'r') as file:
        for ind, key in enumerate(file.keys()):
            freqs.append(np.asarray(file[str(key)]['raw_data']['freqs'][()]))
            S21_complex.append(np.asarray(file[str(key)]['raw_data']['complex'][()]))
            powers.append(file[str(key)]['params']['Powers'][()])
    return freqs, S21_complex, powers

def atten_data(file_path_at):

    room_temp_freqs_50 = room_temp_S21_complex_50 = room_temp_powers = None
    room_temp_freqs_51 = room_temp_S21_complex_51 = room_temp_powers = None
    room_temp_freqs_52 = room_temp_S21_complex_52 = room_temp_powers = None
    room_temp_freqs_53 = room_temp_S21_complex_53 = room_temp_powers = None
    room_temp_freqs_54 = room_temp_S21_complex_54 = room_temp_powers = None
    room_temp_freqs_55 = room_temp_S21_complex_55 = room_temp_powers = None
    for filename in sorted(os.listdir(folder_path_at),reverse=True):
        file_path_at = os.path.join(folder_path_at, filename)
        if os.path.isfile(file_path_at):
            if '50' in filename: 
                room_temp_freqs_50, room_temp_S21_complex_50, room_temp_powers = resa.data_import(file_path_at)
            elif '51' in filename:
                room_temp_freqs_51, room_temp_S21_complex_51, room_temp_powers = resa.data_import(file_path_at)
            elif '52' in filename:
                room_temp_freqs_52, room_temp_S21_complex_52, room_temp_powers = resa.data_import(file_path_at)
            elif '53' in filename: 
                room_temp_freqs_53, room_temp_S21_complex_53, room_temp_powers = resa.data_import(file_path_at) 
            elif '54' in filename:
                room_temp_freqs_54, room_temp_S21_complex_54, room_temp_powers = resa.data_import(file_path_at) 
            elif '55' in filename:
                room_temp_freqs_55, room_temp_S21_complex_55, room_temp_powers = resa.data_import(file_path_at) 
    return room_temp_freqs_50, room_temp_S21_complex_50, room_temp_freqs_51, room_temp_S21_complex_51, room_temp_freqs_52, room_temp_S21_complex_52, room_temp_freqs_53, room_temp_S21_complex_53, room_temp_freqs_54, room_temp_S21_complex_54, room_temp_freqs_55, room_temp_S21_complex_55

def save_results_to_hdf5_amat(file_for_saving, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, amat_participation_ratios, variation= 0.2, geom = None):
    if not os.path.exists(file_for_saving):
        file = h5py.File(file_for_saving, 'w')
        try:
            file.create_dataset('res_index', data=list(range(len(powers))))
            file.create_dataset('powers', data=powers)
            file.create_dataset('res_freqs', data=fr)
            file.create_dataset('Qis', data=Qi)
            file.create_dataset('Qcs', data=Qc)
            file.create_dataset('av_photon', data=av_n)
            file.create_dataset('fr_err', data=fr_err)
            file.create_dataset('Qi_err', data=Qi_err)
            file.create_dataset('Qc_err', data=Qc_err)
            file.create_dataset('R2', data=list(R2))
            file.close()
            add_amat_participation_ratios_hdf5(file_for_saving, amat_participation_ratios, variation)
        except:
            file.close()
            print('Could not write into the file!')
    else:
        print('File already exists! Delete the old file.')

def refining_fr_and_fspans_v2(QT, vna, res_freqs, fspan = 10e6, powers = np.linspace(-20,-20,1),
      Attenuation=0, IF = 1e3, npoints = 2001, avgs= [1], s_param = 'S21', plot = False, save_hdf5 = False, folder =
       r'/home/jovyan/calibrations/2025/MX2_CD4_CPW/cryo_tests', name_tag = 'fspan_search', filename_at = r'/home/jovyan/calibrations/2025/MX2_CD4_CPW/RT_tests/in_54_attenuation.hdf5',
                           offset = 0, R2_threshold = 99.5, max_deviation = 2):

    res_freqs_hz = np.multiply(res_freqs, 1e9)
    fspans = []
    fr_final = []

    for r, fr_phase in enumerate(res_freqs_hz):
        Ql_arr = []
        Qi_arr = []
        Qc_arr = []
        fr_arr = []
        fr_current = fr_phase
        k = 0
        while True:
            filename = f'f_span_searching_r{r}_k{k}.hdf5'
            freq, amps, phases = QT.resonator_spectroscopy_2(vna, fcentre=fr_current, fspan=fspan, powers=powers,
                                                               Attenuation=Attenuation, IF=IF,
                                                               n_pnts=npoints, avgs=avgs, s_param=s_param,  plot=plot,
                                                               save_hdf5=True,
                                                               data_folder=folder, filename=filename,
                                                               name_tag=name_tag)
            file_to_read = os.path.join(folder, filename)
            freqs, S21_complex, powers_new = data_import(file_to_read)
            res = 0
            room_temp_freqs, room_temp_S21_complex, room_temp_powers = data_import(filename_at)
            bas_fit_param, bas_fit_2_evaluation, fr, Ql, Qi, Qc, S21_result_evaluation_raw, S21_res_renorm, av_n, \
                fr_err, Qc_err, Qi_err, R2_raw = process_one_resonator_all_powers_raw(res, freqs, powers_new,
                                                                                           S21_complex, offset,
                                                                                           room_temp_freqs,
                                                                                           room_temp_S21_complex)

            fr_arr.append(fr[0])
            Ql_arr.append(Ql[0])
            Qi_arr.append(Qi[0])
            Qc_arr.append(Qc[0])


            fr_current = fr_arr[-1]
            if R2_raw[0] * 100 < R2_threshold:
                fspan = 0.9 * fspan

            else:
                if 20 * fr[0] / Ql[0] > 10e6:
                    pass
                else: fspan = 20 * fr[0] / Ql[0]
            if k > 3:
                Qi_dev = max(abs(Qi_arr[-3:] - np.mean(Qi_arr[-3:]))) * 100 / np.mean(Qi_arr[-3:])
                Qc_dev = max(abs(Qc_arr[-3:] - np.mean(Qc_arr[-3:]))) * 100 / np.mean(Qc_arr[-3:])
                fr_dev = max(abs(fr_arr[-3:] - np.mean(fr_arr[-3:]))) * 100 / np.mean(fr_arr[-3:])
                Ql_dev = max(abs(Ql_arr[-3:] - np.mean(Ql_arr[-3:]))) * 100 / np.mean(Ql_arr[-3:])
                if max(Qi_dev, Qc_dev, fr_dev, Ql_dev) < max_deviation:
                    break
            k = k + 1
            fig51, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].plot(np.asarray(fr_arr) / 1e9, 'ob')
            ax[0].set_xlabel('iteration #')
            ax[0].set_ylabel('Res frequency [GHz]')
            ax[1].plot(np.asarray(Ql_arr) / 1e6, 'ob')
            ax[1].set_xlabel('iteration #')
            ax[1].set_ylabel('Ql [M]')
            ax[2].plot(np.asarray(Qi_arr) / 1e6, 'ob')
            ax[2].set_xlabel('iteration #')
            ax[2].set_ylabel('Qi [M]')
            ax[3].plot(np.asarray(Qc_arr) / 1e6, 'ob')
            ax[3].set_xlabel('iteration #')
            ax[3].set_ylabel('Qc [M]')
            fig51.tight_layout()
            plt.show(fig51)
            plt.close

        fspans.append(fspan)
        fr_final.append(fr_arr[-1])
    return fspans, fr_final

SPR_data_10_4_5 = np.array([[4.313, 0.0002396953432,	0.002140425325,	2.63E-05],
[4.907,0.0004882820426,	0.002027135994, 2.21E-05],
[5.489, 0.0005914577505,	0.002172649894,	2.73E-05],
[6.013, 0.0005316952994,	0.002105140026,	2.46E-05],
[6.635,0.0005410920877,	0.0021152657,	2.47E-05],
[7.225,0.0005155661343,	0.0020646812,	2.45E-05],
[7.682, 0.000504538763,	0.002072395612,	2.29E-05],
[8.273, 0.0004549445392,	0.002016117881,	2.01E-05]])

SPR_data_20_9 = np.array([[4.320, 0.0001990 , 0.002071, 0.000006752],
                          [4.924, 0.0004808, 0.001170, 0.00002275],
                          [5.499, 0.0003350, 0.001022, 0.00004930],
                          [6.017, 0.0003437, 0.001036, 0.00001361],
                          [6.783, 0.0003649, 0.001057, 0.00001544],
                          [7.225, 0.0002457, 0.0007674, 0.000008561],
                          [7.678, 0.0003494, 0.001032, 0.00001430],
                          [8.282, 0.0003195, 0.0009771, 0.00001249]])

SPR_data_50_23 = np.array([[4.339, 0.000086486,	0.0002510485607,	0.0000022707],
                           [4.931, 0.0001456197842,	0.0003687215063,	0.000004882],
                           [5.499, 0.0001206174246,	0.0001584221226,	0.000003635],
                           [6.014, 0.0001103260346,	0.0002954140958,	0.000003085],
                           [6.775, 0.000103370149,	0.000289315206,	0.000002846],
                           [7.233, 0.0001210098246, 0.0003112476825, 0.000003614 ],
                           [7.656, 0.0001211611753	, 0.0003145542884,0.000003708 ],
                           [8.247, 0.00007845, 0.0002202478861, 0.000001773]])

amat_participation_ratios =np.array([[4.1020e+00, 3.2300e-04, 2.2561e-04, 1.3310e-05],
       [4.7110e+00, 2.9834e-04, 2.0806e-04, 1.1440e-05],
       [5.2450e+00, 4.1426e-04, 2.9928e-04, 1.7640e-05],
       [5.7520e+00, 3.3651e-04, 2.3060e-04, 1.3320e-05],
       [6.3810e+00, 3.3057e-04, 2.2731e-04, 1.2440e-05],
       [6.6970e+00, 3.1756e-04, 2.2229e-04, 1.2050e-05],
       [7.3790e+00, 3.3244e-04, 2.3311e-04, 1.2870e-05],
       [7.9610e+00, 4.2642e-04, 3.3598e-04, 1.9390e-05]])


def add_spr_v1(filename, SPR_data, tol=0.25):

    with h5py.File(filename, "r") as f:
        list1 = f["res_freqs"][:]    # 1D numpy array
        list1= np.nanmean(list1, axis=1)
        list1= np.divide(list1, 1e9)
    
    
    p_MS_ar, p_SA_ar , p_MA_ar, p_surf  = [], [], [], []
    list1_value, matched_ref = [], []
    
    ref_col = SPR_data[:, 0]
    ref_col
    ratios = SPR_data[:, 1:4]  # shape (N, 3)
    
    for val in list1:
        mask = np.abs(ref_col - val) <= tol
        if np.any(mask):
            matched = ratios[mask]            # (K, 3)
            p_MS_ar.extend(matched[:, 0])
            p_SA_ar.extend(matched[:, 1])
            p_MA_ar.extend(matched[:, 2])
            p_surf.extend(matched.sum(axis=1))
            list1_value.extend([val] * matched.shape[0])
            matched_ref.extend(ref_col[mask])
    
    # 3) Append into the HDF5 file (create datasets if needed)
    if len(p_MS_ar) == 0:
        print("No matches found; nothing appended.")
    else:
        with h5py.File(filename, "a") as f:
            f.create_dataset("p_MS", data=p_MS_ar)
            f.create_dataset("p_SA", data=p_SA_ar)
            f.create_dataset("p_MA", data=p_MA_ar)
            f.create_dataset("p_surf", data = p_surf)

def save_results_to_hdf5_new_design(file_for_saving, powers, fr, Qi, Qc, av_n, fr_err, Qi_err, Qc_err, R2, SPR_data, tol, geom = None):
    if not os.path.exists(file_for_saving):
        file = h5py.File(file_for_saving, 'w')
        try:
            file.create_dataset('res_index', data=list(range(len(powers))))
            file.create_dataset('powers', data=powers)
            file.create_dataset('res_freqs', data=fr)
            file.create_dataset('Qis', data=Qi)
            file.create_dataset('Qcs', data=Qc)
            file.create_dataset('av_photon', data=av_n)
            file.create_dataset('fr_err', data=fr_err)
            file.create_dataset('Qi_err', data=Qi_err)
            file.create_dataset('Qc_err', data=Qc_err)
            file.create_dataset('R2', data=list(R2))
            file.close()
            add_spr_v1(file_for_saving, SPR_data, tol)
        except:
            file.close()
            print('Could not write into the file!')
    else:
        print('File already exists! Delete the old file.')