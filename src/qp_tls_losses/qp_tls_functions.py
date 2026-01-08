import math
import numpy as np
from scipy.constants import hbar, k
from scipy.special import k0


def tls_loss(n,T,Q0,b2,nc,f):
    return np.tanh(hbar * 2 * np.pi * f / 2 / k / T) / Q0 / np.sqrt(1+(n/nc)**b2*np.tanh(hbar*2*np.pi*f/2/k/T))


def qp_loss(t,a0,sat_temp,n,tc,f):
    return np.sinh(hbar*2*np.pi*f/2/k/t_qp_t(t,sat_temp,n))*k0(hbar*2*np.pi*f/2/k/t_qp_t(t,sat_temp,n))/a0/np.exp(1.764*tc/t_qp_t(t,sat_temp,n))


def t_qp_approx(x, tr_temp, sat_temp, s):
    return sat_temp + s * np.log1p(np.exp((x - tr_temp)/s))


def t_qp_t(x,sat_temp,n):
    return (x**n+sat_temp**n)**(1/n)
