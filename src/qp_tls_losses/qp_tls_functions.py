import math
import numpy as np
from scipy.constants import hbar, k
from scipy.special import k0

def tls_loss(n, T, Q0, b1, b2, D, f):
    return math.tanh(hbar * 2 * np.pi * f / 2 / k / T) / Q0 / np.sqrt(1+(n**b2/D/T**b1)*math.tanh(hbar*2*np.pi*f/2/k/T))

def qp_loss(T,A0,Tc,f):
    return math.sinh(hbar*2*np.pi*f/2/k/T)*k0(hbar*2*np.pi*f/2/k/T)/A0/np.exp(1.764*Tc/T)

def T_QP_approx(x, tr_temp, sat_temp, s):
    return sat_temp + s * np.log1p(np.exp((x - tr_temp)/s))
