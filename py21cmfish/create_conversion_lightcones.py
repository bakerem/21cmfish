import os
import sys
sys.path.append("../")

import numpy as np
from astropy.cosmology import Planck18
import py21cmfast as p21c
from scipy.spatial.transform import Rotation
from astropy import units as un

# WDIR = os.environ['DM21CM_DIR']
# sys.path.append(WDIR)
# from dm21cm.evolve import evolve

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cms
import matplotlib.pylab as pylab

from itertools import combinations
from powerbox.tools import get_power
import physics as phys
import pickle
import h5py

from astropy.cosmology import z_at_value
from tqdm import tqdm

import scipy.interpolate as interpolate
# plt.style.use('science')
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.stats.sampling as sampling
import scipy.fft as fft

from powerbox import PowerBox

import mcfit
from py21cmfast_tools import calculate_ps 
import itertools
import cosmotile

import treecorr

import hankl
import hankel
from plot_params import params
from joblib import Parallel, delayed

pylab.rcParams.update(params)

cols_default = plt.rcParams['axes.prop_cycle'].by_key()['color']

run_name = 'z5.0_HIIDIM=128_BOXLEN=256_fisher_fid_r12345'
cache_name = f'/projectnb/darkcosmo/dark_photon_project/21cmfast_cache'
z_start = 45
z_end = 5
rs_max = 50

lc_file  =  h5py.File(f'{cache_name}/LightCone_{run_name}.h5', 'r')
    # Access data within the file
lc = lc_file["lightcones"]
 
with open(f'{cache_name}/lightconer.pkl', 'rb') as f:
    ang_lcn = pickle.load(f)
user_params = dict(lc_file["user_params"].attrs)
cosmo_params = p21c.CosmoParams(
        OMm = Planck18.Om0,
        OMb = Planck18.Ob0,
        POWER_INDEX = Planck18.meta['n'],
        SIGMA_8 = Planck18.meta['sigma8'],
        hlittle = Planck18.h,
)
cosmo = cosmo_params.cosmo

box_len = user_params["BOX_LEN"] 
hii_dim = user_params["HII_DIM"] * 3
rs_dim = np.array(lc["brightness_temp"]).shape[-1]

# creates an array of redshifts for use later

bt = np.flip(np.array(lc["brightness_temp"]).reshape((hii_dim, hii_dim, rs_dim)), axis=2)
# rs_array = np.flipud(np.linspace(z_end, z_start, rs_dim-147))

box_size_radians = user_params["BOX_LEN"] / cosmo.comoving_distance(z_end).value
lon = np.linspace(0, box_size_radians, hii_dim)
lat = np.linspace(0, box_size_radians, hii_dim)[::-1]  # This makes the X-values increasing from 0.
H, D = np.meshgrid(lat, ang_lcn.lc_distances.value)

rgrid = np.arange(0, D.max(), user_params["BOX_LEN"])
rgrid = rgrid[rgrid > D.min()]
rgrid = np.append(rgrid, D.max())
rs_array = np.flipud([z_at_value(cosmo.comoving_distance, d*un.Mpc).value for d in ang_lcn.lc_distances.value])

# mA = 6e-13 #1.15163687e-13
# epsilon = 8.721557721439982e-07  #4.1215707714155836e-7
Tgamma0 = phys.TCMB(1) * 1000 / phys.kB #mK
omega0 = 5.904e-6 * (2*np.pi) # eV


# process and reshape data from the lightcone for further use

xe_box = np.flip(np.array(lc['x_e_box']).reshape((hii_dim, hii_dim, rs_dim)), axis=2)
density_box = np.flip(np.array(lc["density"]).reshape((hii_dim , hii_dim, rs_dim)), axis=2)
Ts_box =  np.flip(np.array(lc["Ts_box"]).reshape((hii_dim , hii_dim, rs_dim)), axis=2)
n_e = xe_box *  phys.nB * (1+rs_array)**3 * (1+density_box) * (1-3*phys.YHe/4)

# calculate plasma mass and related quantities that are relevant for conversions
mgamma = phys.m_gamma(n_e)
mgamma2 = mgamma**2
logm_gamma2 = np.log(mgamma**2)
dlogmgamma2dz = np.abs(np.diff(logm_gamma2, axis=2, prepend=0)/np.diff(rs_array, prepend=1))
tau21 = 9.85e-3 * phys.TCMB(rs_array) / phys.kB / Ts_box  * phys.omega_baryon * phys.h / 0.0327 * (phys.omega_m / 0.307)**-0.5 * ((1+rs_array)/10)**0.5 
tau21[..., rs_array > rs_max] = np.zeros(xe_box[...,rs_array > rs_max].shape)
xobs = 0.0251 /(1+rs_array) 
xobs_array = xobs * np.ones((hii_dim, hii_dim, rs_dim))



# loop over redshifts and compute where conversions occur
def compute_crossings(mA):
    mA_string = f"{mA:.3e}"
    # initialize empty arrays 
    Pgammatot = Pgamma_i = num_crossings = np.zeros((hii_dim, hii_dim, rs_dim))
    num_crossings = np.zeros((hii_dim, hii_dim, rs_dim), dtype=int)
    cut_rs_dim = rs_array[rs_array<rs_max].shape[0]

    # brightness temperature array that will be modified by the conversions
    bt_w_convs = bt
    for k, z in enumerate(rs_array):
        if z<rs_array[1]:
            crossed_cond_array = np.where(((mgamma[...,k-1] < mA) & (mgamma[...,k] >= mA)) | ((mgamma[...,k-1] > mA) & (mgamma[...,k] <= mA)),1,0)
            num_crossings[...,k] = crossed_cond_array
            # indexed backwards so :k means every redshift above this zi
            Pgamma_i[...,k] = np.pi * mA**2 / ( phys.TCMB(1) * phys.hubble(1+z) * 6.57895e-16 * (1+z)**2) * dlogmgamma2dz[...,k]**(-1) * crossed_cond_array 

    Ptot = np.sum(Pgamma_i[...,rs_array<rs_max])
    Pgammatot = np.tile(np.sum(Pgamma_i[...,rs_array<rs_max], axis=2)[:,:,None], (1, 1, cut_rs_dim)) / (xobs_array[...,rs_array<rs_max])
    # Pgamma_fromzobs = np.cumsum(Pgamma_i[...,rs_array<rs_max], axis=2) / xobs_array[...,rs_array<rs_max] 
    # Pgamma0_tozobs = np.cumsum(np.flip(Pgamma_i[...,rs_array<rs_max], axis=2), axis=2) / xobs_array[...,rs_array<rs_max] 
    # Pgamma0_tozobs = np.cumsum(Pgamma_i[...,::-1], axis=2)[...,::-1] / xobs_array[...] 


    # add these probabilities to the brightness temperature
    # bt_w_convs[...,rs_array<rs_max] += Tgamma0 * Pgammatot * tau21[...,rs_array<rs_max]
    # bt_w_convs[...,rs_array<rs_max] += Tgamma0 * Pgamma_fromzobs * Pgamma0_tozobs * np.exp(-tau21[...,rs_array<rs_max])
    # bt_w_convs[...,rs_array<rs_max] -= bt[...,rs_array<rs_max] * Pgamma0_tozobs
    bt_w_convs[...,rs_array<rs_max] -= Tgamma0 * Pgammatot
    np.save(f"{cache_name}/TgammatoA_mA{mA_string}.npy", Tgamma0 * Pgammatot)
    return Tgamma0 * Pgammatot

# mA_list = np.geomspace(1.5e-14, 1e-13, 15)
# for mA in tqdm(mA_list):
#     compute_crossings(mA) 




