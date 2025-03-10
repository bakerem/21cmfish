import os
import sys
sys.path.append("../")

from webbrowser import get
import py21cmfast as p21c
import os
import numpy as np
import glob
import pickle
import scipy.optimize
# from py21cmfish.create_conversion_lightcones import compute_crossings
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


from .power_spectra import *
from .io import *


class Parameter(object):
    """Class for creating derivatives given 21cm parameters"""

    def __init__(
        self,
        param,
        HII_DIM=200,
        HII_interp_factor=1,
        BOX_LEN=400,
        min_redshift=5.0,
        n_chunks=24,
        k_PEAK_order=2.0,
        mA=0.0,
        epsilon4step=0.0,
        crosszs=None,
        halo_angles=None,
        compute_2d_PS=False,
        output_dir=base_path + "examples/data/",
        PS_err_dir=base_path + "examples/data/21cmSense_noise/21cmSense_fid_EOS21/",
        Park19=None,
        k_HERA=True,
        cosmology="CDM",
        clobber=False,
        new=False,
        fid_only=False,
        vb=True,
    ):
        """

        Parameters
        ----------
        param : str
            Name of parameter (must be the same as 21cmFAST AstroParam)

        HII_DIM : int
            TODO

        BOX_LEN : int
            TODO

        Park19 : None, str
            Use Park+19 z bins when calculating power spectra
            https://ui.adsabs.harvard.edu/abs/2019MNRAS.484..933P/abstract
            'approx' = use our approximation of Park19 noise bins
            'real' = use noise from Jaehong

        k_HERA : bool
            If True (default), use HERA k bins for PS

        cosmology : str
            Label for cosmology, will be dict key for PS etc, default 'CDM'

        PS_err_dir : str
            Directory for errors, if making derivatives, assumes this exists

        clobber : bool
            If False - if paths for .npy file exist, load them

        new : bool
            If True, make new GS and PS files from lightcones

        vb : bool
            Verbose?
        """
        self.vb = vb

        self.param = param
        print("########### fisher set up for", self.param)

        self.k_PEAK_order = k_PEAK_order
        if self.param == "k_PEAK":
            self.param_21cmfast = "log10_k_PEAK"
            self.fid_i = 0
            if self.vb:
                print(f"    param = k_PEAK^-{self.k_PEAK_order}")
        elif self.param == "EPSILON4":
            self.param_21cmfast = None
            self.fid_i = 1
        else:
            self.param_21cmfast = self.param
            self.fid_i = 0

        self.param_label = self.param
        if "L_X" in self.param or "F" in self.param or self.param == "M_TURN":
            self.param_label = "log10 " + self.param

        self.output_dir = output_dir
        self.PS_err_dir = PS_err_dir

        self.HII_DIM = HII_DIM
        self.true_HII_DIM = HII_interp_factor * self.HII_DIM
        self.BOX_LEN = BOX_LEN
        self.min_redshift = min_redshift
        self.cosmology = cosmology

        self.lightcones = None

        with open(f"{output_dir}lightconer.pkl", "rb") as f:
            self.ang_lcn = pickle.load(f)

        self.halo_angles = halo_angles
        self.epsilon4step = epsilon4step
        self.mA = mA
        self.h = self.ang_lcn.cosmo.H(0).value / 100

        if compute_2d_PS and crosszs is None:
            raise ValueError("Need zs to compute 2D cross PS")
        
        if crosszs is not None:
            self.crossz1, self.crossz2 = crosszs
            print("   Computing cross correlation")

        self.fid_only = fid_only

        # Lightcone node redshifts (for global signal etc)
        self.redshifts = None
        self.redshifts_file = f"{self.output_dir}redshifts.npy"
        if os.path.exists(self.redshifts_file):
            self.redshifts = np.load(self.redshifts_file, allow_pickle=True)
            if self.vb:
                print("    Loaded redshifts")

        # All lightcone redshifts
        self.lc_redshifts = None
        self.lc_redshifts_file = f"{self.output_dir}lc_redshifts.npy"
        if os.path.exists(self.lc_redshifts_file):
            self.lc_redshifts = np.load(self.lc_redshifts_file, allow_pickle=True)
            if self.vb:
                print("    Loaded redshifts")

        self.T = None
        if param == "EPSILON4":
            self.T_file = f"{self.output_dir}global_signal_dict_{self.param}_mA={self.mA:.3e}.npy"
        else:
            self.T_file = f"{self.output_dir}global_signal_dict_{self.param}.npy"
        if os.path.exists(self.T_file) and clobber is False:
            self.T = np.load(self.T_file, allow_pickle=True).item()
            if self.vb:
                print("    Loaded T(z) from", self.T_file)

        self.theta_params = None
        if param == "EPSILON4":
            self.theta_file = f"{self.output_dir}params_dict_{self.param}_mA={self.mA:.3e}.npy"
        else:
            self.theta_file = f"{self.output_dir}params_dict_{self.param}.npy"

        if os.path.exists(self.theta_file) and clobber is False:
            self.theta_params = np.load(self.theta_file, allow_pickle=True).item()
            if self.vb:
                print("    Loaded param values from", self.theta_file)

        # define redshift bins
        if self.Park19:
            self.chunk_z_list_HERA = [
                27.15742, 22.97586, 19.66073, 16.98822, 14.80234,
                12.99172, 11.4751, 10.19206, 9.09696, 8.15475,
                7.33818, 6.62582, 6.0006,
            ]
        else:
            self.chunk_z_list_HERA = [
                27.4, 23.4828, 20.5152, 18.1892, 16.3171,
                14.7778, 13.4898, 12.3962, 11.4561, 10.6393,
                9.92308, 9.28986, 8.72603, 8.22078, 7.76543,
                7.35294, 6.97753, 6.63441, 6.31959, 6.0297,
                5.7619, 5.51376, 5.28319, 5.06838,
            ]  # , 4.86777, 4.68]

        # lightcone chunks for PS
        self.n_chunks = n_chunks

        # Use HERA k bins to calculate all PS
        self.k_HERA = k_HERA

        # Power spectrum
        self.PS = None
        self.Park19 = Park19
        self.PS_suffix = ""
        if self.Park19 is not None:
            self.PS_suffix = "_Park19"

        # files to save
        self.PS_file = (
            f"{self.output_dir}power_spectrum_dict_{self.param}{self.PS_suffix}.npy"
        )
        if param=="EPSILON4":
            self.T_file = f"{self.output_dir}global_signal_dict_{self.param}_mA={self.mA:.3e}.npy"
            self.theta_file = f"{self.output_dir}params_dict_{self.param}_mA={self.mA:.3e}.npy"
            self.PS_file = (
            f"{self.output_dir}power_spectrum_dict_{self.param}_mA={self.mA:.3e}{self.PS_suffix}.npy"
        )

        
        self.PS_fid_file = f"{self.output_dir}power_spectrum_fid_21cmsense.npy"
        self.PS_z_HERA_file = f"{self.output_dir}PS_z_HERA{self.PS_suffix}.npy"
        self.PS_z_HERA = None
        self.deriv_GS = None
        self.deriv_PS = None

        # New fiducial
        if new and self.fid_only:
            print(
                "    Fiducial only, get global signal and power spectra from lightcones"
            )

            # Load lightcones
            self.get_lightcones()

            # Make global signal and derivatives
            self.get_global_signal()

            # Load 21cmsense noise
            self.load_21cmsense(Park19=Park19)

            # Make power spectrum, load PS noise and make derivatives
            if self.k_HERA:
                k_min, k_max, n_psbins = self.get_HERA_k_bins_for_PS()
                self.get_power_spectra(k_min=k_min, k_max=k_max, n_psbins=n_psbins)
            else:
                self.get_power_spectra()

            self.make_PS_fid_HERA_grid()

        # New all parameters
        elif new:
            print(
                "    New parameter, making new global signal, \
                    power spectra and derivs from lightcones"
            )
            # Load lightcones
            self.get_lightcones()

            # Make global signal and derivatives
            self.get_global_signal()
            self.derivative_global_signal()

            # Load 21cmsense noise
            if compute_2d_PS:
                self.load_2d_21cmsense(Park19=Park19)

                # Make power spectrum, load PS noise and make derivatives
                if self.k_HERA:
                    k_min, k_max, n_psbins = self.get_HERA_k_bins_for_PS(plot=True)
                    self.get_2d_cross_power_spectra(k_min=k_min, k_max=k_max, n_psbins=n_psbins)
                else:
                    raise ValueError("k_HERA must be True for 2D PS")

                self.make_2d_cross_PS_fid_HERA_grid()
                self.derivative_2d_cross_power_spectrum()

            else:
                self.load_21cmsense(Park19=Park19)

                # Make power spectrum, load PS noise and make derivatives
                if self.k_HERA:
                    k_min, k_max, n_psbins = self.get_HERA_k_bins_for_PS(plot=True)
                    if crosszs is not None:
                        self.get_cross_power_spectra(k_min=k_min, k_max=k_max, n_psbins=n_psbins)
                    else: 
                        self.get_power_spectra(k_min=k_min, k_max=k_max, n_psbins=n_psbins)
                        
                else:
                    if crosszs is not None:
                        self.get_cross_power_spectra(k_min=k_min, k_max=k_max, n_psbins=n_psbins)
                    else:
                        self.get_power_spectra()

                if crosszs is not None:
                    self.make_cross_PS_fid_HERA_grid()
                    self.derivative_cross_power_spectrum()
                else:
                    self.make_PS_fid_HERA_grid()
                    self.derivative_power_spectrum()

        else:
            print("    Loading global signal and power spectra from saved files")
            
            # PS
            if os.path.exists(self.PS_file):
                self.PS = np.load(self.PS_file, allow_pickle=True).item()
                if self.vb:
                    print("    Loaded PS from", self.PS_file)

            if os.path.exists(self.PS_fid_file):
                self.PS_fid = np.load(self.PS_fid_file, allow_pickle=True)
                if self.vb:
                    print(
                        f"    Loaded fiducial PS from {self.PS_fid_file}, \
                            shape:{self.PS_fid.shape}"
                    )

            # PS in HERA bins
            if os.path.exists(self.PS_z_HERA_file):
                self.PS_z_HERA = np.load(self.PS_z_HERA_file, allow_pickle=True)
                self.load_21cmsense(Park19=self.Park19)
                if self.vb:
                    print(
                        "    Loaded PS_z_HERA from",
                        self.PS_z_HERA_file,
                        "shape=",
                        self.PS_z_HERA.shape,
                    )                

            # Derivatives
            if os.path.exists(self.T_file.replace("dict", "deriv_dict")):
                self.deriv_GS = np.load(
                    self.T_file.replace("dict", "deriv_dict"), allow_pickle=True
                ).item()
                if self.vb:
                    print(
                        "    Loaded GS derivatives from",
                        self.T_file.replace("dict", "deriv_dict"),
                    )
            else:
                if self.vb:
                    "    Could not load GS derivatives, remaking them"
                self.derivative_global_signal()


            if os.path.exists(self.PS_file.replace("dict", "deriv_dict")):
                self.deriv_PS = np.load(
                    self.PS_file.replace("dict", "deriv_dict"), allow_pickle=True
                ).item()
                keys = list(self.deriv_PS.keys())
                if self.vb:
                    print(
                        "    Loaded PS derivatives from",
                        self.PS_file.replace("dict", "deriv_dict"),
                        "shape=",
                        self.deriv_PS[keys[0]].shape,
                    )
            else:
                if self.vb:
                    "    Could not load PS derivatives, remaking them"
                self.derivative_power_spectrum()

        # Get fiducial Poisson noise
        try:
            self.load_Poisson_noise()
        except:
            print("Could not load Poisson noise, setting = None")
            self.PS_err_Poisson = None

        return

    def get_lightcones(self, regex=""):
        """
        Load lightcones and theta params
        """

        self.lightcones = []
        self.signal_lightcones = []

        suffix = (
            f"HIIDIM={self.HII_DIM}_BOXLEN={self.BOX_LEN}_fisher_*{regex}*{self.param}*"
        )
        lightcone_filename = (
            f"{self.output_dir}LightCone_z{self.min_redshift:.1f}_*{suffix}.h5"
        )
        fid_filename = lightcone_filename.replace(self.param, "fid")
        if self.param == "EPSILON4":
            lightcone_filename = fid_filename

        if self.vb:
            print(f"    Searching for lightcones with name {lightcone_filename}")
        lc_files = glob.glob(lightcone_filename)
        fid_files = glob.glob(fid_filename)
        lc_files.extend(fid_files)
        if self.param == "EPSILON4":
            lc_files.extend(fid_files)
            if self.mA == None:
                return ValueError("Need to specify mA for EPSILON4")
            if os.path.exists(f"{self.output_dir}TgammatoA_mA{self.mA:.3e}.npy"):
                self.signal_lightcones.append(
                    np.load(f"{self.output_dir}TgammatoA_mA{self.mA:.3e}.npy")
                )
                print(self.signal_lightcones[0].shape)
            else:
                print("no signal found for this mA, generating one instead")
                self.signal_lightcones.append(0)
            if len(self.signal_lightcones) > 1:
                return ValueError("Too many signal lightcones")

        # Don't get minihalo param lightcones if it's not the minihalo param!
        if "MINI" not in self.param:
            lc_files = [f for f in lc_files if "MINI" not in f]

        if self.vb:
            print(f"    Found {len(lc_files)} lightcones to load")

        for f in lc_files:
            self.lightcones.append(p21c.LightCone.read(f))
            print("    Loaded lightcones", f)

        self.k_fundamental, self.k_max, self.Nk = get_k_min_max(
            lightcone=self.lightcones[0], n_chunks=self.n_chunks
        )
        np.save(self.lc_redshifts_file, self.lc_redshifts, allow_pickle=True)
        return

    def get_global_signal(self, save=True, plot=False):
        """
        Get global signal and parameters and save to a file
        """
        self.T = {}
        self.theta_params = {}
        use_ETHOS = (
            False  # TODO fix this self.lightcones[0].flag_options.pystruct['USE_ETHOS']
        )

        for lc in self.lightcones:
            if use_ETHOS:
                h_PEAK = np.round(lc.astro_params.pystruct["h_PEAK"], 1)
                key = f"h_PEAK={h_PEAK:.1f}"
            else:
                key = self.cosmology

            if key not in self.T:
                self.T[key] = []
                self.theta_params[key] = []

            if self.param != "EPSILON4":
                self.T[key].append(lc.global_brightness_temp)
                self.theta_params[key].append(
                    lc.astro_params.pystruct[self.param_21cmfast]
                )

            elif (
                self.param == "EPSILON4"
                and self.epsilon4step not in self.theta_params[key]
            ):
                bt = np.average(
                    np.array(lc.lightcones["brightness_temp"]).reshape(
                        (
                            self.true_HII_DIM,
                            self.true_HII_DIM,
                            np.array(lc.lightcones["brightness_temp"]).shape[-1],
                        )
                    ),
                    axis=(0, 1),
                )
                if type(self.signal_lightcones[0]) == int:
                    signal = 0
                else:
                    signal = np.flipud(
                        np.average(self.signal_lightcones, axis=(0, 1, 2))
                    )
                self.T[key].append(bt + np.sqrt(self.epsilon4step) * signal)
                self.T[key].append(bt)
                self.T[key].append(bt - np.sqrt(self.epsilon4step) * signal)
                self.theta_params[key].extend(
                    [-self.epsilon4step, 0, self.epsilon4step]
                )

            del lc

            if self.redshifts is None:
                self.redshifts = lc.node_redshifts
                np.save(self.redshifts_file, self.redshifts, allow_pickle=True)

        for cosmo_key in self.T:
            self.T[cosmo_key] = np.array(self.T[cosmo_key])
            self.theta_params[cosmo_key] = np.array(self.theta_params[cosmo_key])

            if self.param == "k_PEAK":
                self.theta_params[cosmo_key] = (
                    1.0 / self.theta_params[cosmo_key] ** self.k_PEAK_order
                )

            if self.param == "L_X" or "F" in self.param or self.param == "M_TURN":
                self.theta_params[cosmo_key] = np.log10(
                    self.theta_params[cosmo_key]
                )  # make L_X, F log10

            # Sort increasing theta order
            print(np.argsort(self.theta_params[cosmo_key]))
            self.T[cosmo_key] = self.T[cosmo_key][
                np.argsort(self.theta_params[cosmo_key])
            ]
            self.theta_params[cosmo_key] = self.theta_params[cosmo_key][
                np.argsort(self.theta_params[cosmo_key])
            ]

            if plot:
                if self.param == "EPSILON4":
                    plt.plot(self.lc_redshifts, self.T[cosmo_key].T)
                else:
                    plt.plot(self.redshifts, self.T[cosmo_key].T)

        if save:
            np.save(self.T_file, self.T, allow_pickle=True)
            np.save(self.theta_file, self.theta_params, allow_pickle=True)
            if self.vb:
                print(f"    saved params to {self.theta_file}")
            if self.vb:
                print(f"    saved GS to {self.T_file}")

        return

    def derivative_global_signal(self, save=True, plot=True, ax=None):
        """
        Calculate global signal derivatives
        """

        if plot and ax == None:
            fig, ax = plt.subplots()

        self.deriv_GS = {}

        for cosmo_key in self.T:

            if "h_PEAK=0.0" in cosmo_key:
                ls = "dashed"
            else:
                ls = "solid"

            if self.param == "k_PEAK":
                deriv = np.zeros(
                    (len(self.theta_params[cosmo_key]), len(self.T[cosmo_key][0]))
                )
                for j in range(len(self.theta_params[cosmo_key])):
                    if j > 0:
                        deriv[j] = (self.T[cosmo_key][j] - self.T[cosmo_key][0]) / (
                            self.theta_params[cosmo_key][j]
                            - self.theta_params[cosmo_key][0]
                        )
                        if plot:
                            ax.plot(
                                self.redshifts,
                                deriv[j],
                                lw=1,
                                ls=ls,
                                label="1/k_PEAK^%.1f < %.1e"
                                % (self.k_PEAK_order, self.theta_params[cosmo_key][j]),
                            )

                self.deriv_GS[cosmo_key] = deriv[1]

            else:
                deriv = np.gradient(
                    self.T[cosmo_key], self.theta_params[cosmo_key], axis=0
                )
                self.deriv_GS[cosmo_key] = deriv[1]

                labels = ["one-sided -", "two-sided", "one-sided +"]
                if plot:
                    if self.param == "EPSILON4":
                        for dd, d in enumerate(deriv):
                            if len(deriv) <= 3:
                                ax.plot(
                                    self.lc_redshifts, d, lw=1, ls=ls, label=labels[dd]
                                )
                            else:
                                ax.plot(self.lc_redshifts, d, lw=1, ls=ls)
                    else:
                        for dd, d in enumerate(deriv):
                            if len(deriv) <= 3:
                                ax.plot(
                                    self.redshifts, d, lw=1, ls=ls, label=labels[dd]
                                )
                            else:
                                ax.plot(self.redshifts, d, lw=1, ls=ls)

        if plot:
            ax.legend()
            ax.set_xlabel("Redshift")
            ax.set_ylabel(r"$\partial T_{21}/\partial \theta$")
            ax.set_title(self.param_label)

            fig.tight_layout()
            fig.savefig(
                self.output_dir + f"GS_deriv_{cosmo_key}_{self.param}.png",
                bbox_inches="tight",
            )

        if save:
            GS_deriv_file = self.T_file.replace("dict", "deriv_dict")
            np.save(GS_deriv_file, self.deriv_GS, allow_pickle=True)
            if self.vb:
                print(f"    saved GS derivatives to {GS_deriv_file}")

        return

    def get_HERA_k_bins_for_PS(self, plot=False):
        """
        Given k centers, find the bin edges and length to give to powerbox

        Because 21cmsense interpolates PS onto k grid based on HERA baselines etc,
        we must generate our PS and Poisson noise on that *same* k grid in order
        to get the errors for the Fisher.

        (Because if we used e.g. smaller k bins, our Poisson error would be too large)
        """
        k_centers = (
            self.PS_err[0]["k"] * self.h
        )  # in Mpc^-1 (same in all frequency bins)

        def get_HERA_k_bins(k0, k_centers):
            """
            Given k centers and the first bin edge, find the bins
            """
            k_bins_iter = np.zeros(len(k_centers) + 1)
            k_bins_iter[0] = k0

            # Predicted HERA k bins obtained iteratively
            i = 0
            for i in range(len(k_centers)):
                log_k_bin_recover = 2 * np.log(k_centers[i]) - np.log(k_bins_iter[i])
                k_bins_iter[i + 1] = np.exp(log_k_bin_recover)

            return k_bins_iter

        def minimize_get_HERA_k_bins(k0, k_centers):
            """
            Find first k bin edge so that the k bins
            are equally spaced in log space
            """
            k_bins = get_HERA_k_bins(k0, k_centers)
            return np.std(np.diff(np.log10(k_bins)))

        # Find k0
        opt = scipy.optimize.minimize(
            minimize_get_HERA_k_bins, x0=k_centers[0] / 2, args=k_centers
        )

        if plot:
            k0 = np.linspace(k_centers[0] / 2, k_centers[0] * 2, 100)
            plt.plot(k0, [minimize_get_HERA_k_bins(k, k_centers) for k in k0])
            plt.axvline(opt["x"])

        # Get the bins
        k_bins_iter = get_HERA_k_bins(k0=opt["x"], k_centers=k_centers)
        k_min, k_max, n_psbins = (
            k_bins_iter.min(),
            k_bins_iter.max(),
            len(k_bins_iter + 1),
        )

        # Predicted HERA k bins just from the min and max
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_psbins)

        # Calculate k centers, should be as close as possible to HERA k bins
        k = np.exp((np.log(k_bins[1:]) + np.log(k_bins[:-1])) / 2)
        assert (
            np.abs(k - k_centers) < 1e-5
        ).all(), "ERROR: Could not recreate HERA k bins"

        if self.vb:
            print(f"    Found k_min, k_max, n_psbins to match 21cmsense noise")

        return k_min, k_max, n_psbins

    def get_power_spectra(self, n_psbins=50, k_min=None, k_max=None, save=True):
        """
        Make 21cm power spectra from redshift chunk list (bin edges)

        Parameters
        ----------
        n_psbins : int
            Number of k bins
        k_min : float, optional
            Minimum k value for PS [in 1/Mpc]
        k_max : float, optional
            Maximum k value for PS [in 1/Mpc]
        save : bool
            Save PS to file?
        """


        if self.lightcones is None:
            self.get_lightcones()
        chunk_indices_HERA = [
            np.argmin(np.abs(self.lc_redshifts - z_HERA))
            for z_HERA in self.chunk_z_list_HERA
        ][::-1]

        if self.vb:
            print(
                f"    Making powerspectra in {len(self.chunk_z_list_HERA)} redshift chunks and {n_psbins-1} k bins"
            )

        self.PS = {}
        use_ETHOS = False  # TODO fix this #self.lightcones[0].flag_options.pystruct['USE_ETHOS']

        for lc in self.lightcones:
            if use_ETHOS:
                h_PEAK = np.round(lc.astro_params.pystruct["h_PEAK"], 1)
                key = f"h_PEAK={h_PEAK:.1f}"
            else:
                key = self.cosmology
            if self.param == "EPSILON4":
                if lc is self.lightcones[0]:
                    theta = self.epsilon4step
                elif lc is self.lightcones[1]:
                    theta = -self.epsilon4step
                else:
                    theta = 0
            else:
                theta = lc.astro_params.pystruct[self.param_21cmfast]

            if self.param == "k_PEAK":
                theta = 1.0 / theta**self.k_PEAK_order

            if "L_X" in self.param or "F" in self.param or self.param == "M_TURN":
                theta = np.log10(theta)  # make L_X, F log10

            if key not in self.PS:
                self.PS[key] = {}  ##### TODO load PS nicely

            if self.vb:
                print(f"    Getting PS for {key}, {self.param}={theta}")

            # Make PS
            if k_min is None:
                k_min = self.k_fundamental
            if k_max is None:
                k_max = self.k_max

            if self.vb:
                print(f"        - Using k:{k_min}-{k_max}")

            box_size_radians = (
                self.BOX_LEN
                / self.ang_lcn.cosmo.comoving_distance(self.min_redshift).value
            )
            self.lat = np.linspace(0, box_size_radians, self.true_HII_DIM)
            bt = np.array(lc.lightcones["brightness_temp"]).reshape(
                (
                    self.true_HII_DIM,
                    self.true_HII_DIM,
                    np.array(lc.lightcones["brightness_temp"]).shape[-1],
                )
            )

            if self.param == "EPSILON4":
                if os.path.exists(
                    f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                ):  # TODO fix hardcoded link
                    print(f"halo data found for mA={self.mA:.3e}")
                    halo_xi = np.load(
                        f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                    )
                else:
                    halo_xi = None
                    
                if type(self.signal_lightcones[0]) != int:
                    # field = bt - np.sign(theta) * np.sqrt(np.abs(theta)) * np.flip(self.signal_lightcones[0], axis=2)
                    zs, lambda_CDM_PS = (
                            powerspectra_chunks(
                                bt,
                                self.BOX_LEN,
                                self.true_HII_DIM,
                                self.lc_redshifts,
                                self.ang_lcn.lc_distances,
                                self.lat,
                                self.ang_lcn.cosmo,
                                n_psbins=n_psbins,
                                chunk_indices=chunk_indices_HERA,
                                k_min=k_min,
                                k_max=k_max,
                                halo_angles=self.halo_angles,
                                halo_xi=halo_xi,
                                epsilon4=theta,
                                remove_nans=False,
                            )
                        )
                    zs, signal_PS = (
                            powerspectra_chunks(
                                np.flip(self.signal_lightcones[0], axis=2),
                                self.BOX_LEN,
                                self.true_HII_DIM,
                                self.lc_redshifts,
                                self.ang_lcn.lc_distances,
                                self.lat,
                                self.ang_lcn.cosmo,
                                n_psbins=n_psbins,
                                chunk_indices=chunk_indices_HERA,
                                k_min=k_min,
                                k_max=k_max,
                                halo_angles=self.halo_angles,
                                halo_xi=halo_xi,
                                epsilon4=theta,
                                remove_nans=False,
                            )
                        )
                    total = []
                    for i in range(len(signal_PS)):
                        dictionary = {}
                        dictionary["k"] = lambda_CDM_PS[i]['k']
                        dictionary["delta"] = lambda_CDM_PS[i]["delta"] + theta * signal_PS[i]["delta"]
                        dictionary['err_delta'] = np.sqrt(lambda_CDM_PS[i]["variance"] + theta**2 * signal_PS[i]["variance"]) * lambda_CDM_PS[i]['k']**3 / (2 * np.pi**2)
                        total.append(dictionary)
                    self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = zs, total

                else:
                    field = bt
                    self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                        powerspectra_chunks(
                            field,
                            self.BOX_LEN,
                            self.true_HII_DIM,
                            self.lc_redshifts,
                            self.ang_lcn.lc_distances,
                            self.lat,
                            self.ang_lcn.cosmo,
                            n_psbins=n_psbins,
                            chunk_indices=chunk_indices_HERA,
                            k_min=k_min,
                            k_max=k_max,
                            halo_angles=self.halo_angles,
                            halo_xi=halo_xi,
                            epsilon4=theta,
                            remove_nans=False,
                        )
                    )
            else:
                field = bt
                halo_xi = None
                self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                    powerspectra_chunks(
                        field,
                        self.BOX_LEN,
                        self.true_HII_DIM,
                        self.lc_redshifts,
                        self.ang_lcn.lc_distances,
                        self.lat,
                        self.ang_lcn.cosmo,
                        n_psbins=n_psbins,
                        chunk_indices=chunk_indices_HERA,
                        k_min=k_min,
                        k_max=k_max,
                        halo_angles=self.halo_angles,
                        halo_xi=halo_xi,
                        epsilon4=theta,
                        remove_nans=False,
                    )
                )

            del lc

        if save:
            np.save(self.PS_file, self.PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS to {self.PS_file}")

            np.save(self.PS_z_HERA_file, self.PS_z_HERA, allow_pickle=True)
            if self.vb:
                print(f"    saved PS_z_HERA to {self.PS_z_HERA_file}")

        return

    def get_cross_power_spectra(self, n_psbins=50, k_min=None, k_max=None, save=True):
        """
        Make 21cm power cross spectra from redshift chunk list (bin edges)

        Parameters
        ----------
        n_psbins : int
            Number of k bins
        k_min : float, optional
            Minimum k value for PS [in 1/Mpc]
        k_max : float, optional
            Maximum k value for PS [in 1/Mpc]
        save : bool
            Save PS to file?
        """



        if self.lightcones is None:
            self.get_lightcones()
        chunk_indices_HERA = [
            np.argmin(np.abs(self.lc_redshifts - z_HERA))
            for z_HERA in self.chunk_z_list_HERA
        ][::-1]

        if self.vb:
            print(
                f"    Making powerspectra in {len(self.chunk_z_list_HERA)} redshift chunks and {n_psbins-1} k bins"
            )

        self.PS = {}
        use_ETHOS = False  # TODO fix this #self.lightcones[0].flag_options.pystruct['USE_ETHOS']

        for lc in self.lightcones:
            if use_ETHOS:
                h_PEAK = np.round(lc.astro_params.pystruct["h_PEAK"], 1)
                key = f"h_PEAK={h_PEAK:.1f}"
            else:
                key = self.cosmology
            if self.param == "EPSILON4":
                if lc is self.lightcones[0]:
                    theta = self.epsilon4step
                elif lc is self.lightcones[1]:
                    theta = -self.epsilon4step
                else:
                    theta = 0
            else:
                theta = lc.astro_params.pystruct[self.param_21cmfast]

            if self.param == "k_PEAK":
                theta = 1.0 / theta**self.k_PEAK_order

            if "L_X" in self.param or "F" in self.param or self.param == "M_TURN":
                theta = np.log10(theta)  # make L_X, F log10

            if key not in self.PS:
                self.PS[key] = {}  ##### TODO load PS nicely

            if self.vb:
                print(f"    Getting PS for {key}, {self.param}={theta}")

            # Make PS
            if k_min is None:
                k_min = self.k_fundamental
            if k_max is None:
                k_max = self.k_max

            if self.vb:
                print(f"        - Using k:{k_min}-{k_max}")

            box_size_radians = (
                self.BOX_LEN
                / self.ang_lcn.cosmo.comoving_distance(self.min_redshift).value
            )
            self.lat = np.linspace(0, box_size_radians, self.true_HII_DIM)
            bt = np.array(lc.lightcones["brightness_temp"]).reshape(
                (
                    self.true_HII_DIM,
                    self.true_HII_DIM,
                    np.array(lc.lightcones["brightness_temp"]).shape[-1],
                )
            )

            if self.param == "EPSILON4":
                if os.path.exists(
                    f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                ):  # TODO fix hardcoded link
                    print(f"halo data found for mA={self.mA:.3e}")
                    halo_xi = np.load(
                        f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                    )
                else:
                    halo_xi = None
                    
                field = bt - np.sign(theta) * np.sqrt(np.abs(theta)) * np.flip(self.signal_lightcones[0], axis=2)
                self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                        cross_powerspectra_chunks(
                            field,
                            self.crossz1,
                            self.crossz2,
                            self.BOX_LEN,
                            self.true_HII_DIM,
                            self.lc_redshifts,
                            self.ang_lcn.lc_distances,
                            self.lat,
                            self.ang_lcn.cosmo,
                            n_psbins=n_psbins,
                            chunk_indices=chunk_indices_HERA,
                            k_min=k_min,
                            k_max=k_max,
                            halo_angles=self.halo_angles,
                            halo_xi=halo_xi,
                            epsilon4=theta,
                            remove_nans=False,
                        )
                    )
                         
            else:
                field = bt
                halo_xi = None
                self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                    cross_powerspectra_chunks(
                        field,
                        self.crossz1,
                        self.crossz2,
                        self.BOX_LEN,
                        self.true_HII_DIM,
                        self.lc_redshifts,
                        self.ang_lcn.lc_distances,
                        self.lat,
                        self.ang_lcn.cosmo,
                        n_psbins=n_psbins,
                        chunk_indices=chunk_indices_HERA,
                        k_min=k_min,
                        k_max=k_max,
                        halo_angles=self.halo_angles,
                        halo_xi=halo_xi,
                        epsilon4=theta,
                        remove_nans=False,
                    )
                )

            del lc

        if save:
            np.save(self.PS_file, self.PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS to {self.PS_file}")

            np.save(self.PS_z_HERA_file, self.PS_z_HERA, allow_pickle=True)
            if self.vb:
                print(f"    saved PS_z_HERA to {self.PS_z_HERA_file}")

        return
    
    def get_2d_cross_power_spectra(self, n_psbins=50, k_min=None, k_max=None, save=True):
        """
        Make 21cm power cross spectra from redshift chunk list (bin edges)

        Parameters
        ----------
        n_psbins : int
            Number of k bins
        k_min : float, optional
            Minimum k value for PS [in 1/Mpc]
        k_max : float, optional
            Maximum k value for PS [in 1/Mpc]
        save : bool
            Save PS to file?
        """

        
        if self.lightcones is None:
            self.get_lightcones()
        chunk_indices_HERA = [
            np.argmin(np.abs(self.lc_redshifts - z_HERA))
            for z_HERA in self.chunk_z_list_HERA
        ][::-1]

        if self.vb:
            print(
                f"    Making powerspectra in {len(self.chunk_z_list_HERA)} redshift chunks and {n_psbins-1} k bins"
            )

        self.PS = {}
        use_ETHOS = False  # TODO fix this #self.lightcones[0].flag_options.pystruct['USE_ETHOS']

        for lc in self.lightcones:
            if use_ETHOS:
                h_PEAK = np.round(lc.astro_params.pystruct["h_PEAK"], 1)
                key = f"h_PEAK={h_PEAK:.1f}"
            else:
                key = self.cosmology
            if self.param == "EPSILON4":
                if lc is self.lightcones[0]:
                    theta = self.epsilon4step
                elif lc is self.lightcones[1]:
                    theta = -self.epsilon4step
                else:
                    theta = 0
            else:
                theta = lc.astro_params.pystruct[self.param_21cmfast]

            if self.param == "k_PEAK":
                theta = 1.0 / theta**self.k_PEAK_order

            if "L_X" in self.param or "F" in self.param or self.param == "M_TURN":
                theta = np.log10(theta)  # make L_X, F log10

            if key not in self.PS:
                self.PS[key] = {}  ##### TODO load PS nicely

            if self.vb:
                print(f"    Getting PS for {key}, {self.param}={theta}")

            # Make PS
            if k_min is None:
                k_min = self.k_fundamental
            if k_max is None:
                k_max = self.k_max

            if self.vb:
                print(f"        - Using k:{k_min}-{k_max}")

            box_size_radians = (
                self.BOX_LEN
                / self.ang_lcn.cosmo.comoving_distance(self.min_redshift).value
            )
            self.lat = np.linspace(0, box_size_radians, self.true_HII_DIM)
            bt = np.array(lc.lightcones["brightness_temp"]).reshape(
                (
                    self.true_HII_DIM,
                    self.true_HII_DIM,
                    np.array(lc.lightcones["brightness_temp"]).shape[-1],
                )
            )

            if self.param == "EPSILON4":
                if os.path.exists(
                    f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                ):  # TODO fix hardcoded link
                    print(f"halo data found for mA={self.mA:.3e}")
                    halo_xi = np.load(
                        f"{self.output_dir}halo_data/mccarthy_xis/halo_xi_mA_{self.mA:.3e}_l_max35000.npy"
                    )
                else:
                    halo_xi = None
                    
                field = bt - np.sign(theta) * np.sqrt(np.abs(theta)) * np.flip(self.signal_lightcones[0], axis=2)
                self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                        cross_powerspectra_2d_chunks(
                            field,
                            self.crossz1,
                            self.crossz2,
                            self.BOX_LEN,
                            self.true_HII_DIM,
                            self.lc_redshifts,
                            self.ang_lcn.lc_distances,
                            self.lat,
                            self.ang_lcn.cosmo,
                            n_psbins=n_psbins,
                            chunk_indices=chunk_indices_HERA,
                            k_min=k_min,
                            k_max=k_max,
                            halo_angles=self.halo_angles,
                            halo_xi=halo_xi,
                            epsilon4=theta,
                            remove_nans=False,
                        )
                    )
                         
            else:
                field = bt
                halo_xi = None
                self.PS_z_HERA, self.PS[key][f"{self.param}={theta}, mA={self.mA}"] = (
                    cross_powerspectra_2d_chunks(
                        field,
                        self.crossz1,
                        self.crossz2,
                        self.BOX_LEN,
                        self.true_HII_DIM,
                        self.lc_redshifts,
                        self.ang_lcn.lc_distances,
                        self.lat,
                        self.ang_lcn.cosmo,
                        n_psbins=n_psbins,
                        chunk_indices=chunk_indices_HERA,
                        k_min=k_min,
                        k_max=k_max,
                        halo_angles=self.halo_angles,
                        halo_xi=halo_xi,
                        epsilon4=theta,
                        remove_nans=False,
                    )
                )

            del lc

        if save:
            np.save(self.PS_file, self.PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS to {self.PS_file}")

            np.save(self.PS_z_HERA_file, self.PS_z_HERA, allow_pickle=True)
            if self.vb:
                print(f"    saved PS_z_HERA to {self.PS_z_HERA_file}")

        return

    def load_21cmsense(self, Park19=None):
        """
        Load 21cmsense errors from a given directory and save arrays to self

        Parameters
        ----------
        PS_err_dir : str, optional
            Directory where 21cmSense output is stored

        Park19: str, optional
            Use Park+19 z bins https://ui.adsabs.harvard.edu/abs/2019MNRAS.484..933P/abstract
            'approx' = use our approximation of Park19 noise bins
            'real' = use noise from Jaehong
        """

        if os.path.exists(self.PS_err_dir) is False:
            raise AttributeError(
                f"Path to 21cmsense errors: {self.PS_err_dir} /cannot be found"
            )
        else:
            if self.vb:
                print(f"    Loading 21cmsense errors from {self.PS_err_dir}")

        PS_err = []
        PS_sigma = []
        # PS_fid   = []
        if Park19 == "real":
            Park19_noisefiles = np.array(
                glob.glob(
                    self.PS_err_dir
                    + "../21cmSense_noise_Park19/TotalError_HERA331_PS_500Mpc_z*_1000hr.txt"
                )
            )
            PS_z_Park19 = np.array(
                [float(f.split("_z")[-1].split("_")[0]) for f in Park19_noisefiles]
            )

            Park19_noisefiles = Park19_noisefiles[
                np.argsort(PS_z_Park19)
            ]  # low z to high z
            self.PS_z_Park19 = sorted(PS_z_Park19)

            # For each z bin
            for i, noise_file in enumerate(Park19_noisefiles):
                noise = np.genfromtxt(noise_file)

                PS_dict = {
                    "k": noise[:, 0],
                    "delta": noise[:, 1],
                    "err_mod": noise[:, 1],
                    "err_opt": noise[:, 1],
                    "err_pess": noise[:, 1],
                }
                PS_err.append(PS_dict)
                PS_sigma.append(noise[:, 1])
                # PS_fid.append(noise[:,1])

        else:
            err_files = sorted(
                glob.glob(self.PS_err_dir + f"Errlist_*")
            )
            frequencies = [f.split("_")[-1].split(".txt")[0] for f in err_files][
                ::-1
            ]  # sort in decreasing frequency, increasing redshift
            try:
                k = np.genfromtxt(
                    self.PS_err_dir
                    + f"klist_SplitCore_HERA350.drift_mod_{freq}.txt"
                )
            except:
                k = np.genfromtxt(
                    self.PS_err_dir
                    + f"klist_ska_aaast_mod_{freq}.txt"
                )
            for freq in frequencies:
                try:
                    delta = np.genfromtxt(
                        self.PS_err_dir
                        + f"P21model_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                except:
                    delta = np.zeros_like(k)
                    if self.vb:
                        print("    No delta file found, setting = nan")
                try:
                    err_mod = np.genfromtxt(
                        self.PS_err_dir + f"Errlist_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                except:
                    err_mod = np.genfromtxt(
                        self.PS_err_dir + f"Errlist_ska_aaast_mod_{freq}.txt"
                    )
                    if self.vb:
                        print("    Using SKA error data")
                try:
                    err_opt = np.genfromtxt(
                        self.PS_err_dir
                        + f"Errlist_SplitCore_HERA350.drift_opt_{freq}.txt"
                    )
                    err_pess = np.genfromtxt(
                        self.PS_err_dir
                        + f"Errlist_SplitCore_HERA350.drift_pess_{freq}.txt"
                    )
                except:
                    err_opt = 0.0
                    err_pess = 0.0

                PS_dict = {
                    "k": k,
                    "delta": delta,
                    "err_mod": err_mod,
                    "err_opt": err_opt,
                    "err_pess": err_pess,
                }
                PS_err.append(PS_dict)
                PS_sigma.append(err_mod)
                # PS_fid.append(delta)

        np.save(f"{self.output_dir}21cmSense_PS_dict.npy", PS_err, allow_pickle=True)

        self.PS_err = np.array(PS_err)
        self.PS_sigma = np.array(PS_sigma)
        # self.PS_fid   = np.array(PS_fid)

        return
    
    def load_2d_21cmsense(self, Park19=None):
        """
        Load 21cmsense errors from a given directory and save arrays to self

        Parameters
        ----------
        PS_err_dir : str, optional
            Directory where 21cmSense output is stored

        Park19: str, optional
            Use Park+19 z bins https://ui.adsabs.harvard.edu/abs/2019MNRAS.484..933P/abstract
            'approx' = use our approximation of Park19 noise bins
            'real' = use noise from Jaehong
        """

        if os.path.exists(self.PS_err_dir) is False:
            raise AttributeError(
                f"Path to 21cmsense errors: {self.PS_err_dir} /cannot be found"
            )
        else:
            if self.vb:
                print(f"    Loading 21cmsense errors from {self.PS_err_dir}")

        PS_err = []
        PS_sigma = []
        # PS_fid   = []
        if Park19 == "real":
            Park19_noisefiles = np.array(
                glob.glob(
                    self.PS_err_dir
                    + "../21cmSense_noise_Park19/TotalError_HERA331_PS_500Mpc_z*_1000hr.txt"
                )
            )
            PS_z_Park19 = np.array(
                [float(f.split("_z")[-1].split("_")[0]) for f in Park19_noisefiles]
            )

            Park19_noisefiles = Park19_noisefiles[
                np.argsort(PS_z_Park19)
            ]  # low z to high z
            self.PS_z_Park19 = sorted(PS_z_Park19)

            # For each z bin
            for i, noise_file in enumerate(Park19_noisefiles):
                noise = np.genfromtxt(noise_file)

                PS_dict = {
                    "k": noise[:, 0],
                    "delta": noise[:, 1],
                    "err_mod": noise[:, 1],
                    "err_opt": noise[:, 1],
                    "err_pess": noise[:, 1],
                }
                PS_err.append(PS_dict)
                PS_sigma.append(noise[:, 1])
                # PS_fid.append(noise[:,1])

        else:
            err_files = sorted(
                glob.glob(self.PS_err_dir + f"Errlist_*")
            )
            frequencies = [f.split("_")[-1].split(".txt")[0] for f in err_files][
                ::-1
            ]  # sort in decreasing frequency, increasing redshift
            for freq in frequencies:
                # generate k arrays
                try:
                    kpar = np.genfromtxt(
                        self.PS_err_dir
                        + f"kparlist_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                    kperp = np.genfromtxt(
                        self.PS_err_dir
                        + f"kperplist_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                    k = np.sqrt(kpar[None, :]**2 + kperp[:, None]**2)
                except:
                    k = np.genfromtxt(
                        self.PS_err_dir
                        + f"klist_ska_aaast_mod_{freq}.txt"
                    )
                    kpar = np.genfromtxt(
                        self.PS_err_dir
                        + f"kparlist_ska_aaast_mod_{freq}.txt"
                    )
                    kperp = np.genfromtxt(
                        self.PS_err_dir
                        + f"kperplist_ska_aaast_mod_{freq}.txt"
                    )
                    k = np.sqrt(kpar[None, :]**2 + kperp[:, None]**2)

                # generate power spectra
                try:
                    delta = np.genfromtxt(
                        self.PS_err_dir
                        + f"P21model_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                except:
                    delta = np.zeros_like(k)
                    if self.vb:
                        print("    No delta file found, setting = nan")

                # generate errors
                try:
                    err_mod = np.genfromtxt(
                        self.PS_err_dir + f"Errlist_SplitCore_HERA350.drift_mod_{freq}.txt"
                    )
                except:
                    err_mod = np.genfromtxt(
                        self.PS_err_dir + f"Errlist_ska_aaast_mod_{freq}.txt"
                    )
                    if self.vb:
                        print("    Using SKA error data")
                try:
                    err_opt = np.genfromtxt(
                        self.PS_err_dir
                        + f"Errlist_SplitCore_HERA350.drift_opt_{freq}.txt"
                    )
                    err_pess = np.genfromtxt(
                        self.PS_err_dir
                        + f"Errlist_SplitCore_HERA350.drift_pess_{freq}.txt"
                    )
                except:
                    err_opt = 0.0
                    err_pess = 0.0

                PS_dict = {
                    "k": k,
                    "kperp": kperp,
                    "kpar": kpar,
                    "delta": delta,
                    "err_mod": err_mod,
                    "err_opt": err_opt,
                    "err_pess": err_pess,
                }
                PS_err.append(PS_dict)
                PS_sigma.append(err_mod)
                # PS_fid.append(delta)

        np.save(f"{self.output_dir}21cmSense_PS_dict.npy", PS_err, allow_pickle=True)

        self.PS_err = np.array(PS_err)
        self.PS_sigma = np.array(PS_sigma)
        # self.PS_fid   = np.array(PS_fid)

        return

    def make_PS_fid_HERA_grid(self):
        """
        Make fiducial PS in 21cmsense k bins [Mpc^-1]
        """
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f"    Fiducial: {fid_key}")

        # Make fiducial PS in 21cmsense k bins [Mpc^-1]
        ps_fid_all = self.PS[self.cosmology][fid_key]
        self.PS_fid = np.empty((len(self.PS_z_HERA), len(self.PS_err[0]["k"])))
        for i in range(len(self.PS_z_HERA)):
            k = ps_fid_all[i]["k"]
            PS_interp = np.interp(
                self.PS_err[i]["k"] * self.h, k, ps_fid_all[i]["delta"]
            )
            self.PS_fid[i] = PS_interp

        np.save(self.PS_fid_file, self.PS_fid, allow_pickle=True)
        if self.vb:
            print(f"    saved fiducial PS to {self.PS_fid_file}")

        return

    def make_cross_PS_fid_HERA_grid(self):
        """
        Make fiducial PS in 21cmsense k bins [Mpc^-1]
        """
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f"    Fiducial: {fid_key}")

        # Make fiducial PS in 21cmsense k bins [Mpc^-1]
        ps_fid_all = self.PS[self.cosmology][fid_key]
        self.PS_fid = np.empty((len(self.PS_z_HERA), len(self.PS_err[0]["k"])))
        k = ps_fid_all[0]["k"]
        PS_interp = np.interp(
            self.PS_err[0]["k"] * self.h, k, ps_fid_all[0]["delta"]
        )
        self.PS_fid[0] = PS_interp

        np.save(self.PS_fid_file, self.PS_fid, allow_pickle=True)
        if self.vb:
            print(f"    saved fiducial PS to {self.PS_fid_file}")

        return
    
    def make_2d_cross_PS_fid_HERA_grid(self):
        """
        Make fiducial PS in 21cmsense k bins [Mpc^-1]
        """
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f"    Fiducial: {fid_key}")

        # Make fiducial PS in 21cmsense k bins [Mpc^-1]
        ps_fid_all = self.PS[self.cosmology][fid_key]
        self.PS_fid = np.empty((len(self.PS_z_HERA), len(self.PS_err[0]["kperp"]), len(self.PS_err[0]["kpar"])))
        kpar = ps_fid_all[0]["kpar"]
        kperp = ps_fid_all[0]["kperp"]
        KPAR, KPERP = np.meshgrid(kpar, kperp, indexing="ij")
        PS_interp = interpn((self.PS_err[0]["kperp"] * self.h, self.PS_err[0]["kpar"] * self.h),
            ps_fid_all[0]["delta"],
            np.stack((KPAR.ravel(),KPERP.ravel()),axis=1),
            method="linear",
            bounds_error=False,
            fill_value=None,
        ).reshape(self.PS_err[0]["delta"].shape)
        self.PS_fid[0] = PS_interp

        np.save(self.PS_fid_file, self.PS_fid, allow_pickle=True)
        if self.vb:
            print(f"    saved fiducial PS to {self.PS_fid_file}")

        return

    def load_Poisson_noise(self):
        """
        Load Poisson noise from PS
        """
        if self.vb: print('    Loading Poisson noise for PS')

        PS_err_Poisson = []
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f'    Fiducial: {fid_key}')

        ps_fid_all = self.PS[self.cosmology][fid_key]
        for i in range(len(self.PS_z_HERA)):

            PS_err_Poisson_sim = ps_fid_all[i]['err_delta']

            k_sim = ps_fid_all[i]['k']
            k_err = self.PS_err[i]['k']*0.7 # h Mpc^-1 --> Mpc^-1

            if self.k_HERA is False: # this is a bad idea
                # interpolate onto 21cmsense k values
                PS_err_Poisson.append(np.interp(k_err, k_sim, PS_err_Poisson_sim))

            else: # this is right
                assert (np.abs(k_sim - k_err) < 1e-5).all(), "ERROR: simulated k bins do not match HERA k bins"
                PS_err_Poisson.append(PS_err_Poisson_sim)

        self.PS_err_Poisson = np.array(PS_err_Poisson)

        return
    

    def load_cross_Poisson_noise(self):
        """
        Load Poisson noise from PS
        """
        if self.vb:
            print("    Loading Poisson noise for PS")

        PS_err_Poisson = []
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f"    Fiducial: {fid_key}")

        ps_fid_all = self.PS[self.cosmology][fid_key]
        PS_err_Poisson_sim = ps_fid_all[0]["err_delta"]

        k_sim = ps_fid_all[0]["k"]
        k_err = self.PS_err[0]["k"] * self.h  # h Mpc^-1 --> Mpc^-1

        if self.k_HERA is False:  # this is a bad idea
            # interpolate onto 21cmsense k values
            PS_err_Poisson.append(np.interp(k_err, k_sim, PS_err_Poisson_sim))
        else:  # this is right
            assert (
                np.abs(k_sim - k_err) < 1e-5
            ).all(), "ERROR: simulated k bins do not match HERA k bins"
            PS_err_Poisson.append(PS_err_Poisson_sim)


        self.PS_err_Poisson = np.array(PS_err_Poisson)

        return
    
    def load_2d_Poisson_noise(self):
        """
        Load Poisson noise from PS
        """
        if self.vb:
            print("    Loading Poisson noise for PS")

        PS_err_Poisson = []
        if self.fid_only:
            fid_key = list(self.PS[self.cosmology].keys())[0]
        else:
            fid_key = sorted(list(self.PS[self.cosmology].keys()))[self.fid_i]
        print(f"    Fiducial: {fid_key}")

        ps_fid_all = self.PS[self.cosmology][fid_key]
        PS_err_Poisson_sim = ps_fid_all[0]["err_delta"]

        kpar_sim = ps_fid_all[0]["kpar"]
        kperp_sim = ps_fid_all[0]["kperp"]
        kpar_err = self.PS_err[0]["kpar"] * self.h  # h Mpc^-1 --> Mpc^-1
        kperp_err = self.PS_err[0]["kperp"] * self.h  # h Mpc^-1 --> Mpc^-1

        
        assert (
            np.abs(kpar_sim - kperp_err) < 1e-5
        ).all(), "ERROR: simulated k bins do not match HERA k bins"
        PS_err_Poisson.append(PS_err_Poisson_sim)

        self.PS_err_Poisson = np.array(PS_err_Poisson)

        return

    def derivative_power_spectrum(self, save=True, plot=True, ax=None):
        """
        Calculate power spectrum derivatives

        Parameters
        ----------
        save : bool, optional
            Save PS derivative to file?

        plot : bool, optional
            Plot PS derivatives as a function of redshift

        ax : Union[None, plt.Axes]
            Matplotlib axes to plot on. If `None`, make a new figure
            with subplots
        """

        self.deriv_PS = {}

        for cosmo_key in sorted(self.T):

            if "h_PEAK=0.0" in cosmo_key:
                ls = "dashed"
            else:
                ls = "solid"

            if plot:
                fig, ax = plt.subplots(
                    4,
                    int(np.round(len(self.PS_z_HERA) / 4, 0)),
                    sharex=True,
                    sharey=False,
                    figsize=(15, 9),
                )
                ax = ax.ravel()
                fig.suptitle(cosmo_key + "  -  " + self.param)

            self.deriv_PS[cosmo_key] = np.zeros(
                (len(self.PS_err), len(self.PS_err[0]["k"]))
            )
            # For each z bin

            for i in range(len(self.PS_z_HERA)):

                # Get PS for each theta
                theta = []
                PS = []
                for theta_key in self.PS[cosmo_key]:
                    theta.append(float(theta_key.split("=")[1].split(",")[0]))
                    PS.append(self.PS[cosmo_key][theta_key][i]["delta"])

                # # Add fiducial
                # if self.param != 'k_PEAK':
                #     theta_fid = self.theta_params[cosmo_key][1]
                # else:
                #     theta_fid = self.theta_params[cosmo_key][0]
                # theta.append(theta_fid)
                # PS.append(self.PS[cosmo_key][f'{self.param}={theta_fid}'][i]['delta'])

                k = self.PS[cosmo_key][theta_key][i]["k"]
                # Sort PS in order of increasing theta
                # [x-h, x, x+h] for two-sided Derivatives (all astro params) where x is fid
                # [x, x+h1, x+h2] for asymmetric derivatives (1/kpeak)
                theta = np.array(theta)
                PS = np.array(PS)[np.argsort(theta)]
                theta = theta[np.argsort(theta)]

                assert theta[-1] > theta[0], "thetas are not ordered correctly"

                if i == 0:
                    print("theta =", theta)

                # Calculate derivative
                if self.param == "k_PEAK":
                    deriv = np.zeros((len(theta), len(PS[0])))
                    for j in range(len(theta)):
                        if j > 0:
                            deriv[j] = (PS[j] - PS[0]) / (theta[j] - theta[0])
                            if plot:
                                ax[i].semilogx(
                                    k,
                                    deriv[j],
                                    lw=1,
                                    ls=ls,
                                    label="1/k_PEAK^%.1f < %.1e"
                                    % (self.k_PEAK_order, theta[j]),
                                )
                else:
                    deriv = np.gradient(PS, theta, axis=0)
                    assert (
                        deriv[1][~np.isnan(deriv[1])]
                        == (PS[2][~np.isnan(deriv[1])] - PS[0][~np.isnan(deriv[1])])
                        / (theta[2] - theta[0])
                    ).all(), "two-sided derivative is wrong"

                if self.k_HERA:
                    self.deriv_PS[cosmo_key][i] = deriv[1]
                else:
                    # interpolate onto k for 21cmsense in 1/Mpc [21cmsense output is in h/Mpc]
                    self.deriv_PS[cosmo_key][i] = np.interp(
                        self.PS_err[i]["k"] * self.h, k, deriv[1]
                    )

                if plot:
                    try:
                        ax[i].set_title(f"z={self.PS_z_HERA[i]:.1f}")
                    except:
                        pass

                    if self.param != "k_PEAK":
                        ax[i].semilogx(k, deriv.T, alpha=0.7)

                    ax[i].scatter(
                        self.PS_err[i]["k"] * self.h,
                        self.deriv_PS[cosmo_key][i],
                        c="k",
                        s=5,
                        ls="dashed",
                        zorder=100,
                    )

                    ax[i].set_xlabel(r"k [Mpc$^{-1}$]")
                    ax[i].set_ylabel(r"$\partial \Delta^2_{21} (mK^2)/\partial \theta$")

            if plot:
                ax[0].legend()
                fig.tight_layout()
                if self.param == "EPSILON4":
                    fig.savefig(
                        self.output_dir
                        + f"PS_deriv_{self.param}_{cosmo_key}{self.PS_suffix}_mA={self.mA:.3e}.png",
                        bbox_inches="tight",
                    )
                else:
                    fig.savefig(
                        self.output_dir
                        + f"PS_deriv_{self.param}_{cosmo_key}{self.PS_suffix}.png",
                        bbox_inches="tight",
                    )

        if save:
            PS_deriv_file = self.PS_file.replace("dict", "deriv_dict")
            np.save(PS_deriv_file, self.deriv_PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS derivatives to {PS_deriv_file}")

        return
    
    def derivative_cross_power_spectrum(self, save=True, plot=True, ax=None):
        """
        Calculate power spectrum derivatives

        Parameters
        ----------
        save : bool, optional
            Save PS derivative to file?

        plot : bool, optional
            Plot PS derivatives as a function of redshift

        ax : Union[None, plt.Axes]
            Matplotlib axes to plot on. If `None`, make a new figure
            with subplots
        """

        self.deriv_PS = {}

        for cosmo_key in sorted(self.T):

            if "h_PEAK=0.0" in cosmo_key:
                ls = "dashed"
            else:
                ls = "solid"

            if plot:
                fig, ax = plt.subplots(
                    sharex=True,
                    sharey=False,
                    figsize=(9, 9),
                )
                fig.suptitle(cosmo_key + "  -  " + self.param)

            self.deriv_PS[cosmo_key] = np.zeros(
                (len(self.PS_err), len(self.PS_err[0]["k"]))
            )
            # Get PS for each theta
            theta = []
            PS = []
            for theta_key in self.PS[cosmo_key]:
                theta.append(float(theta_key.split("=")[1].split(",")[0]))
                PS.append(self.PS[cosmo_key][theta_key][0]["delta"])

            # # Add fiducial
            # if self.param != 'k_PEAK':
            #     theta_fid = self.theta_params[cosmo_key][1]
            # else:
            #     theta_fid = self.theta_params[cosmo_key][0]
            # theta.append(theta_fid)
            # PS.append(self.PS[cosmo_key][f'{self.param}={theta_fid}'][i]['delta'])

            k = self.PS[cosmo_key][theta_key][0]["k"]
            # Sort PS in order of increasing theta
            # [x-h, x, x+h] for two-sided Derivatives (all astro params) where x is fid
            # [x, x+h1, x+h2] for asymmetric derivatives (1/kpeak)
            theta = np.array(theta)
            PS = np.array(PS)[np.argsort(theta)]
            theta = theta[np.argsort(theta)]

            assert theta[-1] > theta[0], "thetas are not ordered correctly"

            print("theta =", theta)

            # Calculate derivative
            if self.param == "k_PEAK":
                deriv = np.zeros((len(theta), len(PS[0])))
                for j in range(len(theta)):
                    if j > 0:
                        deriv[j] = (PS[j] - PS[0]) / (theta[j] - theta[0])
                        if plot:
                            ax.semilogx(
                                k,
                                deriv[j],
                                lw=1,
                                ls=ls,
                                label="1/k_PEAK^%.1f < %.1e"
                                % (self.k_PEAK_order, theta[j]),
                            )
            else:
                deriv = np.gradient(PS, theta, axis=0)
                assert (
                    deriv[1][~np.isnan(deriv[1])]
                    == (PS[2][~np.isnan(deriv[1])] - PS[0][~np.isnan(deriv[1])])
                    / (theta[2] - theta[0])
                ).all(), "two-sided derivative is wrong"

            if self.k_HERA:
                self.deriv_PS[cosmo_key][0] = deriv[1]
            else:
                # interpolate onto k for 21cmsense in 1/Mpc [21cmsense output is in h/Mpc]
                self.deriv_PS[cosmo_key][0] = np.interp(
                    self.PS_err[0]["k"] * self.h, k, deriv[1]
                )

            if plot:
                try:
                    ax.set_title(f"z={self.PS_z_HERA[0]:.1f}")
                except:
                    pass

                if self.param != "k_PEAK":
                    ax.semilogx(k, deriv.T, alpha=0.7)

                ax.scatter(
                    self.PS_err[0]["k"] * self.h,
                    self.deriv_PS[cosmo_key][0],
                    c="k",
                    s=5,
                    ls="dashed",
                    zorder=100,
                )

                ax.set_xlabel(r"k [Mpc$^{-1}$]")
                ax.set_ylabel(r"$\partial \Delta^2_{21} (mK^2)/\partial \theta$")

        if plot:
            ax.legend()
            fig.tight_layout()
            if self.param == "EPSILON4":
                fig.savefig(
                    self.output_dir
                    + f"PS_deriv_{self.param}_{cosmo_key}{self.PS_suffix}_mA={self.mA:.3e}.png",
                    bbox_inches="tight",
                )
            else:
                fig.savefig(
                    self.output_dir
                    + f"PS_deriv_{self.param}_{cosmo_key}{self.PS_suffix}.png",
                    bbox_inches="tight",
                )

        if save:
            PS_deriv_file = self.PS_file.replace("dict", "deriv_dict")
            np.save(PS_deriv_file, self.deriv_PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS derivatives to {PS_deriv_file}")

        return


    def derivative_2d_cross_power_spectrum(self, save=True, plot=True, ax=None):
        """
        Calculate power spectrum derivatives

        Parameters
        ----------
        save : bool, optional
            Save PS derivative to file?

        plot : bool, optional
            Plot PS derivatives as a function of redshift

        ax : Union[None, plt.Axes]
            Matplotlib axes to plot on. If `None`, make a new figure
            with subplots
        """

        self.deriv_PS = {}

        for cosmo_key in sorted(self.T):

            if "h_PEAK=0.0" in cosmo_key:
                ls = "dashed"
            else:
                ls = "solid"

            if plot:
                fig, ax = plt.subplots(
                    sharex=True,
                    sharey=False,
                    figsize=(9, 9),
                )
                fig.suptitle(cosmo_key + "  -  " + self.param)

            self.deriv_PS[cosmo_key] = np.zeros(
                (len(self.PS_err), len(self.PS_err[0]["k"]))
            )
            # Get PS for each theta
            theta = []
            PS = []
            for theta_key in self.PS[cosmo_key]:
                theta.append(float(theta_key.split("=")[1].split(",")[0]))
                PS.append(self.PS[cosmo_key][theta_key][0]["delta"])

            # # Add fiducial
            # if self.param != 'k_PEAK':
            #     theta_fid = self.theta_params[cosmo_key][1]
            # else:
            #     theta_fid = self.theta_params[cosmo_key][0]
            # theta.append(theta_fid)
            # PS.append(self.PS[cosmo_key][f'{self.param}={theta_fid}'][i]['delta'])

            k = self.PS[cosmo_key][theta_key][0]["k"]
            # Sort PS in order of increasing theta
            # [x-h, x, x+h] for two-sided Derivatives (all astro params) where x is fid
            # [x, x+h1, x+h2] for asymmetric derivatives (1/kpeak)
            theta = np.array(theta)
            PS = np.array(PS)[np.argsort(theta)]
            theta = theta[np.argsort(theta)]

            assert theta[-1] > theta[0], "thetas are not ordered correctly"

            print("theta =", theta)

            # Calculate derivative
            if self.param == "k_PEAK":
                deriv = np.zeros((len(theta), len(PS[0])))
                for j in range(len(theta)):
                    if j > 0:
                        deriv[j] = (PS[j] - PS[0]) / (theta[j] - theta[0])
            else:
                deriv = np.gradient(PS, theta, axis=0)
                assert (
                    deriv[1][~np.isnan(deriv[1])]
                    == (PS[2][~np.isnan(deriv[1])] - PS[0][~np.isnan(deriv[1])])
                    / (theta[2] - theta[0])
                ).all(), "two-sided derivative is wrong"

            if self.k_HERA:
                self.deriv_PS[cosmo_key][0] = deriv[1]

        if save:
            PS_deriv_file = self.PS_file.replace("dict", "deriv_dict")
            np.save(PS_deriv_file, self.deriv_PS, allow_pickle=True)
            if self.vb:
                print(f"    saved PS derivatives to {PS_deriv_file}")

        return