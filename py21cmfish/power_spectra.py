# define functions to calculate PS, following py21cmmc
import numpy as np
from powerbox.tools import get_power
import mcfit
from scipy import interpolate

# changes: nothing substantial, just changing some of these functions to better fit 
# the angular lightcone class
def get_k_min_max(lightcone, n_chunks=24):
    """
    Get the minimum and maximum k in 1/Mpc to calculate powerspectra for
    given size of box and number of chunks
    """

    BOX_LEN = lightcone.user_params.pystruct['BOX_LEN']
    HII_DIM = lightcone.user_params.pystruct['HII_DIM']

    k_fundamental = 2*np.pi/BOX_LEN*max(1,len(lightcone.lightcone_distances)/n_chunks/HII_DIM) #either kpar or kperp sets the min
    k_max         = k_fundamental * HII_DIM
    Nk            = np.floor(HII_DIM/1).astype(int)
    return k_fundamental, k_max, Nk


def compute_power(
                    box,
                    length,
                    n_psbins,
                    log_bins=True,
                    ignore_kperp_zero=True,
                    ignore_kpar_zero=False,
                    ignore_k_zero=False,
                    ):
    """
    Convenience function for computing the power spectrum of a 3D box that wraps get_power from powerbox.  This code is borrowed from the example
    notebook in the 21cmFAST documentation. get_power takes a 3D box and Fourier transforms it and then performs a spherical average in $k$-space. 
    """
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
        # bins_upto_boxlen=True,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res



def powerspectra(field, 
                 rs_array,
                 box_len,
                 hii_dim,
                 lat,
                 cosmo_params,
                 n_psbins=50, 
                 nchunks=10, 
                 min_k=0.1, 
                 max_k=1.0, 
                 logk=True):
    """
    This function wraps compute_power to compute the power spectrum of many chunks of a single
    lightcone and returns the dimensionless power spectrum. 
    """
    data = []
    n_slices = rs_array.shape[0]
    chunk_indices = list(range(0,n_slices,round(n_slices / nchunks)))

    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(n_slices-1)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        cell_size = box_len / hii_dim
        chunklen = (end - start) * cell_size
        comoving_size = np.max(lat) * cosmo_params.comoving_distance(np.average([rs_array[start], rs_array[end-1]])).value
        power, k = compute_power(
            field[:, :, start:end],
            (comoving_size, comoving_size, chunklen),
            n_psbins,
            log_bins=logk,
        )
        data.append({"k": k, "P": power, "delta": k**3 * power/ (2*np.pi**2), "chunk_indices": chunk_indices})
    return data




def powerspectra_chunks(field, 
                        box_len, 
                        hii_dim, 
                        rs_array,
                        lc_distances, 
                        lat, 
                        cosmo_params,
                        nchunks=10,
                        chunk_indices=None,
                        n_psbins=50,
                        k_min=0.1,
                        k_max=1.0,
                        logk=True,
                        model_uncertainty=0.15,
                        error_on_model=True,
                        ignore_kperp_zero=True,
                        ignore_kpar_zero=False,
                        ignore_k_zero=False,
                        remove_nans=True,
                        get_halo_PS=False,
                        halo_angles=None,
                        halo_xi=None,
                        epsilon4=0.0,
                        vb=False):

    """
    Make power spectra for given number of equally spaced redshift chunks OR list of redshift chunk lightcone indices

    Output:
        k : 1/Mpc
        delta : mK^2
        err_delta : mK^2

    TODO this isn't using k_min, k_max...
    """
    data = []

    # Create lightcone redshift chunks
    # If chunk indices not given, divide lightcone into nchunks equally spaced redshift chunks
    if chunk_indices is None:
        raise ValueError("chunk_indices must be provided") #TODO implement this
        chunk_indices = list(range(0,lightcone.n_slices,round(lightcone.n_slices / nchunks),))

        if len(chunk_indices) > nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(lightcone.n_slices)

    else:
        nchunks = len(chunk_indices) - 1

    chunk_redshift = np.zeros(nchunks)

    # Calculate PS in each redshift chunk
    for i in range(nchunks):
        if vb:
            print(f'Chunk {i}/{nchunks}...')
        start    = chunk_indices[i]
        end      = chunk_indices[i + 1]
        cell_size = box_len / hii_dim
        chunklen = (end - start) * cell_size

        chunk_chi = np.flipud(lc_distances)[start:end] 
        chunk_redshift[i] = np.median(rs_array[start:end])
        comoving_size = np.max(lat) * cosmo_params.comoving_distance(np.average([rs_array[start], rs_array[end-1]])).value
        if get_halo_PS==True:
            halo_radial_seps = chunk_chi.value * halo_angles
            dz = box_len / hii_dim
            Tgamma0 = 2.7255 * 1000  #mK
            omega0 = 5.904e-6 * (2*np.pi) # eV
            xs = np.geomspace(np.min(halo_radial_seps), np.max(halo_radial_seps), 100)
            interpolated_xi = interpolate.interp1d(halo_radial_seps, halo_xi)
            halo_ks, Perp_P = mcfit.w2C(xs, nu=0, lowring=True)(interpolated_xi(xs)* (Tgamma0 / omega0 * (1+chunk_redshift[i]))**2 * epsilon4, extrap=(True,"const"))
            Perp_P *= dz 
            halo_circ_P = np.pi * halo_ks / dz * Perp_P
            if np.all(halo_xi == 0):
                halo_circ_P = np.zeros_like(xs)

        if chunklen == 0:
            print(f'Chunk size = 0 for z = {rs_array[start]}-{rs_array[end]}')
        else:
            field_power, k, variance = compute_power(
                    field[:, :, start:end],
                    (comoving_size, comoving_size, chunklen),
                    n_psbins,
                    log_bins=logk,
                    k_min=k_min,
                    k_max=k_max,
                    ignore_kperp_zero=ignore_kperp_zero,
                    ignore_kpar_zero=ignore_kpar_zero,
                    ignore_k_zero=ignore_k_zero,)

            if remove_nans:
                power, k, variance = power[~np.isnan(power)], k[~np.isnan(power)], variance[~np.isnan(power)]
            else:
                variance[np.isnan(power)] = np.inf
            if get_halo_PS==True:
                power = interpolate.interp1(halo_ks, halo_circ_P, fill_value=0, bounds_error=False)(k) + field_power
            else:
                power = field_power
            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2),})
            # data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2), "err_delta": np.sqrt(variance) * k ** 3 / (2 * np.pi ** 2),})

    return chunk_redshift, data

