"""
Holds custom Data object, necessary functions, and constants for analysis.
"""


import numpy as np
from enterprise.signals.parameter import function
import enterprise.constants as const
from enterprise.signals.selections import Selection, by_backend
from enterprise.signals import parameter, white_signals, gp_signals, signal_base
from enterprise.pulsar import PintPulsar, FeatherPulsar
import scipy.linalg as sl
from scipy.signal.windows import tukey
import pandas as pd
from pyarrow import feather
import pickle
import json
from tqdm import tqdm
import jax
import jax.numpy as jnp
from typing import Optional

from . import utilities as utils

class Data:

    """
    Data object which stores data-based constants used in the analysis.
    
    Required Attributes
    -------------------
    name : str
        The name of the dataset.
    psrs : list
        List of enterprise.PintPulsar or FeatherPulsar objects.
    wn_dict : dict
        Dictionary of white noise parameters.
    nfreqs : int
        Number of frequency bins to use in analysis.
        Defaults to 30.
    psr_dists_dict : dict
        Dictionary of pulsar distances, uncertainties,
        and measurement method. None by default, in which
        case the distances from 'psrs' is used.
    det_window_ext_factor : float
        Factor by which to extend sampling window for
        deterministic signals to avoid Gibbs phenomena.
        Defaults to 2.
    nfreqs_det : int
        Number of frequency bins used to represent deterministic
        signals in Fourier basis. Defaults to 60.
    ecorr : bool
        Whether or not to include ECORR in the white noise model.
        Defaults to True.
    marg_tm : bool
        Whether or not to analytically marginalize over linear deviations
        to the timing model. Defaults to True.
    nfreqs_dm : int
        Shorthand for a dispersion measure (DM) basis: adds
        FourierBasis('dm', nfreqs_dm, chromatic_index=2, fref=dm_fref) to
        'extra_bases'. Defaults to 0 (no DM basis).
    dm_fref : float
        Reference radio frequency [MHz] of the DM basis. DM Fourier coefficients
        are timing delays at this radio frequency. Defaults to 1400 MHz.
    extra_bases : list of FourierBasis
        Fourier bases in addition to the achromatic pulsar noise basis 'red'
        (nfreqs bins), e.g. chromatic bases for DM variations or scattering, or a
        separate basis for a common process. Spectral models refer to a basis by
        its name. The noise-weighted inner products between every pair of bases
        (and the deterministic basis) are computed once here. Defaults to None.

    Bases
    -----
    The stochastic bases are named. 'red' is the achromatic Fourier basis with
    'nfreqs' bins on the PTA span, which the pulsar noise and (by default) the
    common process share. Each FourierBasis in 'extra_bases' adds another, on the
    PTA span, a span of its own, or each pulsar's own span (Tspan='psr'). The
    deterministic basis 'det' is built separately (see 'nfreqs_det').
    """

    def __init__(self,
                 name : str,
                 psrs : list[PintPulsar] | list[FeatherPulsar],
                 wn_dict : dict[str, float],
                 nfreqs : int = 30,
                 psr_dists_dict : Optional[dict[str, tuple]] = None,
                 det_window_ext_factor : float = 2.0,
                 nfreqs_det : int = 60,
                 ecorr : bool = True,
                 marg_tm : bool = True,
                 per_psr_data_dict_filepath : Optional[str] = None,
                 nfreqs_dm : int = 0,
                 dm_fref : float = 1400.0,
                 extra_bases : Optional[list['FourierBasis']] = None):

        self.name = name
        self.psrs = psrs
        self.wn_rn_dict = wn_dict
        self.nfreqs = nfreqs
        self.psr_dists_dict = psr_dists_dict
        self.det_window_ext_factor = det_window_ext_factor
        self.nfreqs_det = nfreqs_det
        self.ecorr = ecorr
        self.marg_tm = marg_tm
        self.dm_fref = dm_fref

        # stochastic bases besides 'red', in the order they are stacked
        extra_bases = list(extra_bases or [])
        if nfreqs_dm > 0:
            if any(b.name == 'dm' for b in extra_bases):
                raise ValueError("Give the DM basis either as nfreqs_dm or in extra_bases, not both.")
            extra_bases.append(FourierBasis('dm', nfreqs_dm, chromatic_index=2., fref=dm_fref))
        names = [b.name for b in extra_bases]
        if len(set(names)) != len(names) or set(names) & {'red', 'det'}:
            raise ValueError(f"Basis names must be unique and not 'red' or 'det', got {names}.")
        self.extra_bases = extra_bases

        # remove red noise parameters (if any) from "white noise" dictionary
        self.wn_dict = dict()
        for key, val in self.wn_rn_dict.items():
            if 'red_noise' not in key:
                self.wn_dict[key] = val

        # build per-pulsar data dictionary
        # this holds all the data per-pulsar
        if per_psr_data_dict_filepath is not None:
            self.per_psr_data_dict = load_per_psr_data_dict(per_psr_data_dict_filepath)
        else:
            self.per_psr_data_dict = self.build_per_psr_data_dict()
        
        # general PTA attributes
        self.psr_names = list(self.per_psr_data_dict.keys())
        self.npsrs = len(self.psr_names)
        self.Tspan = self.per_psr_data_dict[self.psr_names[0]]['Tspan']
        self.ncomponents = 2 * self.nfreqs
        self.freqs = jnp.arange(1, self.nfreqs + 1) / self.Tspan
        
        # pulsar sky locations
        self.psr_phi = jnp.array([self.per_psr_data_dict[name]['phi']
                                  for name in self.psr_names])
        self.psr_theta = jnp.array([self.per_psr_data_dict[name]['theta']
                                    for name in self.psr_names])
        self.psrpos = jnp.array(utils.phitheta_to_psrpos(self.psr_phi, self.psr_theta))

        # pulsar measured distances and uncertainty
        self.psr_dists_measured = jnp.array([self.per_psr_data_dict[name]['pdist']
                                             for name in self.psr_names])[:, 0]
        self.psr_dists_std = jnp.array([self.per_psr_data_dict[name]['pdist']
                                       for name in self.psr_names])[:, 1]
        self.psr_dist_method = np.array([self.per_psr_data_dict[name]['psr_dist_method']
                                          for name in self.psr_names])
        
        # # NEW STUFF FOR NONDIAGONAL PHI
        # self.Fs = [jnp.array(self.per_psr_data_dict[psrname]['F']) for psrname in self.psr_names]
        # self.toas = [jnp.array(self.per_psr_data_dict[psrname]['toas']) for psrname in self.psr_names]
        # self.min_toas_per_psr = jnp.array([self.per_psr_data_dict[psrname]['min_toa'] for psrname in self.psr_names])
        
        # constants needed for stochastic part of posterior
        self.Sigma_0_inv_jc = jnp.stack([self.per_psr_data_dict[psrname]['Sigma_inv']/utils.renorm**2
                                         for psrname in self.psr_names])
        self.Sigma_0_inv_j = jnp.array(sl.block_diag(*[(self.per_psr_data_dict[psrname]['Sigma_inv']/utils.renorm**2).astype(np.float32)
                                                       for psrname in self.psr_names]))
        self.a_hat_j = jnp.array(np.concatenate([self.per_psr_data_dict[psrname]['a_hat']
                                                 for psrname in self.psr_names]).astype(np.float32))

        self.phiinv_0_j = jnp.array(np.concatenate([self.per_psr_data_dict[psrname]['phiinv']/utils.renorm**2
                                                    for psrname in self.psr_names]).astype(np.float32))
        self.phiinv_logdet_0_j = jnp.sum(np.log(self.phiinv_0_j*utils.renorm**2))
        self.Sigma_0_logdet_j = jnp.array(np.sum([self.per_psr_data_dict[psrname]['logdet']\
                                                  for psrname in self.psr_names]))
        self.Si0_a_hat_j = jnp.dot(self.Sigma_0_inv_j, self.a_hat_j) * utils.renorm
        self.FNFs = jnp.array([self.per_psr_data_dict[psrname]['TNT'] / utils.renorm**2
                               for psrname in self.psr_names])
        self.FNrs = jnp.array([self.per_psr_data_dict[psrname]['TNr'] / utils.renorm
                               for psrname in self.psr_names])

        # Also make phiinv_0 cubed. For vmap, need it in both forms:
        # - Npsr x (nfreqs x nfreqs) and
        # - Nfreqs x (Npsr x Npsr)
        self.phiinv_0_vecs_j = jnp.stack([self.per_psr_data_dict[psrname]['phiinv']/utils.renorm**2
                                          for psrname in self.psr_names])   # npsrs x nfreqs
        self.phiinv_0_cube_pf = jnp.zeros((self.phiinv_0_vecs_j.shape[0], self.phiinv_0_vecs_j.shape[1], self.phiinv_0_vecs_j.shape[1]))
        self.phiinv_0_cube_fp = jnp.zeros((self.phiinv_0_vecs_j.shape[1], self.phiinv_0_vecs_j.shape[0], self.phiinv_0_vecs_j.shape[0]))
        self.ii_diag_pf = jnp.arange(self.phiinv_0_vecs_j.shape[1])
        self.ii_diag_fp = jnp.arange(self.phiinv_0_vecs_j.shape[0])
        self.phiinv_0_cube_pf = self.phiinv_0_cube_pf.at[:, self.ii_diag_pf, self.ii_diag_pf].set(self.phiinv_0_vecs_j)
        self.phiinv_0_cube_fp = self.phiinv_0_cube_fp.at[:, self.ii_diag_fp, self.ii_diag_fp].set(self.phiinv_0_vecs_j.T)
        self.a_hat_2d_pf = jnp.stack(([self.per_psr_data_dict[psrname]['a_hat'] * utils.renorm
                                       for psrname in self.psr_names]))  # npsrs x nfreqs
        self.Si0_a_hat_j_pf = jnp.stack([np.dot(self.per_psr_data_dict[psrname]['Sigma_inv']/utils.renorm**2,
                                        self.per_psr_data_dict[psrname]['a_hat'] * utils.renorm)
                                        for psrname in self.psr_names]) # equivalent to TNr, but may be marginalized over WN if desired

        # constants needed for deterministic part of likelihood
        self.num_coeff_det = self.per_psr_data_dict[self.psr_names[0]]['num_coeff_det']
        self.sparse_toas_det = [self.per_psr_data_dict[name]['sparse_toas_det']
                                for name in self.psr_names]
        self.sparse_toas_det_jax = jnp.array(self.sparse_toas_det)
        sparse_toas_shifted = [(sparse_toas - utils.tref) * utils.cw_renorm
                               for sparse_toas in self.sparse_toas_det]
        self.sparse_toas_shifted_scaled = jnp.asarray(sparse_toas_shifted, dtype=jnp.float32)
        self.Nsparse = self.sparse_toas_shifted_scaled.shape[1]
        self.freqs_forFFT = jnp.array([self.per_psr_data_dict[name]['freqs_forFFT']
                                       for name in self.psr_names])
        self.Tspan_ext = jnp.array([self.per_psr_data_dict[name]['Tspan_ext']
                                    for name in self.psr_names])[0]
        self.Tukey_det = jnp.array(tukey(self.Nsparse, alpha=(self.Tspan_ext - self.Tspan)/self.Tspan_ext))
        self.TDNTDs = jnp.array([self.per_psr_data_dict[name]['TDNTD']
                                 for name in self.psr_names]) / (utils.renorm**2.)
        self.TNTDs = jnp.array([self.per_psr_data_dict[name]['TNTD']
                                for name in self.psr_names]) / (utils.renorm**2.)
        self.TDNrs = jnp.array([self.per_psr_data_dict[name]['TDNr']
                                for name in self.psr_names]) / utils.renorm

        # the bases actually stored in the per-pulsar dictionary (which may come from a file)
        self.extra_bases = _extra_bases_from_dict(self.per_psr_data_dict[self.psr_names[0]])

        # dispersion measure (DM) basis arrays, kept as attributes for convenience
        self.has_dm = 'dm' in self.basis_names
        if self.has_dm:
            self.dm_fref = self.get_basis('dm').fref
            self.nfreqs_dm = self.get_basis('dm').nfreqs
            self.ncomponents_dm = 2 * self.nfreqs_dm
            self.freqs_dm = self.basis_freqs('dm')
            # F_dm^T N^-1 F_dm, shape (Npsrs, 2*Nf_dm, 2*Nf_dm)
            self.FdmNFdms = jnp.array([self.per_psr_data_dict[name]['FdmNFdm']
                                       for name in self.psr_names]) / utils.renorm**2
            # F^T N^-1 F_dm, shape (Npsrs, 2*Nf, 2*Nf_dm)
            self.FNFdms = jnp.array([self.per_psr_data_dict[name]['TNFdm']
                                     for name in self.psr_names]) / utils.renorm**2
            # F_dm^T N^-1 r, shape (Npsrs, 2*Nf_dm)
            self.FdmNrs = jnp.array([self.per_psr_data_dict[name]['FdmNr']
                                     for name in self.psr_names]) / utils.renorm
            # F_dm^T N^-1 F_D, shape (Npsrs, 2*Nf_dm, Nd)
            self.FdmNTDs = jnp.array([self.per_psr_data_dict[name]['FdmNTD']
                                      for name in self.psr_names]) / utils.renorm**2
        else:
            self.nfreqs_dm = 0
            self.ncomponents_dm = 0

        # the constants above are computed in float64, but stored in float32,
        # the precision of the posterior evaluation (see __init__.py)
        for key, value in list(vars(self).items()):
            if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.inexact):
                setattr(self, key, value.astype(jnp.complex64 if jnp.iscomplexobj(value) else jnp.float32))


    # The stochastic bases are named: 'red' (nfreqs bins, built by enterprise) plus
    # 'extra_bases'. The methods below also work on Data objects pickled before
    # 'extra_bases' existed, whose only extra basis can be the DM basis.

    @property
    def basis_names(self):
        """Names of the stochastic bases, in the order they are stacked."""
        return ['red'] + [b.name for b in self._extra_bases()]

    def _extra_bases(self):
        if 'extra_bases' in self.__dict__:
            return self.__dict__['extra_bases']
        return _extra_bases_from_dict(self.per_psr_data_dict[self.psr_names[0]])

    def get_basis(self, name):
        """The FourierBasis named 'name' ('red' or one of 'extra_bases')."""
        if name == 'red':
            return FourierBasis('red', self.nfreqs)
        for basis in self._extra_bases():
            if basis.name == name:
                return basis
        raise ValueError(f"Data object has no basis '{name}'; its bases are {self.basis_names}.")

    def basis_Tspan(self, name):
        """
        Span [s] whose harmonics the basis 'name' uses: a float, or for a basis with
        Tspan='psr' an array of shape (Npsrs,) holding each pulsar's span.
        """
        basis = self.get_basis(name)
        if basis.per_psr:
            return np.array([self.per_psr_data_dict[p]['Tspan_psr'] for p in self.psr_names])
        return float(self.Tspan) if basis.Tspan is None else basis.Tspan

    def basis_freqs(self, name):
        """
        Frequencies [Hz] k / Tspan, k = 1...nfreqs, of the basis 'name': shape
        (nfreqs,), or (Npsrs, nfreqs) for a basis with Tspan='psr'.
        """
        k = np.arange(1, self.get_basis(name).nfreqs + 1)
        Tspan = self.basis_Tspan(name)
        freqs = k / Tspan[:, None] if np.ndim(Tspan) else k / Tspan
        return jnp.asarray(freqs, dtype=jnp.float32)

    def basis_ncomponents(self, name):
        """Number of columns of the basis 'name' ('det' for the deterministic basis)."""
        return self.num_coeff_det if name == 'det' else 2 * self.get_basis(name).nfreqs

    def gram(self, psr_name, a, b):
        """
        Noise-weighted inner product F_a^T N^-1 F_b of bases 'a' and 'b'
        ('det' for the deterministic basis) for one pulsar, in float64 and the
        [ns]-based units of the stored float32 arrays (e.g. FNFs), i.e. matching
        prior covariances in [ns]^2. Falls back to the float32 arrays when no
        float64 copy exists.
        """
        order = self.basis_names + ['det']
        transpose = order.index(a) > order.index(b)
        if transpose:
            a, b = b, a
        G = self._inner_product(psr_name, _gram_key(a, b), nbasis=2)
        return G.T if transpose else G

    def basis_Nr(self, psr_name, a):
        """Noise-weighted residual inner product F_a^T N^-1 r of basis 'a' for one pulsar (units as in gram)."""
        return self._inner_product(psr_name, _nr_key(a), nbasis=1)

    def _inner_product(self, psr_name, key, nbasis):
        entry = self.per_psr_data_dict[psr_name]
        if key in entry:
            return np.asarray(entry[key], dtype=np.float64) / utils.renorm**nbasis
        if key in _LEGACY_ARRAYS:
            return np.asarray(getattr(self, _LEGACY_ARRAYS[key])[self.psr_names.index(psr_name)], dtype=np.float64)
        raise KeyError(f"No inner product '{key}' stored for {psr_name}.")


    def build_single_psr_enterprise_model(self, psr, Tspan, log10_Arn=-12, gamma=4.33):
        """
        Build single pulsar enterprise PTA object.
        Used to compute TNT, TNr, etc.

        Parameters
        ----------
        psr : enterprise.pulsar.PintPulsar or FeatherPulsar
            enterprise puslar object.
        Tspan : float
            Observation span of entire PTA, not this individual pulsar.
        log10_Arn : float
            Reference log-amplitude of power law. Used
            to regularize covariance matrices for gamma-ray PTAs.
        gamma : float
            Reference spectral index of power law. Used
            to regularize covariance matrices gamma-ray PTAs.

        Returns
        -------
        pta : enterprise.PTA
            Enterprise PTA object for single pulsar.
        """
        selection = Selection(by_backend)

        # white noise parameters (set later with dictionary)
        efac = parameter.Constant()
        t2equad = parameter.Constant()

        # red noise parameters
        rn_log10_A = parameter.Constant(log10_Arn)
        gamma = parameter.Constant(gamma)

        # signals / noise
        mn = white_signals.MeasurementNoise(efac=efac,              # type: ignore
                                            log10_t2equad=t2equad,
                                            selection=selection)
        rn_pl = powerlaw(log10_A=rn_log10_A, gamma=gamma)           # type: ignore
        rn = gp_signals.FourierBasisGP(spectrum=rn_pl, components=self.nfreqs, Tspan=Tspan)

        # single pulsar PTA model
        model = mn + rn

        # timing model
        if self.marg_tm:   # marginalize over linear deviations to the timing model parameters
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            model += tm

        # build PTA model
        if self.ecorr:
            ecorr = parameter.Constant()
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,      # type: ignore
                                                selection=selection)
            model += ec

        pta = signal_base.PTA([model(psr)])

        return pta


    def build_per_psr_data_dict(self):
        """
        Build data dictionary which stores data and
        associated objects (per pulsar) needed for analysis.

        Returns
        -------
        data_dict : dict
            Dictionary which stores pulsar data needed for analysis.
        """

        # get Tspan
        tmin = [p.toas.min() for p in self.psrs]
        tmax = [p.toas.max() for p in self.psrs]
        Tspan = np.max(tmax) - np.min(tmin)

        # save necessary data in dictionary
        data_dict = dict()

        with tqdm(range(len(self.psrs)), desc='building pulsar models') as pbar:
            for i in pbar:
                psr = self.psrs[i]
                pbar.set_postfix_str(f'running {psr.name}')

                # reference power law parameters to regularize covariance matrices
                log10_Arn = -12
                gamma = 4.33

                # number of frequency bins for pulsar noise model
                nfrequencies = self.nfreqs

                # enterprise PTA object for one pulsar
                pta_psr = self.build_single_psr_enterprise_model(psr, Tspan, log10_Arn=-12, gamma=4.33)

                # set white noise parameters
                pta_psr.set_default_params(self.wn_dict)

                # no free parameters in this model
                params = dict()

                # arrays needed for posterior evaluation
                TNT = pta_psr.get_TNT(params)[0]
                TNr = pta_psr.get_TNr(params)[0]
                phiinv = pta_psr.get_phiinv(params)[0]
                Sigma_inv = TNT + np.diag(phiinv)       # type: ignore
                Li = sl.cholesky(Sigma_inv, lower=True)
                Sigma = sl.cho_solve((Li, True), np.identity(len(Li)))
                a_hat = sl.cho_solve((Li, True), TNr)
                logdet = -2 * np.sum(np.log(np.diag(Li)))

                # save alternative basis used for deterministic signals
                # window extension for deterministic FFT (avoids Gibbs phenomena)
                window_ext = Tspan * self.det_window_ext_factor
                Tspan_ext = Tspan + 2. * window_ext
                Nf_det = self.nfreqs_det
                num_coeff_det = 2 * Nf_det

                # sparse TOAs for CW FFT
                sparse_toas_det = np.linspace(np.min(tmin) - window_ext, np.max(tmax) + window_ext,
                                            num_coeff_det + 2, endpoint=False)
                Nsparse = sparse_toas_det.shape[0]
                freqs_forFFT = np.fft.fftfreq(Nsparse, Tspan_ext / Nsparse)

                # Fourier design matrix for CW
                F_D = np.zeros((psr.toas.shape[0], num_coeff_det))
                for j in range(Nf_det):
                    F_D[:, 2 * j] = np.sin(2. * np.pi * freqs_forFFT[j + 1] * psr.toas)
                    F_D[:, 2 * j + 1] = np.cos(2. * np.pi * freqs_forFFT[j + 1] * psr.toas)

                # arrays needed for posterior evaluation with CW in model
                T = pta_psr.get_basis(params)[0]
                psr_signal_collection = pta_psr._signalcollections[0]
                ndiag = psr_signal_collection.get_ndiag(params)     # type: ignore
                res = psr_signal_collection.get_detres(params)      # type: ignore
                TDNTD = ndiag.solve(F_D, left_array=F_D)
                TNTD = ndiag.solve(F_D, left_array=T)
                TDNr = ndiag.solve(res, left_array=F_D)

                # extra bases: inner products with themselves, each other, the
                # 'red' basis T, the deterministic basis F_D, and the residuals
                extra_entries = dict(basis_specs_json=json.dumps([b.to_dict() for b in self.extra_bases]))
                stacked = [('red', T)]
                for basis in self.extra_bases:
                    F_b = basis.design_matrix(psr.toas, psr.freqs, Tspan)
                    for name_a, F_a in stacked:
                        extra_entries[_gram_key(name_a, basis.name)] = ndiag.solve(F_b, left_array=F_a)
                    extra_entries[_gram_key(basis.name, basis.name)] = ndiag.solve(F_b, left_array=F_b)
                    extra_entries[_gram_key(basis.name, 'det')] = ndiag.solve(F_D, left_array=F_b)
                    extra_entries[_nr_key(basis.name)] = ndiag.solve(res, left_array=F_b)
                    stacked.append((basis.name, F_b))
                    if basis.name == 'dm':
                        extra_entries.update(nfrequencies_dm=basis.nfreqs, dm_fref=basis.fref)

                # use parallax and DM methods for pulsar distance
                if self.psr_dists_dict is not None and psr.name in list(self.psr_dists_dict.keys()):
                    pdist = self.psr_dists_dict[psr.name]
                    psr_dist_and_uncertainty = (pdist[0], pdist[1])
                    if pdist[2] == 'PX':  # parallax distance method
                        psr_dist_method = 'PX'
                    else:  # DM distance method
                        psr_dist_method = 'DM'
                # otherwise use distance stored in pulsar objects with normal prior
                else:
                    psr_dist_and_uncertainty = (psr.pdist[0], psr.pdist[1])
                    psr_dist_method = 'other'
                            
                # store pulsar data and associated objects in dictionary
                data_dict[psr.name] = dict(
                    phi = psr.phi,
                    theta = psr.theta,
                    Tspan = Tspan,
                    Tspan_psr = psr.toas.max() - psr.toas.min(),
                    log10_Arn = log10_Arn,
                    gamma = gamma,
                    nfrequencies = nfrequencies,
                    ncomponents = 2 * nfrequencies,
                    Li = Li,
                    Sigma = Sigma,
                    Sigma_inv = Sigma_inv,
                    phiinv = phiinv,
                    a_hat = a_hat,
                    logdet = logdet,
                    TNr = TNr,
                    TNT = TNT,
                    pdist = psr_dist_and_uncertainty,
                    psr_dist_method = psr_dist_method,
                    # min_toa = np.min(toas),
                    # toas = psr.toas,
                    # residuals = psr.residuals,
                    # F = pta_psr.get_basis()[0],
                    # FD = F_D,
                    num_coeff_det = num_coeff_det,
                    TDNTD = TDNTD,
                    TNTD = TNTD,
                    TDNr = TDNr,
                    sparse_toas_det = sparse_toas_det,
                    freqs_forFFT = freqs_forFFT,
                    Tspan_ext = Tspan_ext,
                    **extra_entries,
                )

        return data_dict
    
    # save data object as pickle file
    def save_data(self, filepath=None):
        """
        Save Data object as pickle file.
        
        Parameters
        ----------
        filepath : str
            Local path to save data object.

        Returns
        -------
        None
        """
        if filepath is None:
            filepath = f'{self.name}.pkl'

        with open(filepath, 'wb') as fp:
            pickle.dump(self, fp)
        
        print(f'Saved data object to {filepath}.')




class FourierBasis:
    """
    A (possibly chromatic) Fourier basis for per-pulsar or common Gaussian processes.

    Column 2k (2k + 1) is sin (cos) of 2 pi f_k t with f_k = (k + 1) / Tspan, scaled
    per TOA by (fref / nu)^chromatic_index, where nu is the radio frequency of the TOA.
    Coefficients are therefore timing delays at the reference radio frequency, as in
    enterprise.signals.utils.createfourierdesignmatrix_dm and _chromatic.

    Parameters
    ----------
    name : str
        Name by which spectral models refer to the basis.
    nfreqs : int
        Number of frequency bins.
    chromatic_index : float
        0 for achromatic processes, 2 for dispersion measure (DM) variations,
        4 for scattering. Defaults to 0.
    fref : float
        Reference radio frequency [MHz]. Defaults to 1400 MHz.
    Tspan : float, 'psr' or None
        Span [s] whose harmonics the basis uses. 'psr' gives each pulsar the
        harmonics of its own observing span (max - min of its TOAs), as enterprise
        does by default for intrinsic pulsar noise. Pulsars then have different
        frequencies, so only per-pulsar processes (IndependentSpectralModel) in the
        'marginalized' sampling model may live on such a basis, without
        inter-frequency correlations. Defaults to None, the PTA span.
    """

    def __init__(self, name : str, nfreqs : int, chromatic_index : float = 0.,
                 fref : float = 1400., Tspan : Optional[float | str] = None):
        self.name = name
        self.nfreqs = int(nfreqs)
        self.chromatic_index = float(chromatic_index)
        self.fref = float(fref)
        if isinstance(Tspan, str) and Tspan != 'psr':
            raise ValueError(f"Tspan must be a span [s], 'psr' or None, got {Tspan!r}.")
        self.Tspan = Tspan if Tspan is None or Tspan == 'psr' else float(Tspan)

    @property
    def per_psr(self):
        """Whether each pulsar uses the harmonics of its own observing span."""
        return self.Tspan == 'psr'

    def __repr__(self):
        Tspan = repr(self.Tspan) if self.per_psr else self.Tspan
        return (f"FourierBasis('{self.name}', nfreqs={self.nfreqs}, chromatic_index={self.chromatic_index:g}, "
                f"fref={self.fref:g}, Tspan={Tspan})")

    def to_dict(self):
        return dict(name=self.name, nfreqs=self.nfreqs, chromatic_index=self.chromatic_index,
                    fref=self.fref, Tspan=self.Tspan)

    def design_matrix(self, toas, radio_freqs, Tspan):
        """
        Design matrix of shape (Ntoas, 2 * nfreqs) for one pulsar.

        Parameters
        ----------
        toas : array
            TOAs [s] of the pulsar.
        radio_freqs : array
            Radio frequencies [MHz] of the TOAs.
        Tspan : float
            Observation span [s] of the PTA, used unless the basis sets its own
            (with Tspan='psr', the span of 'toas').
        """
        toas = np.asarray(toas, dtype=np.float64)
        if self.per_psr:
            T = toas.max() - toas.min()
        else:
            T = Tspan if self.Tspan is None else self.Tspan
        f = np.arange(1, self.nfreqs + 1) / T
        phase = 2. * np.pi * toas[:, None] * f[None, :]
        F = np.zeros((toas.shape[0], 2 * self.nfreqs))
        F[:, ::2] = np.sin(phase)
        F[:, 1::2] = np.cos(phase)
        if self.chromatic_index != 0.:
            F *= ((self.fref / np.asarray(radio_freqs, dtype=np.float64)) ** self.chromatic_index)[:, None]
        return F


# names of the noise-weighted inner products stored per pulsar: those of the
# 'red', 'dm' and 'det' bases keep the names used throughout Prometheus
_GRAM_KEYS = {('red', 'red'): 'TNT', ('red', 'det'): 'TNTD', ('det', 'det'): 'TDNTD',
              ('red', 'dm'): 'TNFdm', ('dm', 'dm'): 'FdmNFdm', ('dm', 'det'): 'FdmNTD'}
_NR_KEYS = {'red': 'TNr', 'det': 'TDNr', 'dm': 'FdmNr'}
# float32 Data attributes holding the same inner products
_LEGACY_ARRAYS = {'TNT': 'FNFs', 'TNr': 'FNrs', 'TNTD': 'TNTDs', 'TDNTD': 'TDNTDs', 'TDNr': 'TDNrs',
                  'TNFdm': 'FNFdms', 'FdmNFdm': 'FdmNFdms', 'FdmNTD': 'FdmNTDs', 'FdmNr': 'FdmNrs'}


def _gram_key(a, b):
    """Per-pulsar dictionary key of F_a^T N^-1 F_b (a before b in the stacking order)."""
    return _GRAM_KEYS.get((a, b), f'{a}N{b}')


def _nr_key(a):
    """Per-pulsar dictionary key of F_a^T N^-1 r."""
    return _NR_KEYS.get(a, f'{a}Nr')


def _extra_bases_from_dict(psr_entry):
    """The extra bases stored in one pulsar's entry of the per-pulsar dictionary."""
    if 'basis_specs_json' in psr_entry:
        return [FourierBasis(**spec) for spec in json.loads(psr_entry['basis_specs_json'])]
    if 'FdmNFdm' in psr_entry:   # DM basis built before extra bases were stored by name
        return [FourierBasis('dm', int(psr_entry['nfrequencies_dm']), chromatic_index=2.,
                             fref=float(psr_entry.get('dm_fref', 1400.)))]
    return []


@function
def powerlaw(f, log10_A=-16, gamma=5, components=2):
    """
    Power law function in NumPy (as opposed to JAX).
    Used to construct reference spectra for regularizing
    convariance matrices in posterior evaluation.

    Parameters
    ----------
    f : array
        Frequencies [Hz] used in pulsar noise model.
        Each frequency must be repeated 'components' times.
    log10_A : float
        log-amplitude of power law.
    gamma : float
        spectral index of power law.
    components : int
        Number of basis functions needed for each frequency.
        (e.g. sine/cosine -> components=2)

    Returns
    -------
    pta : enterprise.PTA
        Enterprise PTA object.
    """
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    pl = (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components)
    return pl


def save_per_psr_data_dict(data_dict, filepath='data_dict.feather',):
    """
    Save data dictionary as feather file.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing necessary data for analysis.
        Output of 'build_data_dict'.
    filepath : str
        Local destination to save 'data_dict'.

    Returns
    -------
    None
    """
    # Create a list of dictionaries – each row corresponds to one pulsar.
    rows = []
    for pulsar, params in data_dict.items():
        row = {"pulsar": pulsar}
        for key, value in params.items():
            row[key] = utils.convert_value(value)
        rows.append(row)

    # Create a DataFrame from the rows.
    df = pd.DataFrame(rows)

    # Write the DataFrame to a Feather file.
    feather.write_feather(df, filepath)
    print(f'Saved processed data to {filepath}.')


def load_per_psr_data_dict(filepath):
    """
    Load data dictionary from feather file.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing necessary data for analysis.
        Output of 'build_data_dict'.
    filepath : str
        Local destination to save 'data_dict'.

    Returns
    -------
    None
    """
    df_loaded = feather.read_feather(filepath)
    # First convert the DataFrame to a list of row dictionaries.
    records = df_loaded.to_dict(orient="records")
    # Reconstruct the original nested dictionary.
    data_dict_reconstructed = {}
    for row in records:
        pulsar = row["pulsar"]
        params = {}
        for key, value in row.items():
            if key != "pulsar":
                params[key] = utils.restore_value(value)
        data_dict_reconstructed[pulsar] = params
    return data_dict_reconstructed

