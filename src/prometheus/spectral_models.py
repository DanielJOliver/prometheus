"""
Holds classes for stochastic Gaussian spectral models.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from typing import Callable, Optional

from . import utilities as utils
from .data import Data
from . import spectra
from .FFTInt import build_phi_from_chat


class SpectralModel:
    """
    Base class for spectral models corresponding to Gaussian processes.
    An advanced user can use this class for their stochastic models instead
    of the models below, which are limited (but easier to implement).
    
    Required Attributes
    -------------------
    name : str
        The name of the spectral model.
    parameter_bounds : array
        (Nparam, 2) array where 'Nparam' is the total
        number of parameters of the spectral model.
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_cube_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the total number of parameters of the spectral model and
        an array of frequencies of shape (Nf,) where 'Nf' is the number of frequency bins. 
        The function should output a (2*Nf, Npsrs, Npsrs) array where 'Npsrs' is the
        number of pulsars in the array. The (i, j, k) element of the output array
        is the (prior) covariance of ith Fourier coefficient in pulsars j and k.
        The output should have units of [ns]^2.

        **See the advanced modeling example notebooks for examples of get_phi_cube_func.**
    get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Optional: the per-pulsar spectrum, only needed with freq_corrs=True.
    basis : str
        Name of the Fourier basis (see data.Data) whose coefficients this model
        describes. Defaults to 'red', the achromatic pulsar noise basis.
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_diag_func : Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
                 get_phi_cube_func : Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
                 freq_corrs : Optional[bool] = False,
                 additional_ln_factor : Optional[Callable] = None,
                 basis : str = 'red',
                 ):

        self.name = name
        self.parameter_bounds = jnp.array(parameter_bounds, dtype=jnp.float32)
        self.param_mins = self.parameter_bounds[..., 0]
        self.param_maxs = self.parameter_bounds[..., 1]
        self.nparams_base = self.param_mins.shape[0]
        self.data = data
        self.get_phi_diag_func = get_phi_diag_func
        self.freq_corrs = freq_corrs
        self.additional_ln_factor = additional_ln_factor

        if get_phi_cube_func is None:
            raise ValueError("get_phi_cube_func must be provided.")
        if self.freq_corrs and get_phi_diag_func is None:
            raise ValueError("get_phi_diag_func must be provided when freq_corrs=True.")

        self._get_phi_cube_func = get_phi_cube_func

        # the Fourier basis this model lives on
        self.basis = basis
        self.freqs = self.data.basis_freqs(basis)
        self.nfreqs_basis = len(self.freqs)

        if self.freq_corrs:
            chat_kwargs = dict(oversample=8, nfreqs=self.nfreqs_basis, Tspan=self.data.basis_Tspan(basis))
            if self.get_phi_diag_func is spectra.free_spectral:
                # free_spec_spline = spectra.make_cubic_spline_free_spectral_model(self.freqs)
                # free_spec_spline = spectra.make_linear_spline_free_spectral_model(self.freqs)
                free_spec_spline = spectra.make_akima_spline_free_spectral_model(self.freqs, extrapolate='akima')
                get_phi_free_spec = build_phi_from_chat(free_spec_spline, self.data, **chat_kwargs)
                self.get_phi_func = lambda x, f: get_phi_free_spec(x)
            else:
                get_phi_spec_func = build_phi_from_chat(self.get_phi_diag_func, self.data, **chat_kwargs)
                self.get_phi_func = lambda x, f: get_phi_spec_func(x)
        else:
            self.get_phi_func = lambda x, f: jnp.diag(self.get_phi_diag_func(x, f))

        # the posterior is evaluated in float32, whatever the precision of the spectrum
        get_phi_func = self.get_phi_func
        self.get_phi_func = lambda x, f: get_phi_func(x, f).astype(jnp.float32)

    def get_phi_cube(self, params, freqs):
        """
        Gets the prior covariance matrix for the Fourier coefficients
        conditioned on this spectral model and some parameters.

        Parameters
        ----------
        params : array
            Array of spectral-hyper-parameters.
        freqs : array
            Array of shape (Nf,) of frequencies, where Nf is the number
            of frequency bins modeled.
        
        Returns
        -------
        phi_cube : array
            Array of shape (2Nf, Npsrs, Npsrs) where Nf is the number of frequency
            bins modeled and Npsrs is the number of pulsars in the array. This 
            array is the prior covariance of the Fourier coefficients such that
            the (i, j, k) element is the covariance of the ith Fourier coefficient
            between pulsars j and k. The output array should have units of [ns]^2.
        """
        return self._get_phi_cube_func(params, freqs).astype(jnp.float32)


class IndependentSpectralModel(SpectralModel):

    """
    Spectral model which batched across pulsars. Used primarily for 
    pulsar noise models, where each pulsar uses the same (statistically
    independent) model (e.g. a power law).

    The model describes all coefficients of its Fourier basis: by default the
    achromatic 'red' basis (intrinsic pulsar noise); give e.g. basis='dm' for
    dispersion measure variations (see DMSpectralModel) or any other basis of
    the Data object. Several models on the same basis add their covariances.

    Inherits from :class:`SpectralModel`.

    Required Attributes
    -------------------
    name : str
        The name of spectral model.
    parameter_bounds : array
        The minima and maxima allowed values of the parameters
        The shape of the array is (Nparam, 2) where 'Nparam'
        is number of parameters of the spectral model (per pulsar).
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the number of parameters of the spectral model and
        an array of frequencies of shape (Nf,) where 'Nf' is the number of
        unique frequency bins. The function outputs an array of shape (2*Nf,)
        which is the diagonal elements of prior covariance matrix for the Fourier
        coefficients obeying that spectral model. The output array must should
        use units of [ns]^2. See spectra.py for examples.
    freq_corrs : bool
        Whether or not to include inter-frequency correlations in the
        prior covariance. Defaults to False.
    basis : str
        Name of the Fourier basis of the Data object the model lives on.
        Defaults to 'red'.
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 freq_corrs : Optional[bool] = False,
                 additional_ln_factor : Optional[Callable] = None,
                 basis : str = 'red',
                 ):
        
        self.get_phi_diag_func = get_phi_diag_func
        self.freq_corrs = freq_corrs
        self.additional_ln_factor = additional_ln_factor

        def get_phi_cube_func(params, freqs):
            """
            Gets the prior covariance matrix for the Fourier coefficients
            conditioned on this spectral model and some parameters.

            Parameters
            ----------
            params : array
                Array of spectral-hyper-parameters.
            freqs : array
                Array of shape (Nf,) of frequencies, where Nf is the number
                of frequency bins modeled.
            
            Returns
            -------
            phi_cube : array
                Array of shape (2Nf, Npsrs, Npsrs) where Nf is the number of frequency
                bins modeled and Npsrs is the number of pulsars in the array. This 
                array is the prior covariance of the Fourier coefficients such that
                the (i, j, k) element is the covariance of the ith Fourier coefficient
                between pulsars j and k. The covariance should use units of [ns]^2.
            """
            phi_diags = jax.vmap(get_phi_diag_func, in_axes=(0, None))(params, freqs).T
            nfreqs2, npsrs = phi_diags.shape
            phi_cube = jnp.zeros((nfreqs2, npsrs, npsrs), dtype=phi_diags.dtype)
            ii = jnp.arange(npsrs)
            return phi_cube.at[:, ii, ii].set(phi_diags)

        super().__init__(
            name=name,
            parameter_bounds=parameter_bounds,
            data=data,
            get_phi_diag_func=self.get_phi_diag_func,
            freq_corrs=self.freq_corrs,
            get_phi_cube_func=get_phi_cube_func,
            additional_ln_factor=additional_ln_factor,
            basis=basis,
        )
        self.nfreqs = self.nfreqs_basis


class DMSpectralModel(IndependentSpectralModel):

    """
    Spectral model for dispersion measure (DM) Gaussian processes, batched
    across pulsars: an IndependentSpectralModel on the Data object's 'dm' basis,

        F_dm = F * (data.dm_fref / nu)^2,

    where nu is the radio frequency of each TOA. Coefficients are therefore
    timing delays at the reference radio frequency, in [ns], following the
    enterprise convention (a power law amplitude is that of the delay at
    data.dm_fref).

    The Data object must be built with nfreqs_dm > 0. The model uses all
    data.nfreqs_dm frequency bins of the DM basis. Equivalent to
    IndependentSpectralModel(..., basis='dm').

    Inherits from :class:`IndependentSpectralModel`.
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 freq_corrs : Optional[bool] = False,
                 additional_ln_factor : Optional[Callable] = None,
                 ):

        if not getattr(data, 'has_dm', False):
            raise ValueError('Data object has no DM basis. Build it with nfreqs_dm > 0.')

        super().__init__(
            name=name,
            parameter_bounds=parameter_bounds,
            data=data,
            get_phi_diag_func=get_phi_diag_func,
            freq_corrs=freq_corrs,
            additional_ln_factor=additional_ln_factor,
            basis='dm',
        )


class CommonSpectralModel(SpectralModel):

    """
    Spectral model which implements a common process over pulsars under
    some correlation pattern. Used primarily for stochastic gravitational
    wave background models.

    Inherits from :class:`SpectralModel`.

    Required Attributes
    -------------------
    name : str
        The name of spectral model.
    parameter_bounds : list
        The minima and maxima allowed values of the parameters
        The shape of the array is (Nparam, 2) where 'Nparam'
        is number of parameters of the spectral model.
        The ordering is such that index [:, 0] gives the
        minima of parameters and [:, 1] the maxima.
    data : data.Data
        An instance of the Data class from the `data` module.
    get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function which takes an array of parameter values of shape (Nparam,)
        where 'Nparam' is the number of parameters of the spectral model and
        an array of frequencies of shape (Nf,) where 'Nf' is the number of
        unique frequency bins. The function outputs an array of shape (2*Nf,)
        which is the diagonal elements of prior covariance matrix for the Fourier
        coefficients obeying that spectral model. The output array must should
        use units of [ns]^2. See spectra.py for examples.
    correlation_matrix : str or array
        A (Npsrs, Npsrs) array where Npsrs is the number of pulsars in the PTA.
        The array is the correlation pattern between pulsars, e.g. element (i, j)
        is the correlation between pulsars i and j in the array. User may input
        'HD' or 'CURN' for Hellings-Downs and common uncorrelated processes, or
        may supply their own correlation matrix.
    nfreqs : int
        Number of frequencies to use in this spectral model. This value must be
        less than or equal to the number of frequency bins of its basis. If None,
        defaults to all of them.
    basis : str
        Name of the Fourier basis of the Data object the common process lives on.
        Defaults to 'red', the basis of the pulsar noise (the common and per-pulsar
        coefficients are then either shared or separated, see PTAModel).
    """

    def __init__(self,
                 name : str,
                 parameter_bounds : list | np.ndarray | jnp.ndarray,
                 data : Data,
                 correlation_matrix : str | np.ndarray | jnp.ndarray,
                 get_phi_diag_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 nfreqs : Optional[int] = None,
                 freq_corrs : Optional[bool] = False,
                 additional_ln_factor : Optional[Callable] = None,
                 basis : str = 'red'):
        self.data = data
        self.get_phi_diag_func = get_phi_diag_func
        nfreqs_basis = data.get_basis(basis).nfreqs
        self.nfreqs = nfreqs or nfreqs_basis
        if self.nfreqs > nfreqs_basis:
            raise ValueError(f"nfreqs = {self.nfreqs} exceeds the {nfreqs_basis} frequency bins of basis '{basis}'.")
        self.freq_corrs = freq_corrs
        self.additional_ln_factor = additional_ln_factor

        # zeros to append to spectrum if user requests
        # fewer frequency bins than in the basis
        self.zeros = jnp.zeros(2 * (nfreqs_basis - self.nfreqs), dtype=jnp.float32)
        self.block_zeros = jnp.zeros((self.zeros.shape[0], self.zeros.shape[0]), dtype=jnp.float32)
        self.num_coeffs = 2 * self.nfreqs

        # if 'HD' or 'CURN', build correlation matrix for user
        if isinstance(correlation_matrix, str):
            correlation_matrix = utils.resolve_psr_corr_matrix(
                correlation_matrix, data
            )
        self.correlation_matrix = jnp.array(correlation_matrix, dtype=jnp.float32)
        corr_cho_factors = jsl.cho_factor(self.correlation_matrix, lower=True)
        self.inv_correlation_matrix = jsl.cho_solve((corr_cho_factors[0], True),
                                             jnp.identity(corr_cho_factors[0].shape[0], dtype=jnp.float32))
        self.lndet_correlation_matrix = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(self.correlation_matrix))))

        def get_phi_cube_func(params, freqs):
            """
            Gets the prior covariance matrix for the Fourier coefficients
            conditioned on this spectral model and some parameters.

            Parameters
            ----------
            params : array
                Array of spectral-hyper-parameters.
            freqs : array
                Array of shape (Nf,) of frequencies, where Nf is the number
                of frequency bins modeled.

            Returns
            -------
            phi_cube : array
                Array of shape (2*Nf, Npsrs, Npsrs) where Nf is the number of frequency
                bins modeled and Npsrs is the number of pulsars in the array. This 
                array is the prior covariance of the Fourier coefficients such that
                the (i, j, k) element is the covariance of the ith Fourier coefficient
                between pulsars j and k. The covariance should use units of [ns]^2.
            """
            phi_diag = get_phi_diag_func(params, freqs)
            phi_diag = jnp.concatenate((phi_diag[:self.num_coeffs], self.zeros))
            phi_cube = phi_diag[:, None, None] * self.correlation_matrix[None, :, :]
            return phi_cube

        super().__init__(
            name=name,
            parameter_bounds=parameter_bounds,
            data=data,
            get_phi_diag_func=self.get_phi_diag_func,
            freq_corrs=self.freq_corrs,
            get_phi_cube_func=get_phi_cube_func,
            additional_ln_factor=additional_ln_factor,
            basis=basis,
        )


