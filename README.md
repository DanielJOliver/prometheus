# Prometheus

_Prometheus_ is a fast pulsar timing array analysis package which supports common models for intrinsic pulsar noise and a stochastic gravitational wave background. Deterministic signals are supported, so fast fully joint analyses may be performed. The methods of this package are described in the paper: [arXiv:2607.06834](https://arxiv.org/abs/2607.06834).

On a NVIDIA GeForce RTX 3090, parameter estimation on the NANOGrav 15-year data set takes ~15 minutes!

The posterior density sampled in **Prometheus** is equivalent to that of the [Enterprise](https://github.com/nanograv/enterprise) software. However, in implementation they differ in three ways:

1) The Fourier coefficients representing stochastic timing delays are directly sampled in **Prometheus**, rather than integrated from the model. That is, they are numerically marginalized over instead of analytically marginalized.

2) The Fourier coefficients are sampled under a standardizing coordinate transform, so they approximately obey a standard normal distribution.

3) Deterministic signals are represented in a Fourier basis (under the hood) to retain a hyper-efficient posterior formulation.

The posterior is implemented in [JAX](https://jax.readthedocs.io/en/latest/) which supports JIT and automatic differentiation methods. The posterior sampling is performed using [NumPyro's No U-Turn Sampler](https://num.pyro.ai/en/stable/mcmc.html#id7).

## Installation

Install directly with pip:
```bash
pip install git+https://github.com/XGI-MSU/prometheus.git
```

## Requirements & Conventions

**Prometheus** requires a modern Python with standard packages, including `enterprise`. The analysis should be executed on a NVIDIA GPU with [CUDA-enabled JAX](https://docs.jax.dev/en/latest/installation.html).

The posterior is evaluated in single precision (`float32`). To accomodate this choice, timing residuals use units of _nano-seconds_. Any custom models supplied by the user need to be stable in single precision, JAX compatible, and use base units of nano-seconds. See functions in `spectra.py` and `deterministic.py` for examples.

One step uses double precision (`float64`): the Cholesky factorization of the per-pulsar matrix `M = I + B B^T` in the `'marginalized'` sampling model, which loud signals (e.g. DM variations) give a dynamic range beyond `float32`. Its derivative is still evaluated in `float32`. For this, `__init__.py` enables `float64` in JAX, and **Prometheus** keeps everything else in `float32`: the `Data` object stores `float32` arrays, sampled parameters are rounded to `float32`, and the outputs of spectral and deterministic models (including custom ones) are cast to `float32`. Since JAX then creates `float64` arrays by default, custom code that creates arrays should give them a `float32` dtype (e.g. `jnp.zeros(n, dtype=jnp.float32)`) to stay fast. The exception is NumPyro's sampler, whose position is kept in `float64`: initial values passed to `init_to_value` must not be `float32` arrays.


## Examples

The `examples` folder contains a variety of example analyses in interactive Python notebooks.

- `gwb_pe.ipynb` reproduces (some) results from the NANOGrav 15-year stochastic analysis.

- `cw_pe.ipynb` reproduces (some) results from the NANOGrav 15-year continuous wave analysis.

- `dmgp_pe.ipynb` injects dispersion measure (DM) variations into a simulated dataset with the NANOGrav 15-year array geometry and recovers them with a DM Gaussian process.

- The `advanced_modeling` folder contains examples illustrating the construction and sampling of more complicated models.

## Data structure (`data.py`)

**Prometheus** stores PTA data in a custom `prometheus.data.Data` object. This object may be constructed from a list of `enterprise.pulsar.PintPulsar` (or `FeatherPulsar`) objects and a standard white noise dictionary.

Besides common PTA attributes (number of pulsars, sky locations, etc.), the `Data` object also stores constants used for the evaluation of the posterior (e.g. "TNT", "TNr", etc.).

### Fourier bases

Gaussian processes live on named Fourier bases. `'red'` is the achromatic basis
(`nfreqs` frequency bins on the PTA span) of the intrinsic pulsar noise, which the
common process (e.g. the GWB) uses by default. Further bases are given as a list of
`prometheus.data.FourierBasis` objects, each with a name, a number of frequency bins,
a chromatic index (columns scaled per TOA by `(fref / ν)^index`, with `ν` the radio
frequency of the TOA: 0 for achromatic, 2 for DM variations, 4 for scattering), and
optionally its own span:

```python
from prometheus.data import FourierBasis

data = data.Data(name='NG15', psrs=psrs, wn_dict=wn_dict, nfreqs=30,
                 extra_bases=[FourierBasis('dm', nfreqs=30, chromatic_index=2)])
```

(`nfreqs_dm=30` is a shorthand for the DM basis above.) The noise-weighted inner
products between every pair of bases, the deterministic basis and the residuals
are computed once, when the `Data` object is built; spectral models then refer to a
basis by name.

## Signal and noise models

The white noise model is fixed throughout the analysis. Linear deviations to the reference parameters of the pulsar timing model are analytically marginalized.

### Stochastic spectral models (`spectral_models.py`)

Pulsar noise and gravitational wave background signals are modeled using a set
of Fourier coefficients with zero-mean multivariate normal priors (i.e. Gaussian
processes).

The prior covariance of these coefficients may be parameterized by an arbitrary
hyper-spectral model. Inter-pulsar correlations (e.g. Hellings-Downs) are supported,
and inter-frequency correlations (`freq_corrs=True`) are supported in the
`'marginalized'` sampling model (see below).

Pulsar noise models are most naturally represented by instances of
`prometheus.spectral_models.IndependentSpectralModel`. This assumes that all pulsars use the same (statistically independent) noise model, e.g. a power law.
Each model lives on one Fourier basis (`basis='red'` by default), and any number of
such per-pulsar processes can be modeled at once; processes on the same basis add
their prior covariances.

Gravitational wave background models are most naturally represented by
instances of `prometheus.spectral_models.CommonSpectralModel`. This assumes a common spectrum across all pulsars, consistent with a specified inter-pulsar correlation pattern.

More general spectral models—not fitting in the categories above—can be implemented using the `prometheus.spectral_models.SpectralModel` class. While this option offers maximum
flexibility, constructing a `SpectralModel` typically requires more effort; see
the `advanced_modeling` example notebooks.

#### Dispersion measure Gaussian processes

Chromatic dispersion measure (DM) variations are a per-pulsar process on the DM
basis `F_dm = F * (fref / ν)^2` (enterprise's convention; coefficients are delays at
`fref`, 1400 MHz by default):

```python
dm_model = IndependentSpectralModel(name='dm_params', get_phi_diag_func=spectra.power_law,
                                    parameter_bounds=[[-20., -10.], [0., 7.]], data=data,
                                    basis='dm')
pta_model = PTAModel(psr_model=[psr_model, dm_model], gwb_model=gwb_model)
```

`prometheus.spectral_models.DMSpectralModel` is a shorthand for an
`IndependentSpectralModel` on the `'dm'` basis, and `PTAModel(..., dm_model=dm_model)`
for appending it to the per-pulsar processes. Other chromatic processes (e.g.
scattering) are added the same way, on a `FourierBasis` with their chromatic index.
See `examples/dmgp_pe.ipynb`.


### Deterministic Models (`deterministic_models.py`)

Arbitrary deterministic models are represented with an instantiation of a `prometheus.deterministic_models.DeterministicModel` object.

While arbitrary deterministic models are architecturally "supported", satisfactory sampling for every model cannot be guaranteed. Chain convergence and sampling diagnostics should always be checked!

See the `tests` folder where `float32` stability and the Fourier representation of an example deterministic signal is demonstrated.


## PTA Models (`pta_model.py`)

The `prometheus.pta_model.PTAModel` class contains the constituent signal and noise models. It builds a NumPyro probabilistic sampling model and posterior samples can be obtained via NumPyro's NUTS implementation.

A `PTAModel` can be constructed in one of two operational modes:

1. **Standard mode**  
   The user supplies:
   - one or a list of `IndependentSpectralModel`s (the per-pulsar processes), and
   - optionally, a `CommonSpectralModel`.

2. **Custom mode**  
   The user supplies a single `SpectralModel` that fully specifies the PTA
   spectral structure.

**Standard mode** is recommended for simpler analyses, for example when all
pulsars share the same independent noise model and the gravitational-wave
background has fixed inter-pulsar correlations.

**Custom mode** provides greater flexibility, but constructing a
`SpectralModel` typically requires more user effort.

In both standard and custom modes, an optional `DeterministicModel` may be
included when constructing a `PTAModel`.

### Sampling models

`PTAModel` has two NumPyro sampling models, chosen with `sampling=`:

- **`'shared'`** (`PTAModel.sampling_model_shared`): the common and per-pulsar
  processes are described by *one* set of Fourier coefficients on a shared basis,
  with prior covariance `φ_psr + φ_gwb`, and all coefficients are sampled through a
  standardizing transform. It requires every process to live on the `'red'` basis,
  without inter-frequency correlations, and is the only option in custom mode.
- **`'marginalized'`** (`PTAModel.sampling_model_marginalized`): the per-pulsar and
  common coefficients are separate. The per-pulsar coefficients of the stacked basis
  (e.g. `[F | F_dm]`) are marginalized analytically, and only the common
  coefficients are sampled, under the standardizing transform conditional on the
  marginalized per-pulsar processes. It supports any number of per-pulsar processes
  on any bases, inter-frequency correlations, a common process on its own basis, and
  models without a common process.

By default (`sampling=None`) the `'shared'` model is used whenever it applies and a
common process is modeled, and the `'marginalized'` model otherwise.

Prometheus' **standard** mode is illustrated below.
![standard_mode](prometheus.png)

## Priors

- The Fourier coefficients use zero-mean multivariate normal priors.

- If included in deterministic models, the pulsar distance parameters use normal priors.

- All other parameters (e.g. those of spectral and determinstic models) use uniform priors over the bounds specified by the user. If a sampled parameter is logarithmic (e.g. $\log_{10}\mathcal{M}$), the uniform prior implies a log-uniform prior over the base parameter ($\mathcal{M}$).

- To alter priors, users may supply an `additional_ln_factor` function as input to the to desired model (e.g. `IndependentSpectralModel`, `CommonSpectralModel`, `DeterministicModel`). This function adds a natural-logarithmic term to the posterior evaluation. See examples in the `advanced_modeling` folder. 
