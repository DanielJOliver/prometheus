'''prometheus.'''

# float64 must be available for the factorization in
# pta_model.batched_cholesky_identity_plus_gram; everything
# else is computed in single float32 precision.
import jax
jax.config.update('jax_enable_x64', True)
