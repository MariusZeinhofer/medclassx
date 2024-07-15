"""Test `mask_trafo`."""

import jax.numpy as jnp
import pytest

from medclassx.mask_trafo import mask_vector

@pytest.mark.parametrize(
        "v, mask",
        [
            (
                jnp.array([0., 1., 2., 3., 4., 5.]), 
                jnp.array([0., 1., 2.7, 0., 0., 0.])
            ),
            (
                jnp.array([0., 1., 2., 3., 4., 5.]), 
                jnp.array([0, 1, 2, 0, 0, 0])
            ),
        ]                 
)
def test_mask_vector(v, mask):
    """Test mask_vector function."""

    v_masked, unmask = mask_vector(v, mask) 
    w = unmask(v_masked)

    assert (v_masked == jnp.array([1., 2.])).all()
    assert (w == jnp.array([0., 1., 2., 0., 0., 0.])).all()