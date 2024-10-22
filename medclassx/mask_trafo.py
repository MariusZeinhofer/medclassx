"""Mask a vector and transfer back to original shape."""

import jax.numpy as jnp


def mask_vector(v, vector_mask):
    """Removes the entries of v that correspond to zeros of vector_mask.

    Args:
        v: Array of shape (p,) whose entries corresponding to zeros of vector_mask
            will be removed.
        
        vector_mask: Array of shape (p,), its zeros mask v.

    Returns:
        A tuple consiting of the masked vector v of shape (p-#zeros_of_vector_mask,)
        and a function that transforms the masked and shortened vector back to its
        original shape. The values that where masked are filled with zeros.
    """
    non_zero_idxs = jnp.nonzero(vector_mask)

    def unmask(w, padding=0):
        """Transforms w back to the shape of v, padds masked fields with zero.
        
        Args:
            w: Array of shape (p-#zeros_of_vector_mask,)

            padding: value used to pad when unmasking is done.

        Returns:
            A vector of the shape of v with nonzero entries where vector_mask is not 0.
        """
        v_zeros = padding * jnp.ones_like(v)
        return v_zeros.at[non_zero_idxs].set(w)

    return v[non_zero_idxs], unmask
    


if __name__ == "__main__":
    
    v = jnp.array([5, 2, 3, 4, 1])
    print(v)
    vector_mask = jnp.array([1, 0, 0, 1, 1])
    print(vector_mask)

    w, unmask = mask_vector(v, vector_mask)
    print(w)
    print(unmask(w))