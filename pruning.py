import jax.numpy as jnp


def prune_weights(weights: jnp.ndarray, bias: jnp.ndarray, sparsity_percentage:float):
    """
    Takes in weights and bias of a dense layer and returns 
    a at rank k pruned version

    Args:
      weights: 2D array
      bias: 1D array 
      rank_sparsity: percentage of params to set to 0
    """

    # Sorts indices by abs value
    weights = jnp.copy(weights)
    indices = jnp.unravel_index(
        jnp.argsort(
            jnp.abs(weights),
            axis=None),
        weights.shape)
        
    # Indices to set to 0
    threshold = int(len(indices[0])*sparsity_percentage)
    sparse_indices = (indices[0][0:threshold], indices[1][0:threshold])
    weights[sparse_indices] = 0.
        
    # Sorts indices by abs value
    bias = jnp.copy(bias)
    indices = np.unravel_index(
        jnp.argsort(
            jnp.abs(bias), 
            axis=None), 
        bias.shape)
        
    # Indices to set to 0
    threshold = int(len(indices[0])*sparsity_percentage)
    sparse_indices = (indices[0][0:threshold])
    bias[sparse_indices] = 0.

    return jnp.array(weights), jnp.array(bias)

#TODO prune_nodes()
