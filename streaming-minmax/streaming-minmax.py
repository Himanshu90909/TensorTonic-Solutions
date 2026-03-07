import numpy as np

def streaming_minmax_init(D):
    """
    Initialize streaming min-max state.
    """
    return {
        "min": np.full(D, np.inf),
        "max": np.full(D, -np.inf)
    }


def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update running min/max and normalize the batch.
    
    Parameters:
        state: dict with 'min' and 'max' (shape D,)
        X_batch: array-like of shape (B, D)
        eps: small float to avoid division by zero
    
    Returns:
        X_normalized: normalized batch (B, D)
    """
    X_batch = np.asarray(X_batch)
    
    # Compute batch statistics
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)
    
    # Update running statistics (post-update normalization required)
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)
    
    # Compute denominator safely
    denom = state["max"] - state["min"]
    
    # Normalize using UPDATED statistics
    X_normalized = (X_batch - state["min"]) / (denom + eps)
    
    return X_normalized