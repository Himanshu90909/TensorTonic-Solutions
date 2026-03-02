import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works with scalars, lists, and NumPy arrays.
    Returns a NumPy array of floats.
    """
    # Convert input to NumPy array (ensures vectorization)
    x = np.array(x, dtype=float)
    
    # Apply sigmoid formula
    return 1 / (1 + np.exp(-x))