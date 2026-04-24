import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
  
    # positions: (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]

    # dimension indices: (1, d_model)
    dim = np.arange(d_model)[np.newaxis, :]

    # compute the exponent term: (2i / d_model)
    exponent = (2 * (dim // 2)) / d_model

    # compute angle rates
    angle_rates = 1 / (base ** exponent)

    # compute angles
    angles = pos * angle_rates   # shape: (seq_len, d_model)

    # apply sin to even indices, cos to odd indices
    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # even indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # odd indices

    return pe