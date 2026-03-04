import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return np.array([], dtype=np.int64).reshape(0, 0)

    if max_len is None:
        max_len = max(map(len, seqs)) if seqs else 0

    # Truncate + pad in one go for each sequence
    padded = [
        np.pad(
            np.array(seq[:max_len], dtype=np.int64),
            (0, max(0, max_len - len(seq))),
            constant_values=pad_value
        )
        for seq in seqs
    ]

    return np.array(padded)