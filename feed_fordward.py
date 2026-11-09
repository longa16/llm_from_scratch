import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    h = (np.matmul(x, W1) + b1)
    a = np.maximum(0, h)
    ffn = (np.matmul(a, W2) + b2)
    return ffn