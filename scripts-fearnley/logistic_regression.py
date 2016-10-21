def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1+np.exp(t))
sigmoid(5)