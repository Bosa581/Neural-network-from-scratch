import math

def sig(x):
    # logistic activation function
    return 1 / (1 + math.exp(-x))

def inv_sig(x):
    # derivative of sigmoid: σ(x) * (1 - σ(x))
    s = sig(x)
    return s * (1 - s)

def err(o, t):
    return 0.5 * sum((t[i] - o[i])**2 for i in range(len(o)))

def inv_err(o, t):
    # derivative of squared error with respect to output o: o - t
    return [o[i] - t[i] for i in range(len(o))]

