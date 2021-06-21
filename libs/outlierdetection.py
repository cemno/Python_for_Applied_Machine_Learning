"""
  This file will contain the outlier detection functions. We could have put them
  in the main document but creating libraries is a good habit to get into. We can now
  use these for other things...
"""
import numpy as np

# From the pdf you should see the Z score and the MAD score for calculating outliers.
# You should notice that the Z score is just a normalisation. Try both now.
# For the detection based on the threshold remember that it's a distribution so it
# has normalised values such that the std is -1 to 1....
# For both function you want to return a boolean array as whether to keep the data
# point (True) or remove it (False).
# There are obviously a number of ways to do this, but I suggest looking into np.logical_and

# The Z score version
def z_score( X, t = 2.5):
    mu = X.mean()
    st = X.std()
    z = (X-mu)/st
    b = np.logical_and(z < t, z > -t)
    return b

# Median absolute difference version
def z_mod_score(X, t = 3.5):
    md = np.median(X)
    D = np.median(np.abs(X - md))
    m = 0.6745 * (X - md) / D
    b = np.logical_and(m < t, m > -t)
    return b