"""
Functions to simulate and test when the null is not true everywhere
"""

def random_signal_locations(lat_data, categ, C, pi0):
    """ A function which generates random data with randomly located non-zero 
    contrasts

    Parameters
    -----------------
    lat_data: an object of class field
        giving the data with which to add the signal to
    categ: numpy.nd array,
        of shape (nsubj, 1) each entry identifying a category 
    C: numpy.nd array
        of shape (ncontrasts, nparams)
        contrast matrix
    pi0:

    Returns
    -----------------

    Examples
    -----------------
    
    """
    # Compute important constants
    nsubj = lat_data.fibersize
    dim = lat_data.fieldsize
    ncontrasts = C.shape[0]
    
    # Compute derived constants
    nvox = np.prod(dim) # compute the number of voxels
    m = nvox*n_contrasts # obtain the number of voxel-contrasts
    ntrue = int(np.round(pi0 * m)) # calculate the closest integer to make the 
                                #proportion of true null hypotheses equal to pi0
    
    # Initialize the true signal vector
    signal_entries = np.zeros(m)
    signal_entries[ntrue:] = 1
