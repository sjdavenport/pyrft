""" 
A file containing the random field generation functions
"""
import numpy as np
import pyrft as pr
from scipy.ndimage import gaussian_filter


def smooth(data, FWHM, mask = 0):
  """ smooth
    
  Parameters  
  ---------------------
  data    an object of class field
  FWHM    
  mask   a numpy.nd array,
          with the same dimensions as the data

  Returns
  ---------------------
  An object of class field with 
  
  Examples
  ---------------------
# 2D example
f = pr.wfield((50,50), 10)
smooth_f = pr.smooth(f, 8)
plt.imshow(smooth_f.field[:,:,1])

# 2D example with mask
f = pr.wfield((50,50), 10)
mask = np.zeros((50,50), dtype = 'bool')
mask[15:35,15:35] = 1
smooth_f = pr.smooth(f, 8, mask)
plt.imshow(smooth_f.field[:,:,1])
  """
  # Convert a numpy array to a field if necessary
  if type(data) == np.ndarray:
        data = pr.makefield(data)
  
  # If a non-zero mask is supplied used this instead of the mask associated with data
  if np.sum(np.ravel(mask)) > 0:
      data.mask = mask
  
  # Calculate the standard deviation from the FWHM
  sigma = pr.FWHM2sigma(FWHM)  

  for I in np.arange(data.fibersize):
    data.field[...,I] = gaussian_filter(data.field[...,I]*data.mask, sigma = sigma)*data.mask

  return data

def statnoise(mask, nsubj, FWHM):
  """ statnoise constructs a an object of class Field with specified mask
  and fibersize and consisting of stationary noise (arising from white noise
  smoothed with a Gaussian kernel)
  
  Parameters  
  ---------------------
  mask:   a tuple or a Boolean array,
          If a tuple then it gives the size of the mask (in which case the mask
          is taken to be all true) 
          If a Boolean array then it is the mask itself
  fibersize:   a tuple giving the fiber sizes (i.e. typically nsubj)

  Returns
  ---------------------
  An object of class field with 
  
  Examples
  ---------------------
Dim = (50,50); nsubj = 20; FWHM = 4
F = pr.statnoise(Dim, nsubj, FWHM)
plt.imshow(F.field[:,:,1])

# No smoothing example:
Dim = (50,50); nsubj = 20; FWHM = 0
F = pr.statnoise(Dim, nsubj, FWHM)
plt.imshow(F.field[:,:,1]) 

  Notes
  ---------------------
  Need to adjust this to account for the edge effect!
  Also need to ensure that the field is variance 1!!
  """
  # Set the default dimension not to be 1D
  use1D = 0
  
  # If the mask is logical use that!
  if isinstance(mask, np.ndarray) and mask.dtype == np.bool:
      # If a logical array assign the mask shape
      masksize = mask.shape
  elif  isinstance(mask, tuple):
      # If a tuple generate a mask of all ones
      masksize = mask
      mask = np.ones(masksize, dtype = bool)
  elif isinstance(mask, int):
      use1D = 1
      masksize = (mask,1)
      mask = np.ones(masksize, dtype = bool)
  else:
      raise Exception("The mask is not of the right form")
  
  # Calculate the overall size of the field
  if use1D:
      fieldsize = (masksize[0],) + (nsubj,)
  else:
      fieldsize = masksize + (nsubj,)
  
  # Calculate the sigma value with which to smooth form the FWHM
  sigma = pr.FWHM2sigma(FWHM)  
  
  # Generate normal random noise
  data = np.random.randn(*fieldsize)  

  for n in np.arange(nsubj):
      data[...,n] = gaussian_filter(data[...,n], sigma = sigma)
      
  # Combine the data and the mask to make a field
  out = pr.Field(data, mask)
      
  # Return the output
  return out

def wfield(mask, fibersize, field_type = 'N', field_params = 3):
  """ wfield constructs a an object of class Field with specified mask
  and fibersize and consisting of white noise.
  
  Parameters  
  ---------------------
  mask:   a tuple or a Boolean array,
          If a tuple then it gives the size of the mask (in which case the mask
          is taken to be all true) 
          If a Boolean array then it is the mask itself
  fibersize:   a tuple giving the fiber sizes (i.e. typically nsubj)

  Returns
  ---------------------
  An object of class field with 
  
  Examples
  ---------------------
  exF = pr.wfield(1, 10)
  exF = pr.wfield((5,5), 10)
  
  Notes
  ---------------------
  Need to ensure that this function works in all settings, i.e. 1D masks specified 
  as (10,1) for example!
  """
  
  # Set the default dimension not to be 1D
  use1D = 0
  
  # If the mask is logical use that!
  if isinstance(mask, np.ndarray) and mask.dtype == np.bool:
      # If a logical array assign the mask shape
      masksize = mask.shape
  elif  isinstance(mask, tuple):
      # If a tuple generate a mask of all ones
      masksize = mask
      mask = np.ones(masksize, dtype = bool)
  elif isinstance(mask, int):
      use1D = 1
      masksize = (mask,1)
      mask = np.ones(masksize, dtype = bool)
  else:
      raise Exception("The mask is not of the right form")
  
  # Calculate the overall size of the field
  if use1D:
      fieldsize = (masksize[0],) + (fibersize,)
  else:
      fieldsize = masksize + (fibersize,)
  
  # Generate the data from the specified distribution
  if field_type == 'N':
      data = np.random.randn(*fieldsize)
        
  # Combine the data and the mask to make a field
  out = pr.Field(data, mask)
  
  # Return the output
  return out
  
  