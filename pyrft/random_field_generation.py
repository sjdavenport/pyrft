""" 
A file containing the random field generation functions
"""
import numpy as np
import pyrft as pr

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
  
  