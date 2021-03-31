""" 
A file containing the random field generation functions
"""
import numpy as np
import pyrft as pr

def wfield(mask, fibersize, field_type = 'N', field_params = 3):
  """ wfield constructs a an object of class Field having white noise in the 
    fiber.
  -----------------------------------------------------------------------------
  ARGUMENTS
  - mask:   a tuple giving the size of the mask (in which case it is taken
            to be all true)or a boolean array corresponding to the mask
  - fibersize:   a tuple giving the fiber sizes (i.e. typically nsubj)
  -----------------------------------------------------------------------------
  OUTPUT
  - An object of class field
  -----------------------------------------------------------------------------
  EXAMPLES
  exF = pr.wfield((5,5), 10)
  -----------------------------------------------------------------------------
  """
  
  # If the mask is logical use that!
  if isinstance(mask, np.ndarray) and mask.dtype == np.bool:
      # If a logical array assign the mask shape
      masksize = mask.shape
  elif  isinstance(mask, tuple):
      # If a tuple generate a mask of all ones
      masksize = mask
      mask = np.ones(masksize, dtype = bool)
  elif isinstance(mask, int):
      masksize = (mask,);
      mask = np.ones(masksize, dtype = bool)
  else:
      raise Exception("The mask is not of the right form")
  
  # Calculate the overall size of the field
  if isinstance(fibersize, int):
      fieldsize = masksize + (fibersize,)
  else:
      fieldsize = masksize + fibersize
  
  # Generate the data from the specified distribution
  if field_type == 'N':
      data = np.random.randn(*fieldsize)
  
  # Combine the data and the mask to make a field
  out = pr.Field(data, mask)
  
  # Return the output
  return out
  
  