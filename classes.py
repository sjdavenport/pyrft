"""
    Random field classes
"""
import numpy as np

class Field:
    """
  Field class
  -----------------------------------------------------------------------------
  ARGUMENTS
  - field:  a numpy array of field
  - mask:  a boolean numpty array giving the spatial mask the size of which 
           must be compatible with the field
  -----------------------------------------------------------------------------
  OUTPUT
  - An object of class field
  -----------------------------------------------------------------------------
  EXAMPLES
  # 1D
field = np.random.randn(100,30)
mask = np.ones((1,100), dtype = bool)
exField = pr.Field(field, mask)
print(exField)
  # 2D
field = np.random.randn(100,100,30)
mask = np.ones((100,100), dtype = bool)
exField = pr.Field(field, mask)
print(exField)
  -----------------------------------------------------------------------------
    """
    def __init__(self, field, mask):
        self.field = field
        self.fieldsize = field.shape
        masksize = mask.shape
        
        # Check that the mask is a boolean array        
        if mask.dtype != np.bool:
            raise Exception("The mask must be a boolean array")
            
        # Assign the dimension
        self.D = len(masksize)
        
        # Obtain the fibersize
        self.fibersize = self.field.shape[self.D:][0]
        
        # Cover the 1D case where the mask is a vector! 
        # (Allows for row and column vectors)
        if (self.D == 2) and (masksize[0] == 1 or masksize[1] == 1):
            self.D = 1
            self.masksize = tuple(np.sort(masksize))
            
        # Ensure that the size of the mask matches the size of the field
        if self.D > 1 and field.shape[0:self.D] != masksize:
            raise Exception("The size of the spatial field must match the mask")
        elif self.D == 1 and field.shape[0:self.D] != masksize[1]:
            raise Exception("The size of the spatial field must match the mask")
            
        # If it passes the above tests assign the mask to the array
        self.masksize = masksize
        self.mask = mask
        
    def __str__(self):
        # Initialize string output
        str_output = ''
        
        # Get a list of the attributes
        attributes = vars(self).keys()
        
        # Add each attribute (and its properties to the output)
        for atr in attributes:
            if atr in ['D', 'fibersize']:
                str_output += atr + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__mask']:
                pass
            elif atr in ['_Field__fieldsize']:
                str_output += 'fieldsize' + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__masksize']:
                str_output += 'masksize' + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__field']:
                str_output += 'field' + ': ' + str(getattr(self, atr).shape) + '\n'
            else:
                str_output += atr + ': ' + str(getattr(self, atr).shape) + '\n'

        # Return the string (minus the last \n)
        return str_output[:-1]
    
    #Getting and setting field
    def _get_field(self):
        return self.__field
            
    def _set_field(self, value):
        if hasattr(self, 'mask'):
            if self.D > 1:
                if value.shape[0:self.D] != self.masksize:
                    raise ValueError("The size of the field must be compatible with the mask")
            else:
                if value.shape[0:self.D] != self.masksize[1]:
                    raise ValueError("The size of the field must be compatible with the mask")
        self.__field = value 
        self.fieldsize = value.shape
        
    #Getting and setting mask
    def _get_mask(self):
        return self.__mask
            
    def _set_mask(self, value):
        if (self.D > 1) and value.shape != self.masksize:
            raise ValueError("The size of the mask must be compatible with the field")
        elif (self.D == 1) and tuple(np.sort(value.shape)) != self.masksize:
            raise ValueError("The size of the mask must be compatible with the field")
        if  value.dtype != np.bool:
            raise Exception("The mask must be a boolean array")
        self.__mask = value
        self.masksize = value.shape
        
    #Getting and setting fieldsize
    def _get_fieldsize(self):
        return self.__fieldsize
            
    def _set_fieldsize(self, value):
        if value != self.field.shape:
            raise Exception("The field size cannot be changed directly")
        self.__fieldsize = value
        
    #Getting and setting masksize
    def _get_masksize(self):
        return self.__masksize
            
    def _set_masksize(self, value):
        if hasattr(self, 'mask'):
            if value != self.mask.shape:
                raise Exception("The field size cannot be changed directly")
        self.__masksize = value
        
    # Set properties
    field = property(_get_field, _set_field)
    mask = property(_get_mask, _set_mask)
    fieldsize = property(_get_fieldsize, _set_fieldsize)
    masksize = property(_get_masksize, _set_masksize)
                         