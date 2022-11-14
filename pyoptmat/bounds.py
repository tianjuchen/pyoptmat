import torch
from torch import nn


class bounding(nn.Module):
    """
    Sets up a scaling function wrapper that maps `(0,1)` to `(bounds[0], bounds[1])`
    and clips the values to remain in that range

    Args:
      bounds (tuple(torch.tensor,torch.tensor)):    tuple giving the parameter bounds

    Additional Args:
      min_bounds (torch.tensor):                    clip to avoid going lower than this value
    """

    def __init__(self, *args, scaling=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling

    def forward(self, a):
        """
        Return the actual parameter value

        Args:
          T (torch.tensor):   current temperature
        """
        return self.scaling(self.value(a))


class bound_scale_function(bounding):
    """
    A parameter that is constant with temperature

    Args:
      pvalue (torch.tensor):    the constant parameter value

    Keyword Args:
      p_scale (function):       numerical scaling function, defaults to
                                no scaling
    """

    def __init__(self, pvalue, *args, p_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.pvalue = pvalue
        self.p_scale = p_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.pvalue.device

    def value(self, a):
        """
        Pretty simple, just return the value!

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.p_scale(self.pvalue)

    @property
    def shape(self):
        """
        The shape of the underlying parameter
        """
        return self.pvalue.shape
