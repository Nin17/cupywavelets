"""_summary_
"""

import math

# import pywt
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet


def _as_wavelet(wavelet):
    # Copied from pywt/_utils.py
    """Convert wavelet name to a Wavelet object."""
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if isinstance(wavelet, ContinuousWavelet):
        raise ValueError(
            "A ContinuousWavelet object was provided, but only discrete "
            "Wavelet objects are supported by this function.  A list of all "
            "supported discrete wavelets can be obtained by running:\n"
            "print(pywt.wavelist(kind='discrete'))"
        )
    return wavelet


def _unfold_axes(data, ds, keep_no: int):
    """Unfold i.e. [batch*channel, height, widht] into [batch, channel, height, width]."""
    return data.reshape(ds[:-keep_no] + data.shape[-keep_no:])


def _fold_axes(data, keep_no: int):
    dshape = data.shape
    return data.reshape((math.prod(dshape[:-keep_no]),) + dshape[-keep_no:]), dshape
