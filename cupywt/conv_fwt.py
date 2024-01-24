"""
Convolution based fast wavelet transform.
"""

import warnings

import numpy as np
import pywt

# from cupy import asarray, pad
# from cupyx.scipy.signal import convolve
from numpy import asarray, pad
from scipy.signal import convolve

from .utils import _as_wavelet, _fold_axes, _unfold_axes

__all__ = ("wavedec",)


# def _preprocess_array_dec1d(data):
#     ds = None
#     if len(data.shape) == 1:
#         # add channel and batch dimension.
#         data = data[None, None]
#     elif len(data.shape) == 2:
#         # add the channel dimension.
#         data = data[:, None]
#     else:
#         data, ds = _fold_axes(data, 1)
#         data = data[:, None]
#     return data, ds


# def _postprocess_result_list_dec1d(result_lst, ds):
#     return [_unfold_axes(i, ds, 1) for i in result_lst]
#     # unfold_list = []
#     # for fres in result_lst:
#     #     unfold_list.append(_unfold_axes(fres, ds, 1))
#     # return unfold_list


# def _check_level(sizes, dec_lens, level):
#     if np.isscalar(sizes):
#         sizes = (sizes,)
#     if np.isscalar(dec_lens):
#         dec_lens = (dec_lens,)
#     max_level = min(pywt.dwt_max_level(s, d) for s, d in zip(sizes, dec_lens))
#     if level is None:
#         level = max_level
#     elif level < 0:
#         raise ValueError("Level value of %d is too low . Minimum level is 0." % level)
#     elif level > max_level:
#         warnings.warn(
#             (
#                 "Level value of {} is too high: all coefficients will experience "
#                 "boundary effects."
#             ).format(level)
#         )
#     return level


# def _fwt_pad(data, filt_len: int, mode: str = "reflect"):
#     if mode == "zero":
#         # translate pywt to numpy.
#         mode = "constant"
#     padr = (2 * filt_len - 3) // 2
#     padl = (2 * filt_len - 3) // 2
#     # pad to even signal length.
#     if data.shape[-1] % 2 != 0:
#         padr += 1
#     data = pad(data, ((0, 0),) * (data.ndim - 1) + ((padl, padr),), mode)
#     return data


# def wavedec(data, wavelet, mode="symmetric", level=None, axis=-1):
#     wavelet = _as_wavelet(wavelet)
#     try:
#         axes_shape = data.shape[axis]
#     except IndexError as error:
#         raise np.AxisError("Axis greater than data dimensions") from error
#     level = _check_level(axes_shape, wavelet.dec_len, level)

#     if axis != -1:
#         data = data.swapaxes(axis, -1)
#     data, ds = _preprocess_array_dec1d(data)

#     dec_lo, dec_hi, _, _ = wavelet.filter_bank
#     dec_lo, dec_hi = asarray(dec_lo), asarray(dec_hi)

#     filt_len = dec_lo.size

#     result_list = []
#     res_lo = data
#     print(level)
#     for _ in range(level):
#         res_lo = _fwt_pad(res_lo, filt_len, mode)
#         # TODO check that the indexing with None is correct.
#         res_lo_new = convolve(res_lo, dec_lo[None, None], mode="valid")[..., ::2]
#         res_hi = convolve(res_lo, dec_hi[None, None], mode="valid")[..., ::2]
#         # print(ndimage.convolve1d(res_lo, dec_lo).shape)
#         res_lo = res_lo_new
#         # TODO check the shapes - squeeze might not be necessary.
#         result_list.append(res_hi.squeeze(1))
#     result_list.append(res_lo.squeeze(1))
#     result_list.reverse()

#     print([i.shape for i in result_list])
#     # !!! something going wrong - different to my other implementation.

#     if ds:
#         result_list = _postprocess_result_list_dec1d(result_list, ds)

#     if axis != -1:
#         # result_list = [i.swapaxes(axis, -1) for i in result_list]
#         swap = []
#         for coeff in result_list:
#             swap.append(coeff.swapaxes(axis, -1))
#         result_list = swap
#     return result_list


def _check_level(sizes, dec_lens, level):
    if np.isscalar(sizes):
        sizes = (sizes,)
    if np.isscalar(dec_lens):
        dec_lens = (dec_lens,)
    max_level = min(pywt.dwt_max_level(s, d) for s, d in zip(sizes, dec_lens))
    if level is None:
        level = max_level
    elif level < 0:
        raise ValueError("Level value of %d is too low . Minimum level is 0." % level)
    elif level > max_level:
        warnings.warn(
            (
                "Level value of {} is too high: all coefficients will experience "
                "boundary effects."
            ).format(level)
        )
    return level

def _fwt_pad(data, filt_len: int, mode: str = "reflect"):
    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2
    # pad to even signal length.
    if data.shape[-1] % 2 != 0:
        padr += 1
    data = pad(data, ((0, 0),) * (data.ndim - 1) + ((padl, padr),), mode)
    return data

def _preprocess_array_dec1d(data):
    ds = None
    if len(data.shape) == 1:
        # add channel and batch dimension.
        data = data[None, None]
    elif len(data.shape) == 2:
        # add the channel dimension.
        data = data[:, None]
    else:
        data, ds = _fold_axes(data, 1)
        data = data[:, None]
    return data, ds

def _postprocess_result_list_dec1d(result_lst, ds):
    return [_unfold_axes(i, ds, 1) for i in result_lst]

def _unfold_axes(data, ds, keep_no: int):
    """Unfold i.e. [batch*channel, height, widht] into [batch, channel, height, width]."""
    return data.reshape(ds[:-keep_no] + data.shape[-keep_no:])

def wavedec(data, wavelet, mode="symmetric", level=None, axis=-1):
    wavelet = _as_wavelet(wavelet)
    try:
        axes_shape = data.shape[axis]
    except IndexError as error:
        raise np.AxisError("Axis greater than data dimensions") from error
    level = _check_level(axes_shape, wavelet.dec_len, level)

    if axis != -1:
        data = data.swapaxes(axis, -1)
    data, ds = _preprocess_array_dec1d(data)

    dec_lo, dec_hi, _, _ = wavelet.filter_bank
    dec_lo, dec_hi = asarray(dec_lo), asarray(dec_hi)

    filt_len = dec_lo.size

    result_list = []
    res_lo = data
    print(level)
    for _ in range(level):
        res_lo = _fwt_pad(res_lo, filt_len, mode)
        # TODO check that the indexing with None is correct.
        res_lo_new = convolve(res_lo, dec_lo[None, None], mode="valid")[..., ::2]
        res_hi = convolve(res_lo, dec_hi[None, None], mode="valid")[..., ::2]
        # print(ndimage.convolve1d(res_lo, dec_lo).shape)
        res_lo = res_lo_new
        # TODO check the shapes - squeeze might not be necessary.
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))
    result_list.reverse()

    print([i.shape for i in result_list])
    # !!! something going wrong - different to my other implementation.

    if ds:
        result_list = _postprocess_result_list_dec1d(result_list, ds)

    if axis != -1:
        # result_list = [i.swapaxes(axis, -1) for i in result_list]
        swap = []
        for coeff in result_list:
            swap.append(coeff.swapaxes(axis, -1))
        result_list = swap
    return result_list