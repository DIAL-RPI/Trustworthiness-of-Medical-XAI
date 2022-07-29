# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of XRAI algorithm.
Paper: https://arxiv.org/abs/1906.02825
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from base import CoreSaliency
from integrated_gradients import IntegratedGradients
import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize

_logger = logging.getLogger(__name__)

_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _normalize_image(im, value_range, resize_shape=None):
    """Normalize an image by resizing it and rescaling its values.
    Args:
        im: Input image.
        value_range: [min_value, max_value]
        resize_shape: New image shape. Defaults to None.
    Returns:
        Resized and rescaled image.
    """
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(im,
                    resize_shape,
                    order=3,
                    mode='constant',
                    preserve_range=True,
                    anti_aliasing=True)
    return im


def _get_segments_felzenszwalb(im,
                               resize_image=True,
                               scale_range=None,
                               dilation_rad=5):
    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = im.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        im = _normalize_image(im, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        im = _normalize_image(im, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(im,
                                            scale=scale,
                                            sigma=sigma,
                                            min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE)
        if resize_image:
            seg = resize(seg,
                        original_shape,
                        order=0,
                        preserve_range=True,
                        mode='constant',
                        anti_aliasing=False).astype(int)
        segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        selem = disk(dilation_rad)
        masks = [dilation(mask, selem=selem) for mask in masks]
    return masks


def _attr_aggregation_max(attr, axis=-1):
  return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
  # Compute the attr density over mask1. If mask2 is specified, compute density
  # for mask1 \ mask2
  if mask2 is None:
    added_mask = mask1
  else:
    added_mask = _get_diff_mask(mask1, mask2)
  if not np.any(added_mask):
    return -np.inf
  else:
    return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
  return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
  return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
  masks = []
  for seg in segs:
    for l in range(seg.min(), seg.max() + 1):
      masks.append(seg == l)
  return masks


class XRAIParameters(object):
  """Dictionary of parameters to specify how to XRAI and return outputs."""

  def __init__(self,
               steps=100,
               area_threshold=1.0,
               return_baseline_predictions=False,
               return_ig_attributions=False,
               return_xrai_segments=False,
               flatten_xrai_segments=True,
               algorithm='full'):
    self.steps = steps
    self.area_threshold = area_threshold
    self.return_ig_attributions = return_ig_attributions
    self.return_xrai_segments = return_xrai_segments
    self.flatten_xrai_segments = flatten_xrai_segments
    self.algorithm = algorithm
    self.experimental_params = {'min_pixel_diff': 50}


class XRAIOutput(object):
  """Dictionary of outputs from a single run of XRAI.GetMaskWithDetails."""

  def __init__(self, attribution_mask):
    self.attribution_mask = attribution_mask
    self.baselines = None
    self.ig_attribution = None
    self.segments = None


class XRAI(CoreSaliency):
  """A CoreSaliency class that computes saliency masks using the XRAI method."""

  def __init__(self):
    super(XRAI, self).__init__()
    # Initialize integrated gradients.
    self._integrated_gradients = IntegratedGradients()

#   def _get_integrated_gradients(self, im, call_model_function,
#                                 call_model_args, baselines, steps, batch_size):
#     """Takes mean of attributions from all baselines."""
#     grads = []
#     for baseline in baselines:
#       print('ii')
#       grads.append(
#           self._integrated_gradients.GetMask(
#               im,
#               call_model_function,
#               call_model_args=call_model_args,
#               x_baseline=baseline,
#               x_steps=steps,
#               batch_size=batch_size))

#     return grads

  def _get_integrated_gradients(self, model, input_image, one_hot, image_nums):
    """Takes mean of attributions from all baselines."""
    grads = []
    print('ii')
    grads.append(
          self._integrated_gradients.GetMask(model, input_image, one_hot, image_nums))
    return grads

  def _make_baselines(self, x_value, x_baselines):
    # If baseline is not provided default to im min and max values
    if x_baselines is None:
      x_baselines = []
      x_baselines.append(np.zeros_like(x_value))
    #   x_baselines.append(np.min(x_value) * np.ones_like(x_value))
    #   x_baselines.append(np.max(x_value) * np.ones_like(x_value))
    else:
      for baseline in x_baselines:
        if baseline.shape != x_value.shape:
          raise ValueError(
              'Baseline size {} does not match input size {}'.format(
                  baseline.shape, x_value.shape))
    return x_baselines

  def _predict(self, x):
    raise NotImplementedError

  def GetMask(self,
              model,
              x_value,
              one_hot,
              image_nums,
              baselines=None,
              segments=None,
              base_attribution=None,
              extra_parameters=None):
    results = self.GetMaskWithDetails(model, 
                                      x_value,
                                      one_hot,
                                      image_nums,
                                      baselines=baselines,
                                      segments=segments,
                                      base_attribution=base_attribution,
                                      extra_parameters=extra_parameters)
    return results.attribution_mask

  def GetMaskWithDetails(self,
                         model,
                         x_value,
                         one_hot,
                         image_nums,
                         baselines=None,
                         segments=None,
                         base_attribution=None,
                         extra_parameters=None):
    if extra_parameters is None:
      extra_parameters = XRAIParameters()

    # Check the shape of base_attribution.
    if base_attribution is not None:
      if not isinstance(base_attribution, np.ndarray):
        base_attribution = np.array(base_attribution)
      if base_attribution.shape != x_value.shape:
        raise ValueError(
            'The base attribution shape should be the same as the shape of '
            '`x_value`. Expected {}, got {}'.format(x_value.shape,
                                                    base_attribution.shape))

    # Calculate IG attribution if not provided by the caller.
    if base_attribution is None:
      _logger.info('Computing IG...')
      x_baselines = self._make_baselines(x_value, baselines)

    #   attrs = self._get_integrated_gradients(x_value,
    #                                          call_model_function,
    #                                          call_model_args=call_model_args,
    #                                          baselines=x_baselines,
    #                                          steps=extra_parameters.steps,
    #                                          batch_size=batch_size)
      attrs = self._get_integrated_gradients(model, x_value, one_hot, image_nums)
      # Merge attributions from different baselines.
      attr = np.mean(attrs, axis=0)
      print(attr.shape)
    else:
      x_baselines = None
      attrs = base_attribution
      attr = base_attribution

    # Merge attribution channels for XRAI input
    if len(attr.shape) > 2:
      attr = _attr_aggregation_max(attr)

    _logger.info('Done with IG. Computing XRAI...')
    if segments is not None:
      segs = segments
    else:
      segs = _get_segments_felzenszwalb(x_value)

    if extra_parameters.algorithm == 'full':
      attr_map, attr_data = self._xrai(
          attr=attr,
          segs=segs,
          area_perc_th=extra_parameters.area_threshold,
          min_pixel_diff=extra_parameters.experimental_params['min_pixel_diff'],
          gain_fun=_gain_density,
          integer_segments=extra_parameters.flatten_xrai_segments)
    elif extra_parameters.algorithm == 'fast':
      attr_map, attr_data = self._xrai_fast(
          attr=attr,
          segs=segs,
          min_pixel_diff=extra_parameters.experimental_params['min_pixel_diff'],
          gain_fun=_gain_density,
          integer_segments=extra_parameters.flatten_xrai_segments)
    else:
      raise ValueError('Unknown algorithm type: {}'.format(
          extra_parameters.algorithm))

    results = XRAIOutput(attr_map)
    results.baselines = x_baselines
    if extra_parameters.return_xrai_segments:
      results.segments = attr_data
    # TODO(tolgab) Enable return_baseline_predictions
    # if extra_parameters.return_baseline_predictions:
    #   baseline_predictions = []
    #   for baseline in x_baselines:
    #     baseline_predictions.append(self._predict(baseline))
    #   results.baseline_predictions = baseline_predictions
    if extra_parameters.return_ig_attributions:
      results.ig_attribution = attrs
    return results

  @staticmethod
  def _xrai(attr,
            segs,
            gain_fun=_gain_density,
            area_perc_th=1.0,
            min_pixel_diff=50,
            integer_segments=True):
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=float)

    n_masks = len(segs)
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}
    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
      best_gain = -np.inf
      best_key = None
      remove_key_queue = []
      for mask_key in remaining_masks:
        mask = remaining_masks[mask_key]
        # If mask does not add more than min_pixel_diff to current mask, remove
        mask_pixel_diff = _get_diff_cnt(mask, current_mask)
        if mask_pixel_diff < min_pixel_diff:
          remove_key_queue.append(mask_key)
          if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug('Skipping mask with pixel difference: {:.3g},'.format(
                mask_pixel_diff))
          continue
        gain = gain_fun(mask, attr, mask2=current_mask)
        if gain > best_gain:
          best_gain = gain
          best_key = mask_key
      for key in remove_key_queue:
        del remaining_masks[key]
      if not remaining_masks:
        break
      added_mask = remaining_masks[best_key]
      mask_diff = _get_diff_mask(added_mask, current_mask)
      masks_trace.append((mask_diff, best_gain))

      current_mask = np.logical_or(current_mask, added_mask)
      current_area_perc = np.mean(current_mask)
      output_attr[mask_diff] = best_gain
      del remaining_masks[best_key]  # delete used key
      if _logger.isEnabledFor(logging.DEBUG):
        current_attr_sum = np.sum(attr[current_mask])
        _logger.debug(
            '{} of {} masks added,'
            'attr_sum: {}, area: {:.3g}/{:.3g}, {} remaining masks'.format(
                added_masks_cnt, n_masks, current_attr_sum, current_area_perc,
                area_perc_th, len(remaining_masks)))
      added_masks_cnt += 1

    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
      masks_trace.append(uncomputed_mask)
    if integer_segments:
      attr_ranks = np.zeros(shape=attr.shape, dtype=int)
      for i, mask in enumerate(masks_trace):
        attr_ranks[mask] = i + 1
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

  @staticmethod
  def _xrai_fast(attr,
                 segs,
                 gain_fun=_gain_density,
                 area_perc_th=1.0,
                 min_pixel_diff=50,
                 integer_segments=True):
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=float)
    n_masks = len(segs)
    current_mask = np.zeros(attr.shape, dtype=bool)
    masks_trace = []

    # Sort all masks based on gain, ignore overlaps
    seg_attrs = [gain_fun(seg_mask, attr) for seg_mask in segs]
    segs, seg_attrs = list(
        zip(*sorted(zip(segs, seg_attrs), key=lambda x: -x[1])))

    for i, added_mask in enumerate(segs):
      mask_diff = _get_diff_mask(added_mask, current_mask)
      # If mask does not add more than min_pixel_diff to current mask, skip
      mask_pixel_diff = _get_diff_cnt(added_mask, current_mask)
      if mask_pixel_diff < min_pixel_diff:
        if _logger.isEnabledFor(logging.DEBUG):
          _logger.debug('Skipping mask with pixel difference: {:.3g},'.format(
              mask_pixel_diff))
        continue
      mask_gain = gain_fun(mask_diff, attr)
      masks_trace.append((mask_diff, mask_gain))
      output_attr[mask_diff] = mask_gain
      current_mask = np.logical_or(current_mask, added_mask)
      if _logger.isEnabledFor(logging.DEBUG):
        current_attr_sum = np.sum(attr[current_mask])
        current_area_perc = np.mean(current_mask)
        _logger.debug('{} of {} masks processed,'
                      'attr_sum: {}, area: {:.3g}/{:.3g}'.format(
                          i + 1, n_masks, current_attr_sum, current_area_perc,
                          area_perc_th))
    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
      masks_trace.append(uncomputed_mask)
    if integer_segments:
      attr_ranks = np.zeros(shape=attr.shape, dtype=int)
      for i, mask in enumerate(masks_trace):
        attr_ranks[mask] = i + 1
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace