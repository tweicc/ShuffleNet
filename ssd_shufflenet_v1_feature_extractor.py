# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""SSDFeatureExtractor for ShufflenetV1 features."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
import collections

slim = tf.contrib.slim



def channel_shuffle(inputs, group_nums=8):
    h, w, c = inputs.shape.as_list()[1:]
    assert c % group_nums == 0
    input_reshaped = tf.reshape(inputs, [-1, h, w, group_nums, c // group_nums])
    input_transposed = tf.transpose(input_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(input_transposed, [-1, h, w, c])
    return output


def group_conv(inputs, output_depth, group_nums=8, kernel_size=(1,1), stride=1, scope=None, use_activations=False):
    h, w, c = inputs.shape.as_list()[1:]
    assert c % group_nums == 0
    assert output_depth % group_nums == 0
    with tf.variable_scope(scope, "group_conv",[inputs]):
        input_splits = tf.split(value=inputs, num_or_size_splits=group_nums, axis=3)
        num_channels_in_group = output_depth // group_nums
        outputs = [slim.conv2d(input_split, num_channels_in_group, kernel_size, stride,
                               activation_fn=tf.nn.relu6 if use_activations else None) for input_split in input_splits]
    return tf.concat(outputs, axis=3)

end_points = collections.OrderedDict()

Conv = collections.namedtuple('Conv', ['kernel', 'stride', 'depth'])
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """

def bottleneck(inputs, depth, group_num=8, stride=1, scope=None, keyindex=None):
    with tf.variable_scope(scope, 'shufflenet_v1_bottleneck', [inputs]) as sc:
        h, w, c = inputs.shape.as_list()[1:]
        if stride == 1:
            shortcut = inputs
            depth_bottleneck = depth / 4
        else:
            shortcut = slim.avg_pool2d(inputs, [3, 3], stride, padding="SAME")
            depth -= c
            depth_bottleneck = depth / 4

        residual = group_conv(inputs, depth_bottleneck, group_num, scope='gconv_1', use_activations=True)
        if group_num != 1:
            residual = channel_shuffle(residual, group_num)

        residual = slim.separable_conv2d(residual, None, 3, 1, stride,
                                         activation_fn=None, scope='sepconv2')
        residual = group_conv(residual, depth, group_num, scope='gconv_2')

        if stride == 1:
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu6(tf.concat([shortcut, residual], axis=3))
        end_points[keyindex] = output
        return output





def stack_blocks_dense(net, blocks):
  """Stacks ResNet `Blocks` and controls output feature density.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1



  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):


            net = block.unit_fn(net,  keyindex=block.scope+'/unit_%d' % (i + 1),**unit)



  return net



def shufflenetv1_block(scope, depth, group_num ,num_units, stride):
    return Block(scope, bottleneck, [{
        'depth': depth,
        'group_num': group_num,
        'stride': stride

    }] + [{
        'depth': depth,
        'group_num': group_num,
        'stride': 1
    }] * (num_units - 1))



start_conv = Conv(kernel=[3, 3], stride=2, depth=32)



blocks = [
    shufflenetv1_block('block1_1',depth=384, group_num=1, num_units=1, stride=2),
    shufflenetv1_block('block1_2',depth=384, group_num=8, num_units=3, stride=1),
    shufflenetv1_block('block2',depth=768, group_num=8, num_units=8, stride=2),
    shufflenetv1_block('block3',depth=1536, group_num=8, num_units=4, stride=2),
]

def shufflenet_v1_base(preprocessed_inputs):

    net = preprocessed_inputs
    net = slim.conv2d(net, start_conv.depth, start_conv.kernel,
                      stride=start_conv.stride,
                      normalizer_fn=slim.batch_norm,
                      scope="Conv2d_s")

    end_points["Conv2d_s"] = net
    net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')

    end_points['pool1'] = net
    net = stack_blocks_dense(net, blocks)

    return end_points

class SSDShuffleNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV1 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None):
    """MobileNetV1 Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(SSDShuffleNetV1FeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs, fpn=0):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    feature_map_layout = {
        'from_layer': ['block2/unit_8', 'block3/unit_4'],
        'layer_depth': [-1, -1],
    }

    with slim.arg_scope(self._conv_hyperparams):
        with slim.arg_scope([slim.batch_norm], fused=False):
            image_features = shufflenet_v1_base(preprocessed_inputs)
            for i, j in image_features.items():
                print i, j.get_shape()
            feature_maps = feature_map_generators.multi_resolution_feature_maps(
                feature_map_layout=feature_map_layout,
                depth_multiplier=self._depth_multiplier,
                min_depth=self._min_depth,
                insert_1x1_conv=True,
                image_features=image_features)

    return  feature_maps.values()


