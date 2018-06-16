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

# This Command Will Work
# python wide_deep.py --model_type=wide
# python wide_deep.py --model_type=deep --export_dir=wide_deep_saved_model
# tensorboard --logdir=csv_model
# saved_model_cli show --dir=wide_deep_saved_model/1529182697/
# saved_model_cli show --dir=wide_deep_saved_model/1529182697/ --tag_set serve --all
# saved_model_cli show --dir=wide_deep_saved_model/1529182697/ --tag_set serve --signature_def="predict" --input_exprs='examples=[{"width":[100.], "height":[100.]}]'

"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

# Fields Names
_CSV_COLUMNS = [
    'is_valid', 'width', 'height'
]

# Fields Default Values
_CSV_COLUMN_DEFAULTS = [[0], [0], [0]]

# Amount of data
_NUM_EXAMPLES = {
    'train': 600,
    'validation': 256,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def define_wide_deep_flags():
  """Add supervised learning flags, as well as wide-deep model type."""
  flags_core.define_base()
  flags_core.define_benchmark()

  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name="model_type", short_name="mt", default="wide_deep",
      enum_values=['wide', 'deep', 'wide_deep'],
      help="Select model topology.")

  flags_core.set_defaults(data_dir='csv_data',
                          model_dir='csv_model',
                          train_epochs=40,
                          epochs_between_evals=2,
                          batch_size=40)


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  # 'is_valid', 'width', 'height'
  width = tf.feature_column.numeric_column('width')
  height = tf.feature_column.numeric_column('height')

  base_columns = [
      # width,
      # height
  ]

  # Determine Relationship
  crossed_columns = [
      # tf.feature_column.crossed_column(
      #    ['width', 'height'], hash_bucket_size=1000)
  ]

  # wide_colums = categorical + relationship
  wide_columns = base_columns + crossed_columns

  # Everything but age_bucket (Because Age is already heere)
  deep_columns = [
      width,
      height
  ]

  # Categorical and Everything
  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
      # Use only Wide
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
      # Use only Deep
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
      #  Use both of them
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('is_valid')
    return features, tf.equal(labels, 1)

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def export_model(model, model_type, export_dir):
  """Export to SavedModel format.

  Args:
    model: Estimator object
    model_type: string indicating model type. "wide", "deep" or "wide_deep"
    export_dir: directory to export the model.
  """
  wide_columns, deep_columns = build_model_columns()
  if model_type == 'wide':
    columns = wide_columns
  elif model_type == 'deep':
    columns = deep_columns
  else:
    columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_savedmodel(export_dir, example_input_fn)


def run_wide_deep(flags_obj):
  """Run Wide-Deep training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """

  # Clean up the model directory if present
  shutil.rmtree(flags_obj.model_dir, ignore_errors=True)
  model = build_estimator(flags_obj.model_dir, flags_obj.model_type)

  train_file = os.path.join(flags_obj.data_dir, 'adult.data')
  test_file = os.path.join(flags_obj.data_dir, 'adult.test')

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, flags_obj.batch_size)

  run_params = {
      'batch_size': flags_obj.batch_size,
      'train_epochs': flags_obj.train_epochs,
      'model_type': flags_obj.model_type,
  }

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('wide_deep', 'Census Income', run_params,
                                test_id=flags_obj.benchmark_test_id)

  loss_prefix = LOSS_PREFIX.get(flags_obj.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks, batch_size=flags_obj.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    tf.logging.info('Results at epoch %d / %d',
                    (n + 1) * flags_obj.epochs_between_evals,
                    flags_obj.train_epochs)
    tf.logging.info('-' * 60)

    for key in sorted(results):
      tf.logging.info('%s: %s' % (key, results[key]))

    benchmark_logger.log_evaluation_result(results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, results['accuracy']):
      break

  # Export the model
  if flags_obj.export_dir is not None:
    export_model(model, flags_obj.model_type, flags_obj.export_dir)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_wide_deep(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_wide_deep_flags()
  absl_app.run(main)
