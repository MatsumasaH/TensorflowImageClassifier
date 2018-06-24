# Import Necessary Libraries
import pandas as pd
import tensorflow as tf
########################################################################################################################
########################################################################################################################
# URLs of Train and Test data sets
TRAIN_URL = "C:/Users\Hijiri/ml\models/official/image_classifier/csv_data/image_train.csv"
TEST_URL = "C:/Users/Hijiri/ml\models/official/image_classifier/csv_data/image_test.csv"
isFileLocal = 1
# Label Name
LABEL_NAME = "isValid"
# CSV Fields Structure
CSV_COLUMN_NAMES = ['Width', 'Height', 'isValid']
# Label Possibilities
SPECIES = ['Valid', 'None Valid']
########################################################################################################################
########################################################################################################################

def maybe_download():
    """Just Download it to local"""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name=LABEL_NAME):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    if isFileLocal == 0:
        train_path, test_path = maybe_download()
    else:
        train_path, test_path = TRAIN_URL, TEST_URL

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES)
    # Recognize Last Series as Label Column
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES)
    # Recognize Last Series as Label Column
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

########################################################################################################################
########################################################################################################################
# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
########################################################################################################################
########################################################################################################################

