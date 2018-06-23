# Import #################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data
##########################################################
"""An Example of a DNNClassifier for the Iris dataset."""

# Setting Up Comand Line Arguments
parser = argparse.ArgumentParser()
# Default Batch Size : 100
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# Default Step Size : 1000
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    ####################################################################################################################
    # Parameters you need to change by data ############################################################################
    # Number of Label Type
    num_type = 3
    # List of Label Type
    expected = ['Setosa', 'Versicolor', 'Virginica']
    # Sample Data to Predict
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    ####################################################################################################################
    ####################################################################################################################

    # Receive Command Line Arguments
    args = parser.parse_args(argv[1:])

    # Fetch the data
    # Just get Dataframe Sets of Tran and Test
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    # Set every feature columns as numeric
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    # Set up 10,10 Hidden Layers
    # And There are 3 types of label possibilities
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=num_type)

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    # Enable Tensorflow Error Logging Function
    tf.logging.set_verbosity(tf.logging.INFO)

    # Execute Main Function
    tf.app.run(main)
