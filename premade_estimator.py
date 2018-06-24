# Import #################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start = time.time()
import argparse
import tensorflow as tf
import iris_data
import pandas as pd
##########################################################
import cv2
import imghdr
import os
import pandas as pd
##########################################################
"""An Example of a DNNClassifier for the Iris dataset."""

# Setting Up Comand Line Arguments
#parser = argparse.ArgumentParser()
# Default Batch Size : 100
#parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# Default Step Size : 1000
#parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')


def get_csv_from_image_folder(dir="F:/Data/Favorite"):
    files = os.listdir(dir)
    count = 1
    width = []
    height = []
    location = []
    ####################################################################################################################
    ####################################################################################################################
    for file in files:
        # if count > 50:
        #    break
        print(count)
        try:
            im = cv2.imread(dir + "/" + file)
            location.append(dir + "/" + file)
            width.append(im.shape[0])
            height.append(im.shape[1])
        except:
            pass
        count += 1
    ####################################################################################################################
    ####################################################################################################################
    df = pd.DataFrame()
    df['width'] = width
    df['height'] = height
    df['location'] = location
    df.to_csv("files.csv", index=False, header=False)
    print(df)

def main(argv):
    print("Count Starting");
    ####################################################################################################################
    # Parameters you need to change by data ############################################################################
    # Number of Label Type
    num_type = 2
    # List of Label Type
    expected = ['Valid', 'None Valid']
    # Sample Data to Predict
    predict_x = {
        'Width': [1000, 100, 400, 220],
        'Height': [1000, 100, 200, 200],
    }
    CSV_COLUMN_NAMES = ['Width', 'Height', 'isValid']
    predict_x = pd.read_csv(
        "C:/Users\Hijiri/ml\models/official/image_classifier/csv_data/image_train.csv",
        names=CSV_COLUMN_NAMES).drop("isValid", axis=1)
    # Dummy Class for Jupyter Notepad
    class args:
        pass
    # Batch Size
    args.batch_size = 100
    # Train Steps
    args.train_steps = 10000
    # If you don't need to test
    isTraining = 0
    ####################################################################################################################
    ####################################################################################################################

    # Receive Command Line Arguments
    #args = parser.parse_args(argv[1:])

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
        hidden_units=[10, 10, 10],
        # The model must choose between 3 classes.
        n_classes=num_type,
        model_dir="premade_estimator_model"
    )

    #############################################################################
    #############################################################################
    if isTraining == 1:
        # Train the Model.
        for i in range(1, 10):
            classifier.train(
                input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                         args.batch_size),
                steps=args.train_steps/10)
        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                    args.batch_size))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    #############################################################################
    #############################################################################

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))
    tlist = list(predictions)
    predict_x['Result'] = [i['class_ids'][0] for i in tlist]
    predict_x['Probability'] = [i['probabilities'][0] for i in tlist]
    print(predict_x)
    print("elapsed_time:{0}".format(time.time() - start) + "[sec]")

if __name__ == '__main__':
    # Enable Tensorflow Error Logging Function
    tf.logging.set_verbosity(tf.logging.INFO)
    # Execute Main Function
    tf.app.run(main)
    # Get Image
    # get_csv_from_image_folder()
