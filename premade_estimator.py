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
import shutil
##########################################################
"""An Example of a DNNClassifier for the Iris dataset."""

# Setting Up Comand Line Arguments
#parser = argparse.ArgumentParser()
# Default Batch Size : 100
#parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# Default Step Size : 1000
#parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')


def get_csv_from_image_folder(dir="F:/Data/Favorite", output="files.csv"):
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
    df.to_csv(output, index=False, header=False)
    print(df)

def move_image_by_csv(
        main_dir="F:/Data/Main/",
        des_dir="F:/Data/Category/",
        setting_file="C:/Users/Hijiri/ml\models/official/image_classifier/csv_data/image_result.csv",
        format=['Width', 'Height', 'isValid', 'Probability', 'File']
    ):
    data = pd.read_csv(setting_file, names=format)
    for index, row in data.iterrows():
        print("{} {} {} {} {}".format(row[0], row[1], row[2], row[3], row[4]))
        pro = row['Probability']
        target = row['File']
        des = ""
        if pro < 0.4:
            des = "0-40"
        elif pro < 0.6:
            des = "40-60"
        else:
            des = "60-100"
        des = des_dir + des + "/" + os.path.basename(target)
        shutil.move(target, des)

def main(argv):
    print("Count Starting");
    ####################################################################################################################
    # Parameters you need to change by data ############################################################################
    # Number of Label Type
    num_type = 2
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
    (train_x, train_y), (test_x, test_y), predict_x, file_name = iris_data.load_data()

    ####################################################################################################################
    # Best : 0.863
    # I will add more information
    def tfunction(t):
        t['Size'] = t['Width'] * t['Height']
        t['Ratio'] = t['Height'] / t['Width']
    tfunction(train_x)
    tfunction(test_x)
    tfunction(predict_x)
    ####################################################################################################################

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
        hidden_units=[100, 75, 50],
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
    else:
        predictions = classifier.predict(
            input_fn=lambda:iris_data.eval_input_fn(predict_x,labels=None,batch_size=args.batch_size))
        tlist = list(predictions)
        predict_x['Result'] = [i['class_ids'][0] for i in tlist]
        predict_x['Probability'] = [i['probabilities'][0] for i in tlist]
        predict_x['File Name'] = file_name
        print(predict_x)
        predict_x.to_csv("result.csv", index=False, header=False)
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    #############################################################################
    #############################################################################

if __name__ == '__main__':
    # Enable Tensorflow Error Logging Function
    tf.logging.set_verbosity(tf.logging.INFO)
    # Execute Main Function
    tf.app.run(main)
    # Get Image
    # get_csv_from_image_folder(dir="F:/Data/Search", output="files.csv")
    # Move File
    # move_image_by_csv(
    #     main_dir="F:/Data/Search/",
    #     des_dir="F:/Data/Category/",
    #     setting_file="C:/Users/Hijiri/ml\models/official/image_classifier/csv_data/image_result.csv",
    #     format=['Width', 'Height', 'isValid', 'Probability', 'File']
    # )
