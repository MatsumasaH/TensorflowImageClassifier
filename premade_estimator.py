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
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import gridspec
##########################################################
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
##########################################################
"""An Example of a DNNClassifier for the Iris dataset."""

# python classify_nsfw.py -m data/open_nsfw-weights.npy -l tensorflow test.jpg

# Setting Up Comand Line Arguments
#parser = argparse.ArgumentParser()
# Default Batch Size : 100
#parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# Default Step Size : 1000
#parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')

def nsfw_main(setting_file="tmp.csv"):

    data = pd.read_csv(setting_file, names=['location', 'nsfw'])
    tarray = []

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    IMAGE_LOADER_TENSORFLOW = "tensorflow"
    class args:
        pass
    args.input_file = "girl.jpg"
    args.model_weights = "data/open_nsfw-weights.npy"
    args.image_loader = IMAGE_LOADER_TENSORFLOW
    args.input_type = InputType.TENSOR.name.lower()
    model = OpenNsfwModel()
    # This is important for reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])
        sess.run(tf.global_variables_initializer())

        count = 1
        for index, row in data.iterrows():
            if count > 1000:
               break
            start = time.time()
            image = fn_load_image(row[0])
            predictions = \
                sess.run(model.predictions,
                         feed_dict={model.input: image})
            print("Results for '{}'".format(row[0]))
            print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
            tarray.append(predictions[0][1])
            print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
            count += 1

    data['nsfw'] = tarray
    data.to_csv(setting_file, index=False, header=False)
    print(data)
    return 0

def get_csv(dir="F:/Data/Main", output="C:/Users\Hijiri/ml\models/official/image_classifier/csv_data/image_files.csv"):
    files = os.listdir(dir)
    count = 0
    location = []
    for file in files:
        if count > 1000:
            break
        print(count)
        location.append(dir + "/" + file)
        count += 1
    df = pd.DataFrame()
    df['location'] = location
    df.to_csv(output, index=False, header=False)
    print(df)

def get_csv_from_image_folder(dir="F:/Data/Main", output="C:/Users\Hijiri/ml\models/official/image_classifier/csv_data/image_files.csv"):
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
            #########################################
            # Check Wheter I can Print or Not #######
            print(im.shape[0])
            print(im.shape[1])
            print(file)
            #########################################
            #########################################
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
        des_dir="F:/Data/Category/",
        setting_file="C:/Users/Hijiri/ml/models/official/image_classifier/csv_data/image_result.csv",
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


def move_image_by_csv_for_nsfw(
        des_dir="F:/Data/Category/",
        setting_file="tmp.csv",
        format=['location', 'nsfw']
    ):
    data = pd.read_csv(setting_file, names=format)
    for index, row in data.iterrows():
        print("{} {}".format(row[0], row[1]))
        pro = row[1]
        target = row[0]
        des = ""
        if pro < 0.3:
            des = "pg0"
        elif pro < 0.6:
            des = "pg30"
        elif pro < 0.9:
            des = "pg60"
        else:
            des = "pg90"
        des = des_dir + des + "/" + os.path.basename(target)
        shutil.move(target, des)

def display_data(setting_file="C:/Users\Hijiri/ml\models/official/image_classifier/csv_data/image_result.csv"):
    data = pd.read_csv(setting_file, names=['Width', 'Height', 'Size', 'Ratio', 'isValid', 'Probability', 'Name'])
    data.describe()
    data.hist('Width', bins=100)
    data.hist('Height', bins=100)
    data.hist('Size', bins=100)
    data.hist('Ratio', bins=100)
    data.hist('isValid', bins=100)
    data.hist('Probability', bins=100)
    plt.hexbin(x=data["Height"] * data["Width"], y=data["Height"] / data["Width"], C=data["Probability"])
    plt.ylim([0, 4])
    plt.show()
    plt.hexbin(x=data["Height"], y=data["Width"], C=data["Probability"])
    plt.show()
    plt.scatter(x=data["Height"] * data["Width"], y=data["Height"] / data["Width"], c=data["Probability"])
    plt.show()

def analyze_filename(folder="F:\Data\Favorite"):
    files = os.listdir(folder)
    df = pd.DataFrame()
    df['File Name'] = files
    df['Main Domain'] = df['File Name'].apply(lambda x:x.split("_")[0])
    df['Sub 1'] = df['File Name'].apply(lambda x:x.split("_")[0] + "/"  + x.split("_")[1])
    df['Sub 2'] = df['File Name'].apply(lambda x:x.split("_")[0] + "/"  + x.split("_")[1] + "/"  + x.split("_")[2])
    df2 = df.groupby('Main Domain').size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    return df2

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
    (train_x, train_y), (test_x, test_y), predict_x, file_name, result_path = iris_data.load_data()

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
        predict_x.to_csv(result_path, index=False, header=False)
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    #############################################################################
    #############################################################################

if __name__ == '__main__':
    # Get Files Information to CSV File
    get_csv_from_image_folder()
    # Enable Tensorflow Error Logging Function
    tf.logging.set_verbosity(tf.logging.INFO)
    # Execute Main Function
    tf.app.run(main)
    # Move File
    move_image_by_csv()
