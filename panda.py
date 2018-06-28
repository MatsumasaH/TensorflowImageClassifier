#########################################################
#########################################################
# So Many Libraries...
from __future__ import print_function
# Math
import math
import numpy as np
# IPython
from IPython import display
# Panda
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import gridspec
# ???
from sklearn import metrics
# Tensorflow
import tensorflow as tf
from tensorflow.python.data import Dataset
import importlib
# importlib.reload(panda)
import time
start = time.time()
from tensorflow.python.client import device_lib
#########################################################
#########################################################

def get_available_gpus():
    """Display you current avairable device
    :return:
    Array of Device Infomation
    """
    get_available_gpus()
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main():
    pass


def validation_test():
    tf.logging.set_verbosity(tf.logging.ERROR)

    # run_config = tf.estimator.RunConfig().replace(
    #     session_config=tf.ConfigProto(log_device_placement=True,
    #                                   device_count={'GPU': 1}))

    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))


    def preprocess_features(california_housing_dataframe):
        """Prepares input features from California housing data set.

        Args:
          california_housing_dataframe: A Pandas DataFrame expected to contain data
            from the California housing data set.
        Returns:
          A DataFrame that contains the features to be used for the model, including
          synthetic features.
        """
        # Choose Features
        selected_features = california_housing_dataframe[
            ["latitude",
             "longitude",
             "housing_median_age",
             "total_rooms",
             "total_bedrooms",
             "population",
             "households",
             "median_income"]]
        # Copy Array
        processed_features = selected_features.copy()
        # Create a synthetic feature.
        processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"])
        # Return Array of Features Names
        return processed_features


    def preprocess_targets(california_housing_dataframe):
        """Prepares target features (i.e., labels) from California housing data set.

        Args:
          california_housing_dataframe: A Pandas DataFrame expected to contain data
            from the California housing data set.
        Returns:
          A DataFrame that contains the target feature.
        """
        # Initialize DataFrame Object
        output_targets = pd.DataFrame()
        # Scale the target to be in units of thousands of dollars.
        output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
        # Return Series Object
        return output_targets

    # Choose Training Data
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_examples.describe()

    # Choose Test Data
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))
    training_targets.describe()

    # Choose Validation Data
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_examples.describe()

    # Choose Validation Test Data
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
    validation_targets.describe()

    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()

    def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
        """Trains a linear regression model of multiple features.

        Args:
          features: pandas DataFrame of features
          targets: pandas DataFrame of targets
          batch_size: Size of batches to be passed to the model
          shuffle: True or False. Whether to shuffle the data.
          num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
        Returns:
          Tuple of (features, labels) for next data batch
        """

        # Convert pandas data into a dict of np arrays.
        features = {key: np.array(value) for key, value in dict(features).items()}

        # Construct a dataset, and configure batching/repeating.
        ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def construct_feature_columns(input_features):
        """Construct the TensorFlow Feature Columns.

        Args:
          input_features: The names of the numerical input features to use.
        Returns:
          A set of feature columns
        """
        return set([tf.feature_column.numeric_column(my_feature)
                    for my_feature in input_features])

    def train_model(
            learning_rate,
            steps,
            batch_size,
            training_examples,
            training_targets,
            validation_examples,
            validation_targets):
        """Trains a linear regression model of multiple features.

        In addition to training, this function also prints training progress information,
        as well as a plot of the training and validation loss over time.

        Args:
          learning_rate: A `float`, the learning rate.
          steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
          batch_size: A non-zero `int`, the batch size.
          training_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for training.
          training_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for training.
          validation_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for validation.
          validation_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for validation.

        Returns:
          A `LinearRegressor` object trained on the training data.
        """
        # On eachperiod, Calculate Loss
        periods = 10
        steps_per_period = steps / periods

        # Create a linear regressor object.
        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # Set up norm to avoid infinite increasing
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

        # Feed Optimizer and Feature Column Information
        linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=construct_feature_columns(training_examples),
            optimizer=my_optimizer
        )

        # Create input functions.
        training_input_fn = lambda: my_input_fn(
            training_examples,
            training_targets["median_house_value"],
            batch_size=batch_size)
        predict_training_input_fn = lambda: my_input_fn(
            training_examples,
            training_targets["median_house_value"],
            num_epochs=1,
            shuffle=False)
        predict_validation_input_fn = lambda: my_input_fn(
            validation_examples, validation_targets["median_house_value"],
            num_epochs=1,
            shuffle=False)

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        for period in range(0, periods):
            # Train the model, starting from the prior state.
            linear_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period,
            )
            # Take a break and compute predictions.
            training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
            training_predictions = np.array([item['predictions'][0] for item in training_predictions])
            # Also Validation Prediction
            validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

            # Compute training and validation loss.
            training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

            ############################################################################
            # Add the loss metrics from this period to our list.
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
            ############################################################################
        print("Model training finished.")

        # Output a graph of loss metrics over periods.
        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.legend()

        return linear_regressor

    # Calling Custom Train Model Function
    linear_regressor = train_model(
        learning_rate=0.00003,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    california_housing_test_data = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(
        test_examples,
        test_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)

    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

    print("elapsed_time:{0}".format(time.time() - start) + "[sec]")

    return 0

def tensorflowMain():
    # To Only Display / Save Error Log
    # You can disable if you want
    tf.logging.set_verbosity(tf.logging.ERROR)

    # 100 Rows would be displayed maximum
    pd.options.display.max_rows = 10
    # No column limit
    pd.set_option('display.max_columns', None)
    # No Clipping for the values
    pd.set_option('display.max_colwidth', -1)
    # 0.05 will be 0.1
    pd.options.display.float_format = '{:.1f}'.format

    # Get Data and Convert it to DataFrame Object
    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

    # Randomize Data Order
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

    # Divide Median House Value by 1000
    california_housing_dataframe["median_house_value"] /= 1000.0

    # Display Whole DataFrame
    california_housing_dataframe
    # Display Average and STD
    california_housing_dataframe.describe()

    # Define the input feature: total_rooms.
    # Input Value
    # DataFrame Class
    my_feature = california_housing_dataframe[["total_rooms"]]

    # Configure a numeric feature column for total_rooms.
    # This is just label name
    # This will go directly to linear_regressor
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]

    # Define the label.
    # DataFrame Object
    targets = california_housing_dataframe["median_house_value"]

    # Use gradient descent as the optimizer for training the model.
    # Select optimizer with Mini-Batch Stochastic Gradient Descent(SGD)
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
    # To avoid infinite loop or something?
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # Configure the linear regression model with our feature columns and optimizer.
    # Set a learning rate of 0.0000001 for Gradient Descent.
    # Also, provide feature column name(Not Data)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Feed Input to the LinearRegressor
    def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
        """Trains a linear regression model of one feature.

        Args:
          features: pandas DataFrame of features
          targets: pandas DataFrame of targets
          batch_size: Size of batches to be passed to the model
          shuffle: True or False. Whether to shuffle the data.
          num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
        Returns:
          Tuple of (features, labels) for next data batch
        """
        print("The Input Function was Called")
        # Convert pandas data into a dict of np arrays.
        # dict() : Convert list object to dictionary
        # ex. List of Tuple, Tuple of List, List of Set
        # dictionary.items() : Return List of object of tuple
        # 内包表現 - Python Comprehension
        features = {key: np.array(value) for key, value in dict(features).items()}

        # Construct a dataset, and configure batching/repeating.
        # Feed Full Datas that was received to the function
        ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
        # Just setting batch size and num epoch variables
        # num_epocks is not important, because there are 1000 steps mean 1000 * 1  = 1000 samples will be used
        # That is not even 1 epoch, so there is no meaning to set up epoch here
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()

        return features, labels

    # Now Start Actual Training
    # The reason why variable name is _ is because we only need linear_regressor object to save the data
    # The reason why it's lambda is because I guess not to execute immediately
    # Feed Features DataFrame and Targets DataFrame and Step Limit
    print("Training Starts")
    _ = linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=1000
    )
    print("Training Ends")

    # Create an input function for predictions.
    # Note: Since we're making just one prediction for each example, we don't
    # need to repeat or shuffle the data here.
    # If you just limit my_feature and targets to one value. That's it
    # You are supposed to get perfect prediction function
    prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

    # Call predict() on the linear_regressor to make predictions.
    print("Prediction Starts")
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    print("Prediction Ends")

    # Format predictions as a NumPy array, so we can calculate error metrics.
    print("Conversion Starts")
    predictions = np.array([item['predictions'][0] for item in predictions])
    print("Conversion Ended")

    # Print Mean Squared Error and Root Mean Squared Error.
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

    # You will see the difference is too big here
    min_house_value = california_housing_dataframe["median_house_value"].min()
    max_house_value = california_housing_dataframe["median_house_value"].max()
    min_max_difference = max_house_value - min_house_value
    print("Min. Median House Value: %0.3f" % min_house_value)
    print("Max. Median House Value: %0.3f" % max_house_value)
    print("Difference between Min. and Max.: %0.3f" % min_max_difference)
    print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    calibration_data.describe()
    calibration_data

    # This was horrible, And have no idea about how it went at all
    # So Now We need to visualize the information
    ####################################################################################################################
    # Graph Plotting Sample ############################################################################################
    sample = california_housing_dataframe.sample(n=300)
    # Get the min and max total_rooms values.
    x_0 = sample["total_rooms"].min()
    x_1 = sample["total_rooms"].max()
    # Retrieve the final weight and bias generated during training.
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    # Get the predicted median_house_values for the min and max total_rooms values.
    y_0 = weight * x_0 + bias
    y_1 = weight * x_1 + bias
    # Plot our regression line from (x_0, y_0) to (x_1, y_1).
    plt.plot([x_0, x_1], [y_0, y_1], c='r')
    # Label the graph axes.
    plt.ylabel("median_house_value")
    plt.xlabel("total_rooms")
    # Plot a scatter plot from our data sample.
    plt.scatter(sample["total_rooms"], sample["median_house_value"])
    # Display graph.
    plt.show()
    ####################################################################################################################
    ####################################################################################################################

    def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
        """Trains a linear regression model of one feature.

        Args:
          learning_rate: A `float`, the learning rate.
          steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
          batch_size: A non-zero `int`, the batch size.
          input_feature: A `string` specifying a column from `california_housing_dataframe`
            to use as input feature.
        """

        # 10 Loops
        periods = 10
        # Calculate steps for each period
        steps_per_period = steps / periods

        # Prepare Feature Data
        my_feature = input_feature
        my_feature_data = california_housing_dataframe[[my_feature]]

        # Prepare Label Data
        my_label = "median_house_value"
        targets = california_housing_dataframe[my_label]

        # Create feature columns.
        feature_columns = [tf.feature_column.numeric_column(my_feature)]

        # Create input functions.
        # Input, Target, Batch Size
        training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)

        # Check Whole Data Without Shuffling
        prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

        # Create a linear regressor object.
        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # Always 5
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer=my_optimizer
        )

        ################################################################################################################
        # Graph Size
        plt.figure(figsize=(15, 6))
        # Put more than 1 graph
        plt.subplot(1, 2, 1)
        # Title
        plt.title("Learned Line by Period")
        # Y Label
        plt.ylabel(my_label)
        # X Label
        plt.xlabel(my_feature)
        # Number of Samples
        sample = california_housing_dataframe.sample(n=300)
        # Scatter Plot Setting
        plt.scatter(sample[my_feature], sample[my_label])
        # Just make Color Array
        colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
        ################################################################################################################

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("RMSE (on training data):")
        root_mean_squared_errors = []
        for period in range(0, periods):
            # Train the model, starting from the prior state.
            linear_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period
            )
            # Take a break and compute predictions.
            predictions = linear_regressor.predict(input_fn=prediction_input_fn)
            predictions = np.array([item['predictions'][0] for item in predictions])


            # Compute loss.
            # Basically, Comparing Predictions and Targets
            root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(predictions, targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            root_mean_squared_errors.append(root_mean_squared_error)
            # Finally, track the weights and biases over time.
            # Apply some math to ensure that the data and line are plotted neatly.
            # Calculate Maximum Height
            y_extents = np.array([0, sample[my_label].max()])
            # Sprintf like
            weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
            bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
            # Do some complicated calculation here
            x_extents = (y_extents - bias) / weight
            x_extents = np.maximum(np.minimum(x_extents,
                                              sample[my_feature].max()),
                                   sample[my_feature].min())
            y_extents = weight * x_extents + bias
            # Just Plot Different Color Graph
            plt.plot(x_extents, y_extents, color=colors[period])


        print("Model training finished.")

        ################################################################################################################
        # Output a graph of loss metrics over periods.
        plt.subplot(1, 2, 2)
        plt.ylabel('RMSE')
        plt.xlabel('Periods')
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(root_mean_squared_errors)
        ################################################################################################################

        # Output a table with calibration data.
        calibration_data = pd.DataFrame()
        calibration_data["predictions"] = pd.Series(predictions)
        calibration_data["targets"] = pd.Series(targets)
        display.display(calibration_data.describe())

        print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
        return calibration_data
    ####################################################################################################################
    # train_model(
    #     learning_rate=0.00002,
    #     steps=3000,
    #     batch_size=10
    # )
    ####################################################################################################################
    train_model(
        learning_rate=0.00002,
        steps=500,
        batch_size=5,
        input_feature="population"
    )
    # Create New Series on the dataframe
    california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

    # Train Again with Different Input feature
    calibration_data = train_model(
        learning_rate=0.05,
        steps=500,
        batch_size=5,
        input_feature="rooms_per_person")

    # Create a graph of Predictions VS Targets
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(calibration_data["predictions"], calibration_data["targets"])

    # Display Hist Graph of the data
    plt.subplot(1, 2, 2)
    _ = california_housing_dataframe["rooms_per_person"].hist()

    # Clip the Data by using Lambda
    california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

    _ = california_housing_dataframe["rooms_per_person"].hist()

    # Train Again with Clipped Data
    calibration_data = train_model(
        learning_rate=0.05,
        steps=500,
        batch_size=5,
        input_feature="rooms_per_person")

    # Plot Again to see the difference
    _ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])

def pandaDemo():
    print("Panda Demo")
    # Set panda not to truncate
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', -1)

    # Show Version
    # print(pd.__version__)

    # Prepare Data Set
    cities = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    # print(cities)

    # Create Data Frame
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    population = pd.Series([852469, 1015785, 485199])
    df = pd.DataFrame({ 'City name': city_names, 'Population': population })
    # print(df)

    # Get Data From CSV (Network)
    california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
    # print(california_housing_dataframe.describe())

    # Showing Only Head
    # print(california_housing_dataframe.head())

    # Make Graph
    # You can not on this environment
    california_housing_dataframe.hist('housing_median_age')

    # DataFrame is just like an array
    cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
    # print(type(cities['City name']))
    # print(cities['City name'])

    # You can also access by index
    # print(type(cities['City name'][1]))
    # print(cities['City name'][1])

    # Also you can specify range
    # print(type(cities[0:2]))
    # print(cities[0:2])

    # You can also do this
    # print(population / 1000)

    # Or this
    # print(np.log(population))
    # Or this
    # print(population + 100)

    # Or this
    # print(population.apply(lambda val: val > 1000000))

    # This is how you modify data
    cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
    cities['Population density'] = cities['Population'] / cities['Area square miles']
    # print(cities)

    # You can do also complicated things like this
    cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
    # print(cities)

    # Examples of Indexing
    # print(city_names.index)
    # print(cities.index)
    # cities.reindex([2, 0, 1])

    # How to shuffle
    cities.reindex(np.random.permutation(cities.index))
    # print(cities)

if __name__ == '__main__':
    main()