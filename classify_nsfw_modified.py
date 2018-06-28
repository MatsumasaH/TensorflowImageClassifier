import os
import numpy as np
import tensorflow as tf
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader


def nsfw_main():
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
        image = fn_load_image(args.input_file)
        predictions = \
            sess.run(model.predictions,
                     feed_dict={model.input: image})

        print("Results for '{}'".format(args.input_file))
        print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))