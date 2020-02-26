# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

from scipy import misc
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys
import time

# process command arguments
phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)
print("phone = ", phone)
print("dped_dir = ", dped_dir)
print("test_subnet = ", test_subset)
print("iteration = ", iteration)
print("resolution = ", resolution)
print("use_gpu = ", use_gpu)

# get all available image resolutions
res_sizes = utils.get_resolutions()
print("res_sizes = ", res_sizes)

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)
print("IMAGE_HEIGHT = ", IMAGE_HEIGHT)
print("IMAGE_WIDTH = ", IMAGE_WIDTH)
print("IMAGE_SIZE = ", IMAGE_SIZE)

# disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0},log_device_placement=True) if use_gpu == "false" else None


# create placeholders for input images
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:
    test_dir = dped_dir + phone.replace("_orig", "") + "/test_data/full_size_test_images/"
    print("test_dir = ", test_dir)

    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]
    print("test_photos = ", test_photos)
    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models_orig/" + phone)

        for photo in test_photos:
            print("photo = ", photo)

            # load training image and crop it if necessary

            print("Testing original " + phone.replace("_orig", "") + " model, processing image " + photo)
            image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255
            print("np.float16 ended")

            image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
            print("utils.extract_crop")
            image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])
            print("np.reshape ended")

            # get enhanced image
            print("sess.run started")
            start_time = time.time()
            enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
            print("--- %s seconds ---" % (time.time() - start_time))
            print("sess.run ended")
            enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            print("np.reshape ended")

            before_after = np.hstack((image_crop, enhanced_image))
            print("np.hstack ended")
            photo_name = photo.rsplit(".", 1)[0]
            # photo_name = "food"
            print("photo_name ", photo_name)

            # save the results as .png images
            print("saving image")

            misc.imsave("visual_results/" + phone + "_" + photo_name + "_enhanced.png", enhanced_image)
            print("saved image")
            misc.imsave("visual_results/" + phone + "_" + photo_name + "_before_after.png", before_after)

    else:

        num_saved_models = int(len([f for f in os.listdir("models/") if f.startswith(phone + "_iteration")]) / 2)

        if iteration == "all":
            iteration = np.arange(1, num_saved_models) * 1000
        else:
            iteration = [int(iteration)]

        print("iterations = ", iteration)
        for i in iteration:

            # load pre-trained model
            print("i = ", i)
            saver = tf.train.Saver()
            saver.restore(sess, "models/" + phone + "_iteration_" + str(i) + ".ckpt")

            for photo in test_photos:
                # load training image and crop it if necessary

                print("iteration " + str(i) + ", processing image " + photo)
                image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255

                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
                image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

                # get enhanced image

                enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
                enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

                before_after = np.hstack((image_crop, enhanced_image))
                photo_name = photo.rsplit(".", 1)[0]

                # save the results as .png images

                misc.imsave("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png",
                            enhanced_image)
                misc.imsave("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png",
                            before_after)
