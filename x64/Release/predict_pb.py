import tensorflow as tf
import numpy as np
import os,time,cv2
from tensorflow.python.platform import gfile

import utils, helpers

def predictFromPb(image_path, pb_path, crop_height, crop_width, dataset):
    print("\n******* Begin prediction *******\n")
    print("Image -->", image_path)
    print("Pb Model -->", pb_path)
    print("Crop Height -->", crop_height)
    print("Crop Width -->", crop_width)

    class_names_list, label_values = helpers.get_label_info(os.path.join("", "data/class_dict.csv"))
    num_classes = len(label_values)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        net_input = sess.graph.get_tensor_by_name('Placeholder:0')
        net_output = sess.graph.get_tensor_by_name('logits/BiasAdd:0')

        print("Testing image " + image_path)

        loaded_image = utils.load_image(image_path)
        resized_image =cv2.resize(loaded_image, (crop_width, crop_height))
        input_image = np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]),axis=0)/255.0

        st = time.time()
        output_image = sess.run(net_output,feed_dict={net_input:input_image})
        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        file_name = utils.filepath_to_name(image_path)
        cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

        print("")
        print("run time: " , run_time)
        print("Finished!")
        print("Wrote image " + "%s_pred.png"%(file_name))

