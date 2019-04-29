import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python.platform import gfile

import utils, helpers

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--pb_path', type=str, default=None, required=True, help='The path to the pb weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
#parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
#parser.add_argument('--frontend', type=str, default="ResNet50", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join("data", "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
#print("Dataset -->", args.dataset)
#print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

#net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
#net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

#network, _ = model_builder.build_model(args.model, frontend=args.frontend,
#                                        net_input=net_input,
#                                        num_classes=num_classes,
#                                        crop_width=args.crop_width,
#                                        crop_height=args.crop_height,
#                                        is_training=False)

with gfile.FastGFile(args.pb_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

net_input = sess.graph.get_tensor_by_name('Placeholder:0')
net_output = sess.graph.get_tensor_by_name('logits/BiasAdd:0')


#print('Loading model checkpoint weights')
#saver=tf.train.Saver(max_to_keep=1000)
#saver.restore(sess, args.checkpoint_path)


print("Testing image " + args.image)

loaded_image = utils.load_image(args.image)
resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

st = time.time()
output_image = sess.run(net_output,feed_dict={net_input:input_image})

run_time = time.time()-st

output_image = np.array(output_image[0,:,:,:])

output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(args.image)
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("run time: " , run_time)
print("Finished!")
print("Wrote image " + "%s_pred.png"%(file_name))
