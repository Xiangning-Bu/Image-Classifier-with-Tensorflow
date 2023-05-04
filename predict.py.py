import argparse, logging
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import glob

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)


saved_keras_model_directory = './my_classifier_model/'


def process_image(image_arr):
    image_arr = tf.cast(image_arr, tf.float32)
    image_arr = tf.image.resize(image_arr, (224, 224))
    image_arr /= 255
    return image_arr.numpy()
    
def predict(image_path, model, top_k):

    image = Image.open(image_path)
    image = np.asarray(image)
    # image process 
    image = process_image(image)
    # extra dimension
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    probs, labels = tf.nn.top_k(predictions, k=top_k)
    probs = list(probs.numpy()[0])
    labels = list(labels.numpy()[0])

    return probs, labels

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--image',default='./test_images/orange_dahlia.jpg', type = str, help='Image path')
    parser.add_argument('--model', default='./my_classifier_model', type = str, help='Classifier directory path')
    parser.add_argument('--top_k',  default=5, type=int, help='Return the top K most likely classes')
    parser.add_argument('--category_names', default='./label_map.json', type=str, help='Map labels to flower names mapping')
   
    args =parser.parse_args()

    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    my_classifier_model = tf.keras.models.load_model(saved_keras_model_directory, custom_objects={'KerasLayer':hub.KerasLayer})
    
    top_k = args.top_k

    probs, labels = predict(args.image, my_classifier_model, top_k)

    print ("\n The Top {} Classes :  \n".format(top_k))

    for i, prob, label in zip(range(1, top_k+1), probs, labels):
        print(i)
        print('Label:', label)
        print('Class name:', class_names[str(label+1)].title())
        print('Probability:', prob)

