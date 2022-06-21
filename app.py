import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

print(tf.__version__)

def load_model():
    # https://tfhub.dev/google/imagenet/nasnet_large/classification/5
    model = hub.load('imagenet_nasnet_large_classification_5/')
    return model

def load_classes():
    # https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
    with open('ImageNetLabels.txt', 'r') as f:
        classes = f.readlines()
        classes = [c.replace('\n', '') for c in classes]
        return classes

def load_image(path):
    image = Image.open(path)
    image = image.resize((331, 331), Image.Resampling.LANCZOS)
    image = np.apply_along_axis(lambda x: x / 255., 2, image)
    return image

def predict(model, classes, image):
    predictions = model([image])
    class_idx = np.argmax(predictions)
    result = classes[class_idx]
    return result

model = load_model()
classes = load_classes()
image = load_image('images/barn.jpg')

result = predict(model, classes, image)
print(result)
