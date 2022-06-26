import os
import sys
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_image_path():
    if len(sys.argv) < 2:
        print('No image file specified!')
        exit()

    file = sys.argv[1]

    if not os.path.isfile(file):
        print('The provided file is invalid!')
        exit()

    return os.path.abspath(file)


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


def show_result(image, prediction):
    plt.imshow(image)
    plt.title(prediction, fontdict={ 'size': 20 })
    plt.show()


file_path = get_image_path()
model = load_model()
classes = load_classes()
image = load_image(file_path)

prediction = predict(model, classes, image)
show_result(image, prediction)
