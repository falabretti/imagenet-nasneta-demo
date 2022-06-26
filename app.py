import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


model = load_model()
classes = load_classes()
image = load_image('images/chair.jpg')

prediction = predict(model, classes, image)
show_result(image, prediction)
