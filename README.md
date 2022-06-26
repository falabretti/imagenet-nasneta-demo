# Imagenet classification with NASNet-A demo

This is a demo of image classification with [imagenet/nasnet_large/classification](https://tfhub.dev/google/imagenet/nasnet_large/classification/5).

## Installation

Install dependencies with pip:
```
python -m pip install requirements.txt
```

## Download model and classes

You can download the model [here](https://tfhub.dev/google/imagenet/nasnet_large/classification/5). Extract the model (if you downloaded it compressed) and save it under a directory named `imagenet_nasnet_large_classification_5/`.

You can download the prediction classes [here](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt). Save it under a text file named `ImageNetLabels.txt`.

## Run
```
python app.py <input_image>
```

Where `input_image` can be any image file. Example with one of the example images:
```
python app.py ./images/chair.jpg  
```
