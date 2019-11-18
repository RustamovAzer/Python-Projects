import cv2
import numpy
import argparse


def main():
    ap = argparse.ArgumentParser(description="Console interface for various machine-learning models")
    ap.add_argument("-p", "--prototxt", required=True, help="path to 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    ap.add_argument("-i", "--image", required=True, help="path to image")
    args = vars(ap.parse_args())
    net = cv2.dnn.readNet(args["prototxt"], args["model"])
    img = cv2.imread(args["image"])
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    output = net.forward()
    with open('Classification/synset_words.txt') as f:
        classes = [x[x.find(' ') + 1:] for x in f]
    indexes = numpy.argsort(output[0])[-5:]
    for i in reversed(indexes):
        print(i, ': class:', classes[i], ' probability:', output[0][i])