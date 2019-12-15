import cv2
import numpy
import argparse
import sys
import os
from datetime import date


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
    (h, w) = img.shape[:2]
    for i in range(0, output.shape[2]):
        confidence = output[0, 0, i, 2]
        if confidence > 0.5:
            box = output[0, 0, i, 3:7] * numpy.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(img, text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(os.getcwd() + "/output/" + "face_detection_sample" + date.today().strftime("_%Y_%m_%d") + ".jpg", img)
    cv2.imshow("Output", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
