import cv2
import numpy
import os
from datetime import date


class DnnAdapter:

    def __init__(self, args = None):
        self.model = args["prototxt"]
        self.weights = args["model"]
        self.task_type = args["task"]
        self.net = cv2.dnn.readNet(self.model, self.weights)

    def process_image(self, image):
        img = cv2.imread(image)

        if self.task_type == 'face_detection':
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            output = self.net.forward()
            self._output_face_detection(output, img)
        if self.task_type == 'classification':
            blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
            self.net.setInput(blob)
            output = self.net.forward()
            self._output_classification(output)
        if self.task_type == "road_segmentation": #Не работает
            resized_img = cv2.resize(img, (896, 512), interpolation=cv2.INTER_AREA)
            blob = cv2.dnn.blobFromImage(resized_img, 1, (896, 512), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            output = self.net.forward()
            self._output_road_segmentaotion(output, resized_img)

    def _output_classification(self, output):

        with open('Classification/synset_words.txt') as f:
            classes = [x[x.find(' ') + 1:] for x in f]
        indexes = numpy.argsort(output[0])[-5:]
        for i in reversed(indexes):
            print(i, ': class:', classes[i], ' probability:', output[0][i])

    def _output_face_detection(self, output, img):
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
        cv2.imwrite(os.getcwd() + "/output/" + self.task_type + date.today().strftime("_%Y_%m_%d") + ".jpg", img)
        cv2.imshow("Output", img)
        cv2.waitKey(0)


    #Не работает
    def _output_road_segmentaotion(self, output, resized_img):
        print(output.shape)
        print(resized_img.shape)
        rows,cols = resized_img.shape[:2]
        for i in range(rows):
            for k in range(cols):
                if output[0,0,i,k] > 0.5:
                    resized_img[i,k] = [0,0,255]
                if output[0,1,i,k] > 0.5:
                    resized_img[i,k] = [0,255,0]
                if output[0,2,i,k] > 0.5:
                    resized_img[i,k] = [255,0,0]
        cv2.imshow("Output", resized_img)
        cv2.waitKey(0)
