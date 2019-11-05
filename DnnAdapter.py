import cv2
import numpy


class DnnAdapter:

    def __init__(self, model_path, weights_path=None, task_type=None):  # и другие параметры пустым значением по умолчанию
        self.model = model_path
        self.weights = weights_path
        self.task_type = task_type

    def process_image(self, image):
        # Read image
        img = cv2.imread(image)
        # forward

        if self.task_type == 'face_detection':
            net = cv2.dnn.readNetFromCaffe(self.model, self.weights)
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(img, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(img, text, (startX, y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imshow("Output", img)
            cv2.waitKey(0)
        if self.task_type == 'classification':
            blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
            net = cv2.dnn.readNetFromCaffe(self.model, self.weights)
            net.setInput(blob)
            output = net.forward()
            self._output_classification(output)

    def _output_classification(self, output):

        with open('Classification/synset_words.txt') as f:
            classes = [x[x.find(' ') + 1:] for x in f]
        indexes = numpy.argsort(output[0])[-5:]
        for i in reversed(indexes):
            print('class:', classes[i], ' probability:', output[0][i])
