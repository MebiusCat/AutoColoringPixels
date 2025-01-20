import keras
import tensorflow as tf

import cv2
import numpy as np

PWD = ".\\data\\model_final.h5"

class ImgRecognition:
    def __init__(self):
        self.model = keras.models.load_model(PWD)
        self.labels = {x: x - 1 if x != 1 else -x for x in range(100)}

    def repcon(self, img):
        image_size = (31, 31, 1)
        img_array = cv2.resize(img, image_size[:2])
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = self.model.predict(img_array)
        prediction = self.labels[np.argmax(prediction)]
        return prediction

    def repcon_row(self, batch):
        image_size = (31, 31, 1)
        processed_images = []
        for img in batch:
            img_array = cv2.resize(img, image_size[:2])
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            processed_images.append(np.squeeze(img_array, axis=0))

        predictions = self.model.predict(np.array(processed_images))
        result = []
        for probs in predictions:
            prediction = self.labels[np.argmax(probs)]
            result.append(prediction)
        return result