#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
#code from keypoint example code modified to produce confidence values for emotions

class KeyPointClassifier(object):
    def __init__(
            self,
            model_path='model/keypoint_classifier/keypoint_classifier.tflite',
            num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.last_index = 2
        self.result = 0.00 #stores the confidence values as array

    def __call__(
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        self.result = self.interpreter.get_tensor(output_details_tensor_index)
        #returns the confidence values for each emotion based of the classifier label as a range of 0 to 1 as a float number
        print('confidence value for Angry:', float(self.result[0][0]),'\nconfidence value for Happy:',float(self.result[0][1]),'\nconfidence value for Neutral: ',float(self.result[0][2]),
              '\nconfidence value for Sad: ',float(self.result[0][3]),'\nconfidence value for Surprised',float(self.result[0][4]),'\n---------------------------\n')
        if np.max(self.result) >= 0.85:
            result_index = np.argmax(np.squeeze(self.result))
            self.last_index = result_index
            return result_index
        else:
            return self.last_index

    def getResult(self):
        return self.result
