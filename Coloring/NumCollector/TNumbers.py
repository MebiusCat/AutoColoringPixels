import os
import re
import cv2
import numpy as np
from PIL.ImageChops import difference


PWD = ".\\data\\dict\\"


def transform_to_grey(image):
    return cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

		
class TNumbers():
    def __init__(self):
        self.numbers: dict(NumPoint) = dict()

    def __str__(self):
        return str(self.numbers)

    def load_dict(self):

        for file in os.listdir(PWD):
            image = transform_to_grey(cv2.imread(f"{PWD}\\{file}"))
            num = int(re.search(re.compile('\d+'), file).group())
            if num not in self.numbers:
                self.numbers[num] = NumPoint(num)
            self.numbers[num].add_print(image)


    def likelihood(self, image, log=False):
        num_subtract = []

        if np.mean(image[6: -6, 6: -6]) in (255, 0):
            return (-1, 0)

        for k, t_print in self.numbers.items():
            footprint_like = []
            for t_num in t_print.footprint:
                weight, height = t_num.weight, t_num.height
                x_diff, y_diff = image.shape[0] - weight + 1, image.shape[1] - height + 1

                difference = [sum((t_num.footprint - image[dx:dx + weight, dy: dy + height]).flatten())
                              for dx in range(x_diff) for dy in range(y_diff)]

                if difference:
                    value = min(difference)
                    if value < 300:
                        if log:
                            print(f'easy {t_print.num} value: {value}')
                        return (t_print.num, value)
                    footprint_like.append((t_print.num, value))
                    if value > 15000:
                        break

            num_subtract.append(min(footprint_like, key=lambda x: x[1]))

        if not num_subtract:
            return -1
        if log:
            print(sorted(num_subtract, key=lambda x: x[1])[:5])
        min_value = min(num_subtract, key=lambda x: x[1])
        return (-1, 0) if min_value[1] > 22000 else min_value


class NumPoint:
    def __init__(self, num: int):
        self.num = num
        self.footprint:list(Footprint) = []

    def __str__(self):
        return f'{self.num}'

    def add_print(self, footprint):
        self.footprint.append(Footprint(footprint))


class Footprint:
    def __init__(self, footprint):
        self.footprint = footprint
        self.weight = footprint.shape[0]
        self.height = footprint.shape[1]