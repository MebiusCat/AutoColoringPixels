import pygetwindow as gw
import pyautogui

import cv2
import numpy as np
import time
import TNumbers

import shutil
import os


TMP_PATH = ".\\data\\mnist_tmp\\"
SCREEN_PATH = ".\\data\\screen.png"

TOP_INDENT = 28
BOTTOM_INDENT = 190
LEFT_INDENT = 8
RIGHT_INDENT = 15
IMAGE_INDENT = 3

MAX_NUM = 100

w_pattern = [25, 26, 26, 25, 26, 26, 26, 25]
h_pattern = [26, 25, 26, 25, 26]


def get_prntscr():
    app_window = gw.getWindowsWithTitle('ColoringPixels')[0]

    image = pyautogui.screenshot(
        SCREEN_PATH,
        region=(app_window.left,
                app_window.top + TOP_INDENT,
                app_window.width,
                app_window.height - BOTTOM_INDENT))


def clean_folders():
    # make tmp directory empty for new imgs

    for i in range(MAX_NUM):
        num_path = f'{TMP_PATH}{i:02d}'
        if os.path.exists(num_path):
            shutil.rmtree(num_path)
    num_path = f'{TMP_PATH}{-1}'
    if os.path.exists(num_path):
        shutil.rmtree(num_path)


def generate_filename():
    return f'{int(time.time()) * 1000}'


def transform_to_grey(image):
    return cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]


def cutting_img_mnist():
    nums_collection = TNumbers.TNumbers()
    nums_collection.load_dict()

    clean_folders()
    cv_image_mnist = cv2.cvtColor(cv2.imread(SCREEN_PATH), cv2.COLOR_BGR2GRAY)

    num_rows, num_cols = (24, 40)
    height, width = cv_image_mnist.shape[:2]

    prev_y = IMAGE_INDENT
    for row in range(num_rows):
        prev_x = LEFT_INDENT
        for col in range(num_cols):
            # set coordinates for current cell
            x_start = prev_x
            y_start = prev_y
            x_end = x_start + w_pattern[col % len(w_pattern)]
            y_end = y_start + h_pattern[row % len(h_pattern)]

            prev_x = x_end

            sub_mnist = cv_image_mnist[
                        y_start - IMAGE_INDENT: y_end + IMAGE_INDENT,
                        x_start - IMAGE_INDENT: x_end + IMAGE_INDENT,
                        ]
            key, error = nums_collection.likelihood(sub_mnist)

            num_path = f'{TMP_PATH}{key:02d}'
            if not os.path.exists(num_path):
                os.makedirs(num_path)

            # Save img
            try:
                cv2.imwrite(f'{num_path}\\sub_{generate_filename()}_{row:02d}_{col:02d}.png', sub_mnist)
            except:
                print(f'\\sub_{row}_{col}.png')
        prev_y = y_end


get_prntscr()
cutting_img_mnist()
