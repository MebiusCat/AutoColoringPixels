import pygetwindow as gw
import pyautogui

import cv2
import numpy as np
import time

import shutil
import os

from CNN_numbers import ImgRecognition


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


def set_color(color, position):
    color_str = f'{color:02d}'
    pyautogui.moveTo(position)
    pyautogui.click()
    for digit in color_str:
        pyautogui.write(digit)
    time.sleep(2)


def rec_img_nn_row(log=False, save_as_files=False):
    num_rec = ImgRecognition()
    clean_folders()

    cv_image = cv2.cvtColor(cv2.imread(SCREEN_PATH), cv2.COLOR_BGR2GRAY)

    num_rows, num_cols = (24, 40)
    height, width = cv_image.shape[:2]

    matrix = []
    matrix_img = []
    prev_y = IMAGE_INDENT
    for row in range(num_rows):
        start_time = time.time()
        prev_x = LEFT_INDENT
        curr_row_img = []
        row_pp = []
        for col in range(num_cols):
            # set current coordinates
            x_start = prev_x
            y_start = prev_y
            x_end = x_start + w_pattern[col % len(w_pattern)]
            y_end = y_start + h_pattern[row % len(h_pattern)]

            prev_x = x_end

            # cut cell-image
            sub_img = cv_image[
                      y_start - IMAGE_INDENT : y_end + IMAGE_INDENT,
                      x_start - IMAGE_INDENT : x_end + IMAGE_INDENT]
            curr_row_img.append(sub_img)

            px, py = (x_start + x_end) / 2, (y_start + y_end) / 2
            row_pp.append((px, py))

        prev_y = y_end

        keys = num_rec.repcon_row(curr_row_img)

        matrix.append([(key, px, py) for key, (px, py) in zip(keys, row_pp)])
        matrix_img.append(curr_row_img)
        if save_as_files:
            for kxy, sub_mnist, col in zip(matrix[-1], matrix_img[-1], range(num_cols)):
                key, px, py = kxy
                num_path = f'{TMP_PATH}{key:02d}'
                if not os.path.exists(num_path):
                    os.makedirs(num_path)

                # Saving picture
                try:
                    cv2.imwrite(f'{num_path}\\sub_{generate_filename()}_{row:02d}_{col:02d}.png', sub_mnist)
                except:
                    print(f'\\sub_{row}_{col}.png')
        if log:
            print("Calc % row took: %s seconds ---" % (row, time.time() - start_time))
    return matrix


def color_segment(app_window, s_begin, s_end):
    pyautogui.moveTo(app_window.left + LEFT_INDENT + s_begin[0],
                     app_window.top + TOP_INDENT + s_begin[1])
    pyautogui.mouseDown(button='left',
                        x=app_window.left + LEFT_INDENT + s_begin[0],
                        y=app_window.top + TOP_INDENT + s_begin[1])
    pyautogui.moveTo(app_window.left + LEFT_INDENT + s_end[0],
                     app_window.top + TOP_INDENT + s_end[1])
    pyautogui.mouseUp(button='left',
                      x=app_window.left + LEFT_INDENT + s_end[0],
                      y=app_window.top + TOP_INDENT + s_end[1])


def coloring(mtrx):
    app_window = gw.getWindowsWithTitle('ColoringPixels')[0]

    colors = set([x[0] for x in sum(mtrx, [])])
    order = -1
    for color in colors:
        if color == -1:
            continue # Skip invalid color

        set_color(color, app_window.center)

        for row in mtrx:
            order *= -1 # Changing direction for each row
            begin, end, n_col = None, None, 0
            for cell, px, py in row[::order]:
                if cell == color:
                    if n_col > 15:  # Limit segment size
                        color_segment(app_window, begin, end)
                        begin, n_col = end, 1

                    if not begin:
                        begin = (px, py)
                    end = (px, py)
                    n_col += 1

                else:
                    if begin: # Finalize the current segment
                        color_segment(app_window, begin, end)
                        begin, n_col = None, 0
            if begin:
                color_segment(app_window, begin, end)


get_prntscr()
coloring(rec_img_nn_row(False, True))


