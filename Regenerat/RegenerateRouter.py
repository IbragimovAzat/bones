import json
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from ultralytics import YOLO
import os
from matplotlib import image
from datetime import datetime
import shutil
from Regenerat.RegeneratePresenter import RegeneratePresenter


class RegenerateRouter:
    def run_regenerate_analysis(folderPath, img_path, annot_path, model_name):
        return RegeneratePresenter.handleData(folderPath, img_path, annot_path, model_name)

        # plt.switch_backend('Agg')

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # result_dir = f'static/results/{timestamp}'
        # os.makedirs(result_dir, exist_ok=True)

        # points, new_name = getBoundsNew(image_path)
        # top_point, bottom_point = getBorderPositions(points)
        # margin = getPercentHeight(top_point, bottom_point)
        # contour_points, y_contour_points = getCenterPointsOfContour(points, top_point, bottom_point, margin)

        # slice_center_contour = contour_points
        # group_of_points = getBreakPointsOfGroup(slice_center_contour)
        # break_point = getMinPointsBreak(group_of_points)

        # plt.plot(break_point[1][0], break_point[1][1], marker='o', color='green', markersize=4)

        # top_point, bottom_point = getMinMaxCenterContourPoints(contour_points, y_contour_points)
        # line_points = getLinePoints(break_point, top_point, bottom_point, 4)

        # img_data = image.imread(image_path)
        # x = [top_point[0], break_point[1][0], bottom_point[0]]
        # y = [top_point[1], break_point[1][1], bottom_point[1]]
        # plt.plot(x, y, color="red", linewidth=2)
        # plt.imshow(img_data)

        # output_path = os.path.join(result_dir, "output.png")
        # plt.savefig(output_path)

        # shutil.copy(image_path, os.path.join(result_dir, "input.png"))

        # return output_path, result_dir
