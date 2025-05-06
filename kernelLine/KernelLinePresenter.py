from PIL import Image
from shapely import MultiPolygon
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import torch
import cv2
from matplotlib import image
from matplotlib import pyplot as plt
from ultralytics import YOLO
import random
import os
import shutil
from matplotlib import image as mpimg
import numpy as np
import math
import random
import tensorflow as tf


def show_image_with_points(image_path, points):
    """
    Отображает изображение с заданными точками.

    :param image_path: Путь к изображению.
    :param points: Список точек в формате [[x1, y1], [x2, y2], ...].
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не найдено.")
        return

    for point in points:
        cv2.circle(image, tuple(point), 25, (0, 0, 255), -1)  # Красные точки

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Пример использования
# points = [[686, 1542], [921, 3577]]  # Координаты точек
# show_image_with_points("./BedroImages/bedro 6.jpg", points)


def getBounds(imageName):
    model = YOLO('./bedro.pt')
    output = []
    results = model.predict(imageName, conf=0.8, save=True)
    for result in results:
        for box in result.masks.xy:
            output.append(box.tolist())
    output.sort(key=lambda point: (point[0], point[1]))

    return output


def getPercentHeight(topPoint, bottomPoint):
    """
    Получает отступ для кости
    :param topPoint:
    :param bottomPoint:
    :return:
    """
    return abs(topPoint[1] - bottomPoint[1])*0.15


def getCenterPointsOfContour(points, topPoint, bottomPoint, margin):
    """
    Получает центральные координаты кости
    :param points:
    :param topPoint:
    :param bottomPoint:
    :param margin:
    :return: array, array
    """
    contourPoints = []
    yContourPoints = []

    for i in points:
        for e in points:
            r1 = (round((i[0] + e[0]) / 2), e[1])
            if abs(i[1] - e[1]) < 30 and abs(i[0] - e[0]) > 50 \
                    and bottomPoint[1] - i[1] > margin and i[1] - topPoint[1] > margin \
                    and r1 not in contourPoints and r1 not in yContourPoints:
                contourPoints.append(r1)
                yContourPoints.append(r1[1])

    return contourPoints, yContourPoints


def getBreakPointsOfGroup(sliceContour):
    """
    Метод группирует возможные точки разрыва кости и их радиус
    :param sliceContour:
    :return: array
    """
    group = []
    for i, b in enumerate(sliceContour):
        if i + 2 >= len(sliceContour):
            break
        res = getBreakPoints(b, sliceContour[i + 1], sliceContour[i + 2])
        group.append([b, sliceContour[i + 1], sliceContour[i + 2], res])
    return group


# def getMinPointsBreak(group):
#     """
#     Ищет точку разрыва среди возможных точек разрыва, учитывая изменение угла наклона.
#     Возвращает [[x,y],[x,y],[x,y], radius]
#     """
#     minRadius = float('inf')
#     minPoint = []
#     minSlopeChange = float('inf')  # Добавляем критерий изменения наклона

#     maxAngle = float('-inf')  # Для хранения максимального угла
#     minAngle = float('inf')

#     for i, point in enumerate(group):
#         if i == 0 or i == len(group) - 1:
#             continue  # Пропускаем крайние точки

#         # Углы между соседними точками
#         slope1 = calculate_slope(group[i-1][1], group[i][1])
#         slope2 = calculate_slope(group[i][1], group[i+1][1])

#         slopeChange = abs(slope2 - slope1)  # Изменение угла наклона

#         if slopeChange < minSlopeChange or point[3] < minRadius:
#             minRadius = point[3]
#             minSlopeChange = slopeChange
#             minPoint = point
#     print(f"🔹 Наибольший угол: {maxAngle:.2f}°")
#     print(f"🔹 Наименьший угол: {minAngle:.2f}°")
#     return minPoint


def getMinPointsBreak(group):
    minRadius = float('inf')
    minPoint = []
    minSlopeChange = float('inf')

    maxAngle = float('-inf')  # Для хранения максимального угла
    minAngle = float('inf')   # Для хранения минимального угла

    for i in range(1, len(group) - 1):  # Пропускаем первую и последнюю точки
        slope1 = calculate_slope(group[i-1][1], group[i][1])
        slope2 = calculate_slope(group[i][1], group[i+1][1])

        if slope1 == float('inf') or slope2 == float('inf'):
            continue  # Игнорируем вертикальные участки

        slopeChange = abs(slope2 - slope1)
        angle = calculate_angle_between_lines(slope1, slope2)

        # Обновляем минимальный и максимальный угол
        if angle > maxAngle:
            maxAngle = angle
        if angle < minAngle:
            minAngle = angle

        if slopeChange < minSlopeChange or group[i][3] < minRadius:
            minRadius = group[i][3]
            minSlopeChange = slopeChange
            minPoint = group[i]

    print(f"🔹 Наибольший угол: {maxAngle:.2f}°")
    print(f"🔹 Наименьший угол: {minAngle:.2f}°")

    return minPoint


def getBreakPoints(a, b, c):
    """
    Метод получает точки разрыва
    :param a:
    :param b:
    :param c:

    :return:
    """
    if a[1] == b[1] == c[1]:  # если все точки на одной вертикали
        return float('inf')
    if a[1] == b[1]:  # если ab вертикал.
        b, c = c, b
    elif b[1] == c[1]:  # если bc вертикал.
        a, b = b, a

    f1 = []  # построение сер. перпендикуляра к ab (y = kx + t)
    f1 += [-1 * (b[0] - a[0]) / (b[1] - a[1])]  # k
    f1 += [(a[1] + b[1]) / 2 - f1[0] * (a[0] + b[0]) / 2]  # t

    f2 = []  # построение сер. перпендикуляра к bc (y = kx + t)
    f2 += [-1 * (c[0] - b[0]) / (c[1] - b[1])]  # k
    f2 += [(c[1] + b[1]) / 2 - f2[0] * (c[0] + b[0]) / 2]  # t

    if f1[0] == f2[0]:  # все точки на 1 прямой
        return float('inf')
    # точка пересечения сер. перп. (центр опис. окр.)
    x = (f2[1] - f1[1]) / (f1[0] - f2[0])
    y = f1[0] * x + f1[1]

    r = ((a[0] - x) ** 2 + (a[1] - y) ** 2) ** 0.5  # вычисление радиуса
    return r


def getIncline(point, centralPoint):
    """
        Получает наклон кости (False - лево, True - право )
    :param point:
    :param centralPoint:
    :return:
    """
    if point[0] > centralPoint[0]:
        return False
    else:
        return True


def getLinePoints(breakPoint, topPoint, bottomPoint, density):
    """
    Ищет точки на центральной линии
    :param breakPoint:
    :param topPoint:
    :param bottomPoint:
    :param density:
    :return: array
    """
    centerPoints = []
    d_x = abs(breakPoint[1][0] - topPoint[0]) / density
    d_y = abs(breakPoint[1][1] - topPoint[1]) / density

    if getIncline(topPoint, breakPoint[1]):
        for i in range(1, density + 1):
            plt.plot(breakPoint[1][0] - (d_x * i), breakPoint[1][1] -
                     (d_y * i), marker='o', color='cyan', markersize='4')
            centerPoints.append(
                [breakPoint[1][0] + (d_x * i), breakPoint[1][1] - (d_y * i)])
    else:
        for i in range(1, density + 1):
            plt.plot(breakPoint[1][0] + (d_x * i), breakPoint[1][1] -
                     (d_y * i), marker='o', color='cyan', markersize='4')
            centerPoints.append(
                [breakPoint[1][0] + (d_x * i), breakPoint[1][1] - (d_y * i)])
    d_x = abs(bottomPoint[0] - breakPoint[1][0]) / density
    d_y = abs(bottomPoint[1] - breakPoint[1][1]) / density
    if getIncline(bottomPoint, breakPoint[1]):
        for i in range(1, density + 1):
            plt.plot(breakPoint[1][0] - (d_x * i), breakPoint[1][1] +
                     (d_y * i), marker='o', color='cyan', markersize='4')
            centerPoints.append(
                [breakPoint[1][0] - (d_x * i), breakPoint[1][1] + (d_y * i)])
    else:
        for i in range(1, density + 1):
            plt.plot(breakPoint[1][0] + (d_x * i), breakPoint[1][1] +
                     (d_y * i), marker='o', color='cyan', markersize='4')
            centerPoints.append(
                [breakPoint[1][0] - (d_x * i), breakPoint[1][1] + (d_y * i)])
    return centerPoints


# def calculate_slope(point1, point2):
#     return (point2[1] - point1[1]) / (point2[0] - point1[0])

def calculate_slope(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    if dx == 0:  # Если вертикальная линия
        return float('inf')  # Вертикальный наклон

    return dy / dx


def calculate_angle_between_lines(slope1, slope2):
    # Угол между двумя линиями
    angle_radians = math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
    angle_degrees = math.degrees(angle_radians)
    return 180 - angle_degrees


def getBedroLine(imageName):
    # data = image.imread('./BedroImages/bedro 18.jpg')
    # plt.imshow(data)
    # plt.show()
    # return

    points = getBounds(imageName)
    points = points[0]

    # topPoint, bottomPoint = getBorderPositions(points)
    # topPoint = tf.keras.models.load_model(
    #     "keras/highPoint.keras")
    # bottomPoint = tf.keras.models.load_model("keras/lowPoint.keras")

    # bedro 6
    # topPoint = [674, 1585]
    # bottomPoint = [931, 3495]

    # bedro 47
    topPoint = [819, 2027]
    bottomPoint = [864, 3583]

    # bedro 18
    # topPoint = [467, 1697]
    # bottomPoint = [470, 3406]

    points_array = np.array(points)

    topVectorX = points_array[:, 0]
    topVectorY = points_array[:, 1]

    # Удаление 100 случайных точек
    desired_count = 100
    if len(topVectorX) > desired_count:
        # Генерация случайных индексов для выборки 100 точек
        indices = random.sample(range(len(topVectorX)), desired_count)
        topVectorX = topVectorX[indices]
        topVectorY = topVectorY[indices]

    margin = getPercentHeight(topPoint, bottomPoint)

    contourPoints, yContourPoints = getCenterPointsOfContour(
        points, topPoint, bottomPoint, margin)

    sliceCenterContour = contourPoints

    groupOfPoints = getBreakPointsOfGroup(sliceCenterContour)

    breakPoint = getMinPointsBreak(groupOfPoints)

    plt.plot(breakPoint[1][0], breakPoint[1][1],
             marker='o', color='green', markersize='4')

    # topPoint, bottomPoint = getMinMaxCenterContourPoints(
    #     contourPoints, yContourPoints)
    linePoints = getLinePoints(breakPoint, topPoint, bottomPoint, 4)

    # data = image.imread(imageName)
    image_path = os.path.join('./imageLine', "predict/image0.jpg")

    if os.path.isfile(image_path):
        data = image.imread(image_path)
    else:
        data = image.imread(imageName)
    x = [topPoint[0], breakPoint[1][0], bottomPoint[0]]
    y = [topPoint[1], breakPoint[1][1], bottomPoint[1]]

    # x = []
    # y = []
    # for point in points:
    #     x.append(point[0])
    #     y.append(point[1])
    plt.plot(x, y, color="red", linewidth=2)
    plt.imshow(data)
    filename = f'static/my_plot{random.randint(0, 1000000)}.png'
    plt.show()

    plt.savefig(filename)

    folder_path = f'static/{newName}/predict'
    files = os.listdir(folder_path)

    loaded_file = ""
    done_file = ""
    for file in files:
        newName += f"/predict/{file}"
    newName = "static/" + newName

    # Расчет угла между точками
    # Предполагаем, что breakPoint[1] - это точка разрыва, а topPoint и bottomPoint - точки на линиях
    slope_top = calculate_slope(breakPoint[1], topPoint)
    slope_bottom = calculate_slope(breakPoint[1], bottomPoint)

    angle_between_lines = calculate_angle_between_lines(
        slope_top, slope_bottom)

    isRazrez = ""
    if (170 < angle_between_lines < 190):
        isRazrez = "Остеотомия не требуется"
    else:
        isRazrez = "Требуется разрез"

    return filename, newName, angle_between_lines, isRazrez
