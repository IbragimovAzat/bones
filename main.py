
import torch
import cv2
from matplotlib import image
from matplotlib import pyplot as plt
from ultralytics import YOLO
import random
import os
import shutil


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


def getXByY(contour, y):
    """
    Получает левую и правую границу кости по Y.
    В резульате выдаёт массив координат границ ( x,y)
    :param contour:
    :param y:
    :return: array
    """
    cords = []
    y = round(y)
    for i in contour:
        if i[1] == y and i not in cords:
            cords.append(i)
    cords.sort(key=lambda x: x[0], reverse=True)
    return cords


# def check(contour,  axis):
#     """
#     Проверяет правильно ли построена точка разреза
#     :param contour:
#     :param axis:
#     :return: bool
#     """
#     contour.sort(key=lambda x: x[1], reverse=True)
#     for u in axis:
#         val = getXByY(contour, u[1])
#         x_min = val[-1][0]
#         x_max = val[0][0]
#         if u[0] < (x_min + ((x_max - x_min) / 3)) or u[0] > (x_max - ((x_max - x_min) / 3)):
#             return False
#     return True

def check(contour,  axis):
    """
    Проверяет правильно ли построена точка разреза
    :param contour:
    :param axis:
    :return: bool
    """
    contour.sort(key=lambda x: x[1], reverse=True)
    print(axis)
    print(axis[1])
    for u in axis:
        print(u[1])
        val = getXByY(contour, u[1])
        if not val:
            return False
        x_min = val[-1][0]
        x_max = val[0][0]
        if u[0] < (x_min + ((x_max - x_min) / 3)) or u[0] > (x_max - ((x_max - x_min) / 3)):
            return False
    return True


def readImage(image):
    """
    Считывает изображение по ссылке на картинку
    :param image:
    :return: img, height, width
    """
    img = cv2.imread(image)
    height, width, ch = img.shape
    return img.copy(), height, width


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

# принимает на вход результат работы модели


def getPointsOfBone(width, height, result):
    """
    Превращает контур кости в массив координат кости
    :param width:
    :param height:
    :param result:

    :return:
    """
    points = []
    for x in range(0, width):
        for y in range(0, height):
            if (result[y, x, 2] < 100):
                points.append([x, y])
                cv2.circle(result, (x, y), 1, (0, 0, 255), -1)
    points.sort(key=lambda x: x[1], reverse=True)
    return points


def getBorderPositions(points):
    """
    Получает позиции границ кости
    :param points:
    :return: array, array
     """
    min_x = min(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_x = max(points, key=lambda x: x[0])[0]
    max_y = max(points, key=lambda x: x[1])[1]

    min_point = [min_x, min_y]
    max_point = [max_x, max_y]
    output = [min_point, max_point]
    return output
    print(f"Минимальная точка: {min_point}")
    print(f"Максимальная точка: {max_point}")

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
            if abs(i[1] - e[1]) < 10 and abs(i[0] - e[0]) > 30 \
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


def getMinPointsBreak(group):
    """
    Ищет точку разрыва среди возможных точек разрыва
    Возвращает [[x,y],[x,y],[x,y], radius]
    :param group:
    :return: array
    """
    minRadius = float('inf')
    minPoint = []

    for point in group:
        if point[3] < minRadius:
            minRadius = point[3]
            minPoint = point
    return minPoint

def getMinMaxCenterContourPoints(contourPoints, yContourPoints):
    """
    Ищет наименьшую и наивысшую точки из центральных точек
    :param contourPoints:
    :param yContourPoints:
    :return: array, array
    """
    for i in contourPoints:
        if i[1] == min(yContourPoints):
            topPoint = i
        if i[1] == max(yContourPoints):
            bottomPoint = i
    return topPoint, bottomPoint

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


def getBoundsNew(imageName):
    model = YOLO('bedro.pt')
    output = []
    results = model.predict(imageName, conf=0.8, save=True)

    for result in results:
        for box in result.masks.xy:
            output.append(box.tolist())
    output.sort(key=lambda point: (point[0], point[1]))
    output_filtered = [output[0][i] for i in range(len(output[0])) if i % 1 == 0]
    source_directory = 'runs/segment'
    new_folder_name = 'segment_' + str(random.randint(1000, 9999))
    target_directory = os.path.join('static', new_folder_name)
    shutil.move(source_directory, target_directory)

    return output_filtered, new_folder_name
def main(imageName):
    plt.switch_backend('Agg')

    points, newName = getBoundsNew(imageName)

    topPoint, bottomPoint = getBorderPositions(points)

    margin = getPercentHeight(topPoint, bottomPoint)

    contourPoints, yContourPoints = getCenterPointsOfContour(
        points, topPoint, bottomPoint, margin)

    sliceCenterContour = contourPoints

    groupOfPoints = getBreakPointsOfGroup(sliceCenterContour)

    breakPoint = getMinPointsBreak(groupOfPoints)

    plt.plot(breakPoint[1][0], breakPoint[1][1],
             marker='o', color='green', markersize='4')

    topPoint, bottomPoint = getMinMaxCenterContourPoints(
        contourPoints, yContourPoints)
    linePoints = getLinePoints(breakPoint, topPoint, bottomPoint, 4)

    data = image.imread(imageName)
    x = [topPoint[0], breakPoint[1][0], bottomPoint[0]]
    y = [topPoint[1], breakPoint[1][1], bottomPoint[1]]
    plt.plot(x, y, color="red", linewidth=2)
    plt.imshow(data)
    filename = f'static/my_plot{random.randint(0, 1000000)}.png'
    plt.savefig(filename)

    folder_path = f'static/{newName}/predict'
    files = os.listdir(folder_path)

    for file in files:
        newName += f"/predict/{file}"

    newName = "static/" + newName

    return filename, newName

# if (main('images/5XB_4jXB7Gg.jpg')):
#     print('Линия входит в границу кости')
# else:
#     print('Линия не входит в границу кости')


