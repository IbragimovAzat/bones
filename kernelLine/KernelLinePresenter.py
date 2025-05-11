from PIL import Image
# from scipy.spatial import ConvexHull
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


class KernelLinePresenter:

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
            cv2.circle(image, tuple(point), 25,
                       (0, 0, 255), -1)  # Красные точки

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
        model = YOLO('static/neuralModels/golen.pt')
        output = []
        results = model.predict(imageName, save=False,
                                save_txt=False, save_conf=False)
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

    def getCenterPointsOfContour(points, topPoint, bottomPoint, margin, step=5):
        """
        Получает центральные координаты кости по вертикальным срезам.
        :param points: список точек контура (список [x, y])
        :param topPoint: верхняя точка области
        :param bottomPoint: нижняя точка области
        :param margin: отступ от краёв
        :param step: шаг по оси Y
        :return: список центральных точек, список Y-координат
        """
        points = np.array(points)
        contourPoints = []
        yContourPoints = []

        minY = int(topPoint[1] + margin)
        maxY = int(bottomPoint[1] - margin)

        for y in range(minY, maxY, step):
            close_points = points[np.abs(points[:, 1] - y) < step]
            print(f"Y={y}, close_points={len(close_points)}")  # отладка

            if len(close_points) >= 2:
                leftmost = close_points[np.argmin(close_points[:, 0])]
                rightmost = close_points[np.argmax(close_points[:, 0])]
                width = abs(rightmost[0] - leftmost[0])
                # отладка
                print(
                    f"  Leftmost={leftmost}, Rightmost={rightmost}, Width={width}")

                if width > 20:
                    center_x = int((leftmost[0] + rightmost[0]) / 2)
                    contourPoints.append((center_x, y))
                    yContourPoints.append(y)
        print('contourPoints')
        print(contourPoints)
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
            res = KernelLinePresenter.getBreakPoints(
                b, sliceContour[i + 1], sliceContour[i + 2])
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
            slope1 = KernelLinePresenter.calculate_slope(
                group[i-1][1], group[i][1])
            slope2 = KernelLinePresenter.calculate_slope(
                group[i][1], group[i+1][1])

            if slope1 == float('inf') or slope2 == float('inf'):
                continue  # Игнорируем вертикальные участки

            slopeChange = abs(slope2 - slope1)
            angle = KernelLinePresenter.calculate_angle_between_lines(
                slope1, slope2)

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

        if KernelLinePresenter.getIncline(topPoint, breakPoint[1]):
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
        if KernelLinePresenter.getIncline(bottomPoint, breakPoint[1]):
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
        angle_radians = math.atan(
            abs((slope2 - slope1) / (1 + slope1 * slope2)))
        angle_degrees = math.degrees(angle_radians)
        return 180 - angle_degrees

    def prepare_input(points, input_dim=200):
        # Предполагаем, что points — это список вида [[x1, y1], [x2, y2], ...]
        # Нужно получить 100 X-координат и 100 Y-координат
        num_points = len(points)
        target_points = 100  # Количество точек для каждой координаты (X и Y)

        # Извлекаем X и Y координаты
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Если точек больше 100, обрезаем
        if num_points > target_points:
            x_coords = x_coords[:target_points]
            y_coords = y_coords[:target_points]
        # Если точек меньше 100, заполняем нулями
        elif num_points < target_points:
            x_coords += [0] * (target_points - num_points)
            y_coords += [0] * (target_points - num_points)

        # Объединяем X и Y в один массив
        # [x1, ..., x100, y1, ..., y100]
        input_array = np.array(x_coords + y_coords)

        # Преобразуем в форму (1, 200) для подачи в модель
        input_array = input_array.reshape(1, input_dim)

        return input_array

    def getGolenLine(workingDirectory, imageName):
        # data = image.imread('./BedroImages/bedro 18.jpg')
        # plt.imshow(data)
        # plt.show()
        # return
        plt.switch_backend('Agg')
        points = KernelLinePresenter.getBounds(imageName)
        points = points[0]

        # points_array = np.array(points)
        points_array = KernelLinePresenter.prepare_input(points)

        # topVectorX = points_array[:, 0]
        # topVectorY = points_array[:, 1]
        model = tf.keras.models.load_model(
            "static/neuralModels/golenDots.keras")
        predictions = model.predict(points_array)

        topPoint = [predictions[0][0], predictions[0][1]]
        bottomPoint = [predictions[0][2], predictions[0][3]]

        margin = KernelLinePresenter.getPercentHeight(topPoint, bottomPoint)

        contourPoints, yContourPoints = KernelLinePresenter.getCenterPointsOfContour(
            points, topPoint, bottomPoint, margin)

        print("points")
        print(points)

        print("topPoint")
        print(topPoint)

        print("bottomPoint")
        print(bottomPoint)

        print("margin")
        print(margin)
        sliceCenterContour = contourPoints

        groupOfPoints = KernelLinePresenter.getBreakPointsOfGroup(
            sliceCenterContour)

        breakPoint = KernelLinePresenter.getMinPointsBreak(groupOfPoints)

        plt.plot(breakPoint[1][0], breakPoint[1][1],
                 marker='o', color='green', markersize='4')

        # topPoint, bottomPoint = getMinMaxCenterContourPoints(
        #     contourPoints, yContourPoints)
        linePoints = KernelLinePresenter.getLinePoints(
            breakPoint, topPoint, bottomPoint, 4)

        # data = image.imread(imageName)
        image_path = os.path.join(workingDirectory, "output.jpg")

        if os.path.isfile(image_path):
            data = image.imread(image_path)
        else:
            data = image.imread(imageName)
        x = [topPoint[0], breakPoint[1][0], bottomPoint[0]]
        y = [topPoint[1], breakPoint[1][1], bottomPoint[1]]

        plt.plot(x, y, color="red", linewidth=2)
        plt.imshow(data)
        plt.savefig(image_path, format="jpg")

        # folder_path = f'static/{newName}/predict'
        # files = os.listdir(folder_path)

        # loaded_file = ""
        # done_file = ""
        # for file in files:
        #     newName += f"/predict/{file}"
        # newName = "static/" + newName

        # Расчет угла между точками
        # Предполагаем, что breakPoint[1] - это точка разрыва, а topPoint и bottomPoint - точки на линиях
        slope_top = KernelLinePresenter.calculate_slope(
            breakPoint[1], topPoint)
        slope_bottom = KernelLinePresenter.calculate_slope(
            breakPoint[1], bottomPoint)

        angle_between_lines = KernelLinePresenter.calculate_angle_between_lines(
            slope_top, slope_bottom)

        isRazrez = ""
        if (170 < angle_between_lines < 190):
            isRazrez = "Остеотомия не требуется"
        else:
            isRazrez = "Требуется разрез"

        return image_path, angle_between_lines, isRazrez
