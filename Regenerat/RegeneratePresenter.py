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

# Результат применения модели к изображению


class RegeneratePresenter:
    def get_res_apply_model(model, img):
        return model(img, verbose=True)

    # Получения контура маски в виде массива

    def get_mask_contours(img_np_array, img):
        color = (255, 0, 0)
        mask = img_np_array[0].masks.data[0].cpu().numpy()

        # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
        mask_resized = cv2.resize(
            np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_contours, _ = cv2.findContours(mask_resized.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, mask_contours, -1, color, 5)

        return mask_contours

    # Рисование точек внутри квадратика (контур учитывается)

    def draw_points_in_squares_contours(img, mask_contours, num_squares, num_points):
        square_points = []  # точки сегмента
        square_list_data_coord = []  # координаты сегмента

        for contour in mask_contours:
            x, y, w, h = cv2.boundingRect(contour)
            step = h // num_squares
            for i in range(num_squares):
                points_in_square = []

                # Учитывем, чтобы квадрат включал в себя разметку (не больше, не меньше)
                start_x = x
                start_y = y + i * step
                end_x = x + w
                end_y = start_y + step

                square_list_data_coord.append([start_x, start_y, end_x, end_y])

                # Функции для отображения результата
                # commented
                # test_img = img.copy()
                # cropped_image = test_img[start_y:end_y, start_x:end_x]

                # show_current_drawing(test_img)
                # cv2.imshow("cropped", cropped_image)

                cv2.rectangle(img, (start_x, start_y),
                              (end_x, end_y), (0, 255, 0), 2)
                for _ in range(num_points):
                    rx = random.randint(start_x, end_x)
                    ry = random.randint(start_y, end_y)
                    # Учитываем контур кости, точки должны быть внутри
                    if cv2.pointPolygonTest(contour, (rx, ry), False) > 0:
                        # Раскомментировать - посмотреть как проставлены точки
                        # cv2.circle(img, (rx, ry), 3, (0, 0, 255), -1)
                        points_in_square.append((rx, ry))
                square_points.append(points_in_square)

        return square_list_data_coord, square_points

    # Вывод изображения

    def show_current_drawing(img):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(img)
        return ax

    def calculate_average_color(points, img):
        total_points = len(points)
        if total_points == 0:
            return (0, 0, 0)
        total_b = sum([img[y, x][0] for x, y in points])
        total_g = sum([img[y, x][1] for x, y in points])
        total_r = sum([img[y, x][2] for x, y in points])
        average_color = (total_b // total_points, total_g //
                         total_points, total_r // total_points)
        return average_color

    # Рисование градиента на основании среднего цвета каждого квадрата
    # Функция нужна для графического отображения. Функционально значения не иммет

    def show_current_gradient(avg_color_in_square_points):
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=100)
        for i, color in enumerate(avg_color_in_square_points):
            ax.add_patch(plt.Rectangle(
                (i, 0), 1, 1, color=np.array(color)/255))
        ax.set_xlim(0, len(avg_color_in_square_points))
        ax.set_ylim(0, 1)
        ax.axis('off')

        fig.tight_layout()
        fig.canvas.draw()
        img_gradient = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_gradient = img_gradient.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img_gradient

    # Вычисление Евклидова расстояния

    def eqlid(dictary):
        values = list(dictary.values())
        keys = list(dictary.keys())

        return math.sqrt(np.sum([a*b for a, b in zip(np.array(values) / np.sum(np.array(values)), np.square(keys))]))

    # Гистограмма для каждого квадратика
    # Функциональную роль не имеет

    def show_hist_square(points_in_square, num, img):
        intensities = []  # Интенсивность
        for point in points_in_square:
            x, y = point
            # Предполагается, что изображение в оттенках серого и имеет один цветовой канал
            intensity = img[y, x][0] / 255.0
            intensities.append(intensity)
        # print(intensities)

        # Вектор (x - интенсивность, y - частота)
        intensity_frequencies = {0.1: 0, 0.2: 0, 0.3: 0,
                                 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1: 0}
        for intensity in intensities:
            intens = (round(intensity, 1))
            # print(intens)
            try:
                intensity_frequencies[intens] = intensity_frequencies.get(
                    intens, 0) + 1
            except:
                pass
        intensity_frequencies1 = dict(sorted(intensity_frequencies.items()))

        # print(f"{eqlid(intensity_frequencies1)}")
        # print(intensity_frequencies1)
        # print(list(intensity_frequencies1.values()))
        # intensity_frequencies.to_csv('result.csv', index = False)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.hist(intensities, bins=256, range=(0.0, 1.0), fc='k', ec='k')
        ax.set_xlabel("Значение интенсивности")
        ax.set_ylabel("Частота")
        ax.set_title(f'Гистограмма для квадрата {num}')

        # Преобразование графика в изображение для более удобного вывода. Необязательно
        fig.tight_layout()
        fig.canvas.draw()
        img_hist = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        img_hist = img_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        width, height = fig.canvas.get_width_height()
        expected_size = width * height * 3
        actual_size = img_hist.size

        if actual_size != expected_size:
            scale = int(np.sqrt(actual_size / expected_size))
            img_hist = img_hist.reshape((height * scale, width * scale, 3))
        else:
            img_hist = img_hist.reshape((height, width, 3))

        # width, height = fig.canvas.get_width_height()
        # img_hist = img_hist.reshape((height, width, 3))

        plt.close(fig)
        return img_hist, RegeneratePresenter.eqlid(intensity_frequencies1)

    # Для преобразования json разметки в удобный для работы формат контура

    def reverse_json_np(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        points = np.array(data['shapes'][0]['points'], dtype=np.int32)
        return np.array([points])

    # Объединение квадратов в зону регенерата

    def generate_img_by_square_nums(img, square_list_data_coord, *nums):
        temp_img = img.copy()
        generate_square = []
        nums = sorted(nums[0])
        for num in nums:
            generate_square.append(square_list_data_coord[num])

        generate_coord = [0, 0, 0, 0]

        start_x, start_y, _, _ = generate_square[0]
        generate_coord[0] = start_x
        generate_coord[1] = start_y

        _, _, end_x, end_y = generate_square[-1]
        generate_coord[2] = end_x
        generate_coord[3] = end_y

        start_x, start_y, end_x, end_y = generate_coord
        cropped_image = temp_img[start_y:end_y, start_x:end_x]

        # Для отображения, что получилось
        RegeneratePresenter.show_current_drawing(cropped_image)
        return cropped_image

    # Гистограмма для каждого квадратика

    def show_hist_square(points_in_square, num, img):
        intensities = []  # Интенсивность
        for point in points_in_square:
            x, y = point
            # Предполагается, что изображение в оттенках серого и имеет один цветовой канал
            intensity = img[y, x][0] / 255.0
            intensities.append(intensity)
        # print(intensities)

        # Вектор (x - интенсивность, y - частота)
        intensity_frequencies = {0.1: 0, 0.2: 0, 0.3: 0,
                                 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1: 0}
        for intensity in intensities:
            intens = (round(intensity, 1))
            # print(intens)
            try:
                intensity_frequencies[intens] = intensity_frequencies.get(
                    intens, 0) + 1
            except:
                pass
        intensity_frequencies1 = dict(sorted(intensity_frequencies.items()))

        #############################################
        print(intensity_frequencies1)
        print(f"{RegeneratePresenter.eqlid(intensity_frequencies1)}")
        # print(intensity_frequencies1)
        # print(list(intensity_frequencies1.values()))
        # intensity_frequencies.to_csv('result.csv', index = False)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.hist(intensities, bins=256, range=(0.0, 1.0), fc='k', ec='k')
        ax.set_xlabel("Значение интенсивности")
        ax.set_ylabel("Частота")
        ax.set_title(f'Гистограмма для квадрата {num}')

        # Преобразование графика в изображение для более удобного вывода. Необязательно
        fig.tight_layout()
        fig.canvas.draw()
        img_hist = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # img_hist = img_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        width, height = fig.canvas.get_width_height()
        expected_size = width * height * 3
        actual_size = img_hist.size

        if actual_size != expected_size:
            scale = int(np.sqrt(actual_size / expected_size))
            img_hist = img_hist.reshape((height * scale, width * scale, 3))
        else:
            img_hist = img_hist.reshape((height, width, 3))

        plt.close(fig)
        return img_hist, RegeneratePresenter.eqlid(intensity_frequencies1)

    def create_regenerat_statistic(img, img_for_drawing, avg_color_in_square_points, square_points):
        # Разметка для полотна рисования
        num_squares = len(square_points)
        num_rows = num_squares // 2 + 3
        num_cols = 2

        fig = plt.figure(figsize=(16, num_rows * 6))
        grid = fig.add_gridspec(num_rows, num_cols)

        # Градиент
        # ax_gradient = fig.add_subplot(grid[0, :]) # Занимает две верхние ячейки
        # # Строится на основании средних цветов
        # img_gradient = show_current_gradient(avg_color_in_square_points)
        # ax_gradient.imshow(img_gradient)
        # ax_gradient.axis('off')

        # Исходное и обработанное изображения
        # ax_orig = fig.add_subplot(grid[1, 0])
        # ax_orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # ax_orig.set_title('Исходное изображение')
        # ax_orig.axis('off')

        ax_processed = fig.add_subplot(grid[1, 1])
        ax_processed.imshow(cv2.cvtColor(img_for_drawing, cv2.COLOR_BGR2RGB))
        ax_processed.set_title('Размеченное изображение')
        ax_processed.axis('off')

        # new
        equlid_r = {i: 0 for i in range(len(square_points))}

        # Гистограммы для каждого квадрата
        # for i, points_in_square in enumerate(square_points):
        #     row = (i // num_cols) + 2
        #     col = i % num_cols
        #     ax_hist = fig.add_subplot(grid[row, col])
        #
        #     # new
        #     img_hist, value = show_hist_square(points_in_square, i + 1, img)
        #     equlid_r[i] += value
        #
        #     ax_hist.imshow(img_hist)
        #     ax_hist.axis('off')

        # Если только подсчёт equlid_r нужен, можно оставить такой блок:
        for i, points_in_square in enumerate(square_points):
            _, value = RegeneratePresenter.show_hist_square(
                points_in_square, i + 1, img)
            equlid_r[i] += value

        plt.tight_layout()
        # plt.show()

        return equlid_r

    def search_regenerat_zone(equlid_r):
        potencial_regenerat_zone = []
        koef_equlid_r = 3.25  # сделать значение регулируемым
        average_equlid_r = sum(equlid_r.values()) / len(equlid_r)
        delta_equlid_r = max(equlid_r.values()) - min(equlid_r.values())
        beta_equlid_r = delta_equlid_r / koef_equlid_r

        regenerat_serment = []
        for key, value in equlid_r.items():
            if average_equlid_r - value > beta_equlid_r:
                regenerat_serment.append(key)

        return regenerat_serment

    def analyze_patology_regenert(img_array, model, num_classes):
        # # Загрузка и подготовка изображения
        # #img = load_img(image_path, target_size=(img_width, img_height), color_mode='grayscale')
        # img_array = img_array / 255.0  # Нормализация
        # img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности

        # # Предсказание
        # predictions = model.predict(img_array)
        # distances = np.linalg.norm(predictions - np.eye(num_classes), axis=1)  # Расстояние до каждого класса
        # verdict = np.argmax(predictions)  # Вердикт по классу

        # return distances, verdict

        img_array = cv2.resize(img_array, (128, 128))  # Измените на (128, 128)

        if img_array.shape[-1] == 3:  # Проверка на наличие 3 каналов
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Изменение формы изображения
        img_array = np.expand_dims(img_array, axis=-1)  # Добавление канала
        img_array = img_array / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности

        # Предсказание
        predictions = model.predict(img_array)
        # Расстояние до каждого класса
        distances = np.linalg.norm(predictions - np.eye(num_classes), axis=1)
        verdict = np.argmax(predictions)  # Вердикт по классу

        return distances, verdict

    def get_annotated_json(image_path: str, workDirectory: str, model_path: str = "static/neuralModels/golen.pt'"):
        """
        Генерирует файл JSON с разметкой, полученной из YOLO модели.

        :param image_path: Путь к изображению, для которого выполняется предсказание.
        :param workDirectory: Каталог, куда сохранить файл разметки.
        :param model_path: Путь к файлу модели YOLO (по умолчанию 'golen.pt').
        """
        # Загрузка модели
        model = YOLO(model_path)

        # Предсказание
        results = model.predict(image_path, save=False,
                                save_txt=False, save_conf=False)

        # Предполагаем, что интересует только первый результат
        result = results[0]
        print(result)

        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) <= 0:
            raise Exception(
                "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")
            return

        shapes = []

        for mask, cls_id in zip(result.masks.xy, result.boxes.cls):
            # Преобразуем точки маски к формату (x, y)
            points = [[float(x), float(y)] for x, y in mask]
            label = model.names[int(cls_id)]

            shapes.append({
                "label": label,
                "text": "",
                "points": points,
            })

        # Формируем структуру JSON
        json_data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": shapes
        }

        # Убедимся, что каталог существует
        os.makedirs(workDirectory, exist_ok=True)

        # Формируем имя файла на основе изображения
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(workDirectory, f"{base_name}.json")

        # Сохраняем JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON saved to {json_path}")
        return json_path

    def handleData(workDirectory, img_path, json_path):
        import cv2
        from PIL import Image
        # Ensure output directory exists
        output_dir = os.path.join(workDirectory, "output")
        plt.switch_backend('Agg')
        os.makedirs(output_dir, exist_ok=True)

        if json_path == '':
            json_path = RegeneratePresenter.get_annotated_json(
                img_path, workDirectory, 'static/neuralModels/golen.pt')

        avg_color_in_square_points = []
        img = cv2.imread(img_path)

        img_for_drawing = img.copy()  # копия, чтобы не портить исходный рисунок
        num_squares, num_points = 20, 5000  # номер квадратов и номер точек
        # массив точек внутри контура их средний цвет
        square_points, avg_color_in_square_points = [], []

        # получение контура из json разметки
        mask_contours = RegeneratePresenter.reverse_json_np(json_path)

        # Рисуем контур на черном фоне белым цветом
        black_background = np.zeros_like(img)
        cv2.drawContours(black_background, [
            mask_contours], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Используем маску для копирования только пикселей внутри контура
        img_for_drawing = cv2.bitwise_and(img.copy(), black_background)
        img_for_cut_regenerat = img_for_drawing.copy()

        # Получаем координаты квадратов и точки внутри квадрата
        square_list_data_coord, square_points = RegeneratePresenter.draw_points_in_squares_contours(
            img_for_drawing, mask_contours, num_squares, num_points)

        for _ in square_points:
            avg_color_in_square_points.append(
                RegeneratePresenter.calculate_average_color(_, img))

        equlid_r = RegeneratePresenter.create_regenerat_statistic(
            img, img_for_drawing, avg_color_in_square_points, square_points)
        regenerat_serment = RegeneratePresenter.search_regenerat_zone(equlid_r)

        x, y, w, h = cv2.boundingRect(mask_contours[0])
        cropped_img_for_drawing = img_for_drawing[y:y+h, x:x+w]

        # output_image = RegeneratePresenter.generate_img_by_square_nums(
        #     img_for_cut_regenerat, square_list_data_coord, regenerat_serment)
        output_path = os.path.join(output_dir, "output.png")
        print(output_path)
        # OpenCV uses BGR, but PIL expects RGB; convert if needed
        # if len(output_image.shape) == 3 and output_image.shape[2] == 3:
        save_img = cv2.cvtColor(cropped_img_for_drawing, cv2.COLOR_BGR2RGB)
        # else:
        #     save_img = output_image
        pil_img = Image.fromarray(save_img)
        pil_img.save(output_path)
        return output_path, img_path, equlid_r, regenerat_serment

    # model_for_analyze_regenerat = load_model('test_analyz_regenerat.h5')

    # cropped_potencial_regenerate
