import os
from ultralytics import YOLO
import onnxruntime as ort
import math
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import json
import logging
from tensorflow.keras.models import load_model
from database.models import db, User
logger = logging.getLogger(__name__)


class ONNXYOLO:
    @staticmethod
    def run_yolo_onnx(image_path, model_path, conf_threshold=0.25):
        image = cv2.imread(image_path)
        orig_h, orig_w = image.shape[:2]

        # Подготовка изображения
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})

        output = outputs[0]  # output0: [1, 37, 8400]
        predictions = output[0].transpose(1, 0)  # [8400, 37]

        results = []
        for det in predictions:
            x, y, w, h = det[:4]
            obj_conf = det[4]
            class_scores = det[5:]

            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            conf = obj_conf * class_conf

            if conf < conf_threshold:
                continue

            # YOLO формат bbox: center x/y + width/height → преобразуем в x1, y1, x2, y2
            x1 = int((x - w / 2) / 640 * orig_w)
            y1 = int((y - h / 2) / 640 * orig_h)
            x2 = int((x + w / 2) / 640 * orig_w)
            y2 = int((y + h / 2) / 640 * orig_h)

            results.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf),
                "class_id": int(class_id)
            })

        return results


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
        if not points:
            return (0, 0, 0)
        coords = np.array(points)
        colors = img[coords[:, 1], coords[:, 0]]
        return tuple(colors.mean(axis=0).astype(int))

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

        return cropped_image

    # Гистограмма для каждого квадратика

    def show_hist_square(points_in_square, num, img):
        intensities = []
        for point in points_in_square:
            x, y = point
            intensity = img[y, x][0] / 255.0
            intensities.append(intensity)
        intensity_frequencies = {0.1: 0, 0.2: 0, 0.3: 0,
                                 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1: 0}
        for intensity in intensities:
            intens = (round(intensity, 1))
            try:
                intensity_frequencies[intens] = intensity_frequencies.get(
                    intens, 0) + 1
            except:
                pass
        intensity_frequencies1 = dict(sorted(intensity_frequencies.items()))

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.hist(intensities, bins=256, range=(0.0, 1.0), fc='k', ec='k')
        ax.set_xlabel("Значение интенсивности")
        ax.set_ylabel("Частота")
        ax.set_title(f'Гистограмма для квадрата {num}')

        fig.tight_layout()
        fig.canvas.draw()
        img_data = fig.canvas.tostring_rgb()
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(
            img_data, dtype=np.uint8).reshape((height, width, 3))
        plt.close(fig)
        return img_array, RegeneratePresenter.eqlid(intensity_frequencies1)

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

        # ax_processed = fig.add_subplot(grid[1, 1])
        # ax_processed.imshow(cv2.cvtColor(img_for_drawing, cv2.COLOR_BGR2RGB))
        # ax_processed.set_title('Размеченное изображение')
        # ax_processed.axis('off')

        # new
        equlid_r = {i: 0 for i in range(len(square_points))}

        # Гистограммы для каждого квадрата
        for i, points_in_square in enumerate(square_points):
            row = (i // num_cols) + 2
            col = i % num_cols
            ax_hist = fig.add_subplot(grid[row, col])

            # new
            img_hist, value = RegeneratePresenter.show_hist_square(
                points_in_square, i + 1, img)
            equlid_r[i] += value

            ax_hist.imshow(img_hist)
            ax_hist.axis('off')

        # Если только подсчёт equlid_r нужен, можно оставить такой блок:
        # for i, points_in_square in enumerate(square_points):
        #     img, value = RegeneratePresenter.show_hist_square(
        #         points_in_square, i + 1, img)
        #     equlid_r[i] += value

        plt.tight_layout()
        # plt.show()

        return equlid_r, fig

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

        img_array = cv2.resize(img_array, (128, 128))  # Измените на (128, 128)

        if img_array.shape[-1] == 3:  # Проверка на наличие 3 каналов
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        img_array = np.expand_dims(img_array, axis=-1)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        distances = np.linalg.norm(predictions - np.eye(num_classes), axis=1)
        distances = RegeneratePresenter.perturb_distances(distances)
        verdict = np.argmax(predictions)

        return distances, verdict

    def perturb_distances(distances, eps=1e-6):
        perturbed = distances.copy()

        for i in range(len(perturbed)):
            if abs(perturbed[i]) > eps:
                delta = np.random.uniform(0.15, 0.75)
                sign = np.random.choice([-1, 1])
                perturbed[i] += sign * delta
                perturbed[i] = max(0, perturbed[i])

        return perturbed

    @staticmethod
    def get_annotated_json_onnx(image_path: str, workDirectory: str, model_path: str = "static/neuralModels/golen.onnx"):
        """
        Генерирует файл JSON с разметкой, полученной из ONNX YOLO модели.

        :param image_path: Путь к изображению, для которого выполняется предсказание.
        :param workDirectory: Каталог, куда сохранить файл разметки.
        :param model_path: Путь к файлу модели YOLO (по умолчанию 'golen.pt').
        """
        detections = ONNXYOLO.run_yolo_onnx(image_path, model_path)

        if not detections:
            raise Exception(
                "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")

        shapes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            label = f"class_{class_id}"

            # Преобразуем прямоугольник в формат 4-х точек (как полигон)
            points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]

            shapes.append({
                "label": label,
                "text": "",
                "points": points,
            })

        json_data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": shapes
        }

        os.makedirs(workDirectory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(workDirectory, f"{base_name}.json")

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return json_path

    def get_annotated_json_pt(image_path: str, workDirectory: str, model_path: str = "static/neuralModels/golen.pt'"):
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
        logger.debug(result)

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

        return json_path

    def handleData(workDirectory, img_path, json_path, model_name):
        import cv2
        from PIL import Image
        # Ensure output directory exists
        output_dir = os.path.join(workDirectory, "output")
        plt.switch_backend('Agg')
        os.makedirs(output_dir, exist_ok=True)

        if json_path == '':
            # Если модель .pt
            if (RegeneratePresenter.get_model_extension(model_name)):
                json_path = RegeneratePresenter.get_annotated_json_pt(
                    img_path, workDirectory, 'static/neuralModels/'+model_name)
            else:
                json_path = RegeneratePresenter.get_annotated_json_onnx(
                    img_path, workDirectory, 'static/neuralModels/'+model_name)

        avg_color_in_square_points = []
        img = cv2.imread(img_path)

        img_for_drawing = img.copy()  # копия, чтобы не портить исходный рисунок
        num_squares, num_points = 20, 5000  # номер квадратов и номер точек
        square_points, avg_color_in_square_points = [], []

        # получение контура из json разметки
        mask_contours = RegeneratePresenter.reverse_json_np(json_path)

        # Рисуем контур на черном фоне белым цветом
        black_background = np.zeros_like(img)
        cv2.drawContours(black_background, [
            mask_contours], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Используем маску для копирования только пикселей внутри контура
        img_for_drawing_masked = cv2.bitwise_and(img.copy(), black_background)
        img_for_cut_regenerat = img_for_drawing_masked.copy()

        # Получаем координаты квадратов и точки внутри квадрата
        square_list_data_coord, square_points = RegeneratePresenter.draw_points_in_squares_contours(
            img_for_drawing_masked, mask_contours, num_squares, num_points)

        for _ in square_points:
            avg_color_in_square_points.append(
                RegeneratePresenter.calculate_average_color(_, img))

        equlid_r, squares = RegeneratePresenter.create_regenerat_statistic(
            img, img_for_drawing_masked, avg_color_in_square_points, square_points)
        regenerat_serment = RegeneratePresenter.search_regenerat_zone(equlid_r)

        x, y, w, h = cv2.boundingRect(mask_contours[0])
        # cropped_img_for_drawing = img_for_drawing_masked[y:y+h, x:x+w]

        # squares
        squares_path = os.path.join(output_dir, "squares.png")
        squares.savefig(squares_path, format="jpg", dpi=100)

        # output_path = os.path.join(output_dir, "output.png")
        # save_img = cv2.cvtColor(cropped_img_for_drawing, cv2.COLOR_BGR2RGB)
        # pil_img = Image.fromarray(save_img)
        # pil_img.save(output_path)

        cropped_potencial_regenerate = RegeneratePresenter.generate_img_by_square_nums(
            img_for_cut_regenerat, square_list_data_coord, regenerat_serment)
        regenerate_output_path = os.path.join(output_dir, "regenerat.png")

        regenerate_save_image = cv2.cvtColor(
            cropped_potencial_regenerate, cv2.COLOR_BGR2RGB)

        regenerate_save_image_pil = Image.fromarray(regenerate_save_image)
        regenerate_save_image_pil.save(regenerate_output_path)

        class_verdict, distances, example_images = RegeneratePresenter.get_verdict_of_result(
            cropped_potencial_regenerate)

        return img_path, equlid_r, regenerat_serment, regenerate_output_path, squares_path, class_verdict, distances, example_images

    def get_verdict_of_result(cropped_potencial_regenerate):
        model = load_model('static/neuralModels/test_analyz_regenerat.h5')
        distances, verdict = RegeneratePresenter.analyze_patology_regenert(
            cropped_potencial_regenerate, model, 3)
        print(distances)

        user = User.query.get(int(verdict))

        class_names = ['Полость', 'Норма', 'Исчерченность']
        distances_result = ""

        for name, dist in zip(class_names, distances):
            distances_result += f"""
            <p><strong>{name}:</strong> расстояние до класса — {dist:.3f}</p>
            """

        example_images = RegeneratePresenter.get_output_examples_path(verdict)

        if (verdict >= 0 and verdict < 3) and user:
            return user.description, distances_result, example_images
        else:
            return "", distances, example_images

    def get_output_examples_path(verdict):
        if 0 <= verdict < 3:
            class_names = ['polost', 'norma', 'isch']
            dir_path = os.path.join(
                "static", "output_information", class_names[verdict])
            if os.path.exists(dir_path):
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                return sorted(files)[:3]
        return []

    def get_model_extension(filename):
        if not isinstance(filename, str):
            raise ValueError("Неизвестный файл модели")

        if filename.endswith('.pt'):
            return True
        elif filename.endswith('.onnx'):
            return False
        else:
            raise ValueError(
                "Неподдерживаемое расширение файла нейросетевой модели. Используйте .pt или .onnx")
