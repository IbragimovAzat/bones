import os
from ultralytics import YOLO
import onnxruntime as ort
import math
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import json
from tensorflow.keras.models import load_model
from database.models import db, User
from shapely.geometry import LineString
from scipy.stats import beta


class BiomecAxisPresenter:
    sustav_model = 'static/neuralModels/sustav.pt'
    golen_model = 'static/neuralModels/golen.pt'

    def handle_data(work_dir, img_path, json_path_golen):
        plt.switch_backend('Agg')

        if json_path_golen == '':
            json_path_golen = BiomecAxisPresenter.get_annotations_golen(
                img_path, work_dir)

        json_path_sustav = BiomecAxisPresenter.get_annotations_sustav(
            img_path, work_dir)
        # картинки
        img = cv2.imread(img_path)
        kernel_image = img.copy()

        # контуры
        mask_contours_golen = BiomecAxisPresenter.reverse_json_np(
            json_path_golen)
        mask_contours_sustav = BiomecAxisPresenter.reverse_json_np(
            json_path_sustav)

        # центр сустава
        if (len(mask_contours_sustav[0]) > 0):
            points_sustav = mask_contours_sustav[0]  # (N, 2)
            xc, yc, r = BiomecAxisPresenter.fit_circle_to_points(points_sustav)
            center_coordinates = (int(xc), int(yc))
            cv2.circle(img, center_coordinates, radius=5,
                       color=(0, 0, 255), thickness=-1)
            cv2.polylines(img, mask_contours_sustav, isClosed=True,
                          color=(0, 255, 0), thickness=20)

        saved_path = BiomecAxisPresenter.plot_predicted_contour(
            img_path, json_path_golen, work_dir)

        # (Опционально) — отрисуем аппроксимированную окружность:
        # cv2.circle(img, center_coordinates, radius=int(
        #     r), color=(255, 0, 0), thickness=2)

        cv2.polylines(img, mask_contours_golen, isClosed=True,
                      color=(0, 255, 0), thickness=2)

        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(
            output_dir, f"{filename_no_ext}_with_contour.jpg")
        cv2.imwrite(output_path, img)
        print(output_path)
        return output_path, saved_path

    def plot_predicted_contour(img_path, json_path, output_dir):
        """
        Строит предсказанный контур из JSON и сохраняет изображение с точками.

        :param img_path: Путь к исходному изображению
        :param json_path: Путь к JSON-файлу с аннотацией
        :param output_dir: Каталог, куда сохранить изображение
        :return: Путь к сохранённому изображению
        """
        # === Чтение изображения ===
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")

        # === Чтение точек из JSON ===
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not data['shapes']:
            print("Нет фигур в JSON-файле.")
            return None

        points = np.array(data['shapes'][0]['points'])

        # if current_len > MAX_POINTS:
        #     # Случайным образом выбираем 100 точек из имеющихся
        #     indices = np.random.choice(current_len, MAX_POINTS, replace=False)
        #     points = points[indices]
        # elif current_len < MAX_POINTS:
        #     # Добавляем нули в конец
        #     pad_len = MAX_POINTS - current_len
        #     padding = np.zeros((pad_len, 2))
        #     points = np.concatenate([points, padding], axis=0)

        # MAX_POINTS = 100
        # points = np.array(points)  # замените на свой массив
        # current_len = len(points)

        # if current_len > MAX_POINTS:
        #     # Вычисляем шаг (каждую k-ю точку оставить)
        #     step = current_len // MAX_POINTS
        #     # Срезаем с шагом, чтобы получить равномерно распределённые точки
        #     reduced = points[::step]

        #     # Если всё ещё больше 100 из-за округления, обрезаем
        #     if len(reduced) > MAX_POINTS:
        #         reduced = reduced[:MAX_POINTS]
        #     # Если меньше 100 — добираем из оставшихся
        #     elif len(reduced) < MAX_POINTS:
        #         remaining = MAX_POINTS - len(reduced)
        #         extras = points[1::step][:remaining]
        #         reduced = np.vstack([reduced, extras])

        #     points = reduced

        # elif current_len < MAX_POINTS:
        #     # Добавляем нули до нужного размера
        #     pad_len = MAX_POINTS - current_len
        #     padding = np.zeros((pad_len, 2))
        #     points = np.concatenate([points, padding], axis=0)

        data = [
            285.5932203389831, 324.5762711864407, 347.4576271186441, 374.5762711864407,
            394.91525423728814, 418.6440677966102, 448.3050847457627, 472.8813559322034,
            496.6101694915254, 517.7966101694916, 544.9152542372882, 578.8135593220339,
            600.8474576271187, 615.2542372881356, 622.8813559322034, 624.5762711864407,
            619.4915254237288, 608.4745762711865, 577.1186440677966, 541.5254237288136,
            522.0338983050848, 503.38983050847463, 489.8305084745763, 476.271186440678,
            461.0169491525424, 436.4406779661017, 417.79661016949154, 402.54237288135596,
            379.66101694915255, 363.5593220338983, 338.98305084745766, 327.96610169491527,
            331.35593220338984, 338.135593220339, 358.47457627118644, 368.6440677966102,
            364.40677966101697, 358.47457627118644, 350.8474576271187, 333.0508474576271,
            315.2542372881356, 303.3898305084746, 280.5084745762712, 254.23728813559325,
            222.03389830508476, 211.0169491525424, 194.0677966101695, 177.11864406779662,
            163.55932203389833, 151.6949152542373, 133.05084745762713, 117.79661016949153,
            106.77966101694916, 110.16949152542374, 120.33898305084746, 133.05084745762713,
            152.54237288135593, 175.42372881355934, 191.52542372881356, 205.93220338983053,
            248.3050847457627, 274.5762711864407, 310.1694915254237, 321.1864406779661,
            337.2881355932204, 343.22033898305085, 337.2881355932204, 333.8983050847458,
            316.9491525423729, 296.6101694915254, 280.5084745762712, 270.33898305084745,
            262.7118644067797, 263.5593220338983, 274.5762711864407, 284.7457627118644,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            321.03389830508473, 322.728813559322, 324.4237288135593, 319.33898305084745,
            308.32203389830505, 307.47457627118644, 311.7118644067796, 311.7118644067796,
            317.64406779661016, 329.50847457627117, 338.8305084745763, 342.22033898305085,
            343.06779661016947, 351.5423728813559, 362.5593220338983, 399.8474576271186,
            428.66101694915255, 442.22033898305085, 477.8135593220339, 515.9491525423729,
            552.3898305084746, 602.3898305084746, 645.6101694915254, 705.7796610169491,
            781.2033898305085, 899.0, 1037.9830508474574, 1145.6101694915253,
            1435.4406779661017, 1587.9830508474577, 1856.6271186440677, 1974.4237288135594,
            2032.050847457627, 2061.71186440678, 2104.9322033898306, 2139.677966101695,
            2160.8644067796613, 2168.491525423729, 2173.576271186441, 2171.033898305085,
            2161.71186440678, 2160.0169491525426, 2157.474576271187, 2155.7796610169494,
            2150.6949152542375, 2147.305084745763, 2146.4576271186443, 2151.542372881356,
            2163.406779661017, 2176.9661016949153, 2181.2033898305085, 2178.661016949153,
            2167.64406779661, 2150.6949152542375, 2122.728813559322, 2097.305084745763,
            2068.491525423729, 2029.508474576271, 1994.7627118644068, 1944.7627118644068,
            1708.322033898305, 1471.0338983050847, 1070.186440677966, 952.3898305084746,
            732.8983050847457, 640.5254237288136, 569.3389830508474, 529.5084745762712,
            488.8305084745763, 456.62711864406776, 417.64406779661016, 377.8135593220339,
            353.2372881355932, 326.96610169491527, 322.728813559322, 320.1864406779661,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        # import numpy as np

        # Данные из JSON (points)
        points_data = [
            [301.32501220703125, 314.45001220703125],
            [297.1875, 318.5874938964844],
            [272.3625183105469, 318.5874938964844],
            [268.2250061035156, 322.7250061035156],
            [264.0874938964844, 322.7250061035156],
            [259.9499816894531, 326.86248779296875],
            [259.9499816894531, 331.0],
            [255.8125, 335.13751220703125],
            [255.8125, 359.9624938964844],
            [259.9499816894531, 364.1000061035156],
            [259.9499816894531, 380.6499938964844],
            [264.0874938964844, 384.7875061035156],
            [264.0874938964844, 388.92498779296875],
            [272.3625183105469, 397.20001220703125],
            [272.3625183105469, 405.4750061035156],
            [276.5, 409.61248779296875],
            [276.5, 422.0249938964844],
            [280.63751220703125, 426.1625061035156],
            [280.63751220703125, 430.29998779296875],
            [288.9125061035156, 438.57501220703125],
            [288.9125061035156, 442.7124938964844],
            [293.0500183105469, 446.8500061035156],
            [293.0500183105469, 450.98748779296875],
            [297.1875, 455.125],
            [297.1875, 459.26251220703125],
            [309.6000061035156, 471.67498779296875],
            [309.6000061035156, 484.0874938964844],
            [313.7375183105469, 488.2250061035156],
            [313.7375183105469, 492.36248779296875],
            [322.01251220703125, 500.63751220703125],
            [322.01251220703125, 508.9125061035156],
            [326.1500244140625, 513.0499877929688],
            [326.1500244140625, 554.4249877929688],
            [330.2875061035156, 558.5625],
            [330.2875061035156, 566.8375244140625],
            [334.4250183105469, 570.9749755859375],
            [334.4250183105469, 575.1124877929688],
            [338.5625, 579.25],
            [338.5625, 583.3875122070312],
            [342.70001220703125, 587.5250244140625],
            [342.70001220703125, 703.375],
            [338.5625, 707.5125122070312],
            [338.5625, 724.0625],
            [334.4250183105469, 728.2000122070312],
            [334.4250183105469, 740.6124877929688],
            [330.2875061035156, 744.75],
            [330.2875061035156, 773.7125244140625],
            [326.1500244140625, 777.8499755859375],
            [326.1500244140625, 922.6624755859375],
            [322.01251220703125, 926.7999877929688],
            [322.01251220703125, 939.2125244140625],
            [317.875, 943.3499755859375],
            [317.875, 947.4874877929688],
            [313.7375183105469, 951.625],
            [313.7375183105469, 968.1749877929688],
            [309.6000061035156, 972.3125],
            [309.6000061035156, 1092.300048828125],
            [305.4625244140625, 1096.4375],
            [305.4625244140625, 1150.2249755859375],
            [301.32501220703125, 1154.362548828125],
            [301.32501220703125, 1170.9124755859375],
            [297.1875, 1175.050048828125],
            [297.1875, 1208.1500244140625],
            [293.0500183105469, 1212.2874755859375],
            [293.0500183105469, 1295.0374755859375],
            [288.9125061035156, 1299.175048828125],
            [288.9125061035156, 1307.449951171875],
            [284.7750244140625, 1311.5875244140625],
            [284.7750244140625, 1315.7249755859375],
            [280.63751220703125, 1319.862548828125],
            [280.63751220703125, 1344.6875],
            [276.5, 1348.824951171875],
            [276.5, 1468.8125],
            [272.3625183105469, 1472.949951171875],
            [272.3625183105469, 1485.362548828125],
            [268.2250061035156, 1489.5],
            [268.2250061035156, 1497.7750244140625],
            [264.0874938964844, 1501.9124755859375],
            [264.0874938964844, 1522.5999755859375],
            [259.9499816894531, 1526.737548828125],
            [259.9499816894531, 1617.762451171875],
            [255.8125, 1621.9000244140625],
            [255.8125, 1630.175048828125],
            [251.67498779296875, 1634.3125],
            [251.67498779296875, 1638.449951171875],
            [247.53749084472656, 1642.5875244140625],
            [247.53749084472656, 1659.137451171875],
            [243.39999389648438, 1663.2750244140625],
            [243.39999389648438, 1746.0250244140625],
            [239.2624969482422, 1750.1624755859375],
            [239.2624969482422, 1758.4375],
            [235.125, 1762.574951171875],
            [235.125, 1766.7125244140625],
            [230.98748779296875, 1770.8499755859375],
            [230.98748779296875, 1779.125],
            [226.84999084472656, 1783.262451171875],
            [226.84999084472656, 1832.9124755859375],
            [222.71249389648438, 1837.050048828125],
            [222.71249389648438, 1841.1875],
            [218.5749969482422, 1845.324951171875],
            [218.5749969482422, 1853.5999755859375],
            [214.4375, 1857.737548828125],
            [214.4375, 1866.012451171875],
            [210.29998779296875, 1870.1500244140625],
            [210.29998779296875, 1919.800048828125],
            [206.16249084472656, 1923.9375],
            [206.16249084472656, 1932.2125244140625],
            [197.8874969482422, 1940.487548828125],
            [197.8874969482422, 1948.762451171875],
            [193.75, 1952.9000244140625],
            [193.75, 1977.7249755859375],
            [181.33749389648438, 1990.137451171875],
            [181.33749389648438, 1998.4124755859375],
            [177.1999969482422, 2002.550048828125],
            [177.1999969482422, 2023.237548828125],
            [160.64999389648438, 2039.7874755859375],
            [160.64999389648438, 2043.925048828125],
            [148.23748779296875, 2056.33740234375],
            [148.23748779296875, 2060.47509765625],
            [144.09999084472656, 2064.612548828125],
            [144.09999084472656, 2085.300048828125],
            [139.96249389648438, 2089.4375],
            [139.96249389648438, 2093.574951171875],
            [127.55000305175781, 2105.987548828125],
            [127.55000305175781, 2172.1875],
            [131.6875, 2176.324951171875],
            [131.6875, 2188.737548828125],
            [135.8249969482422, 2188.737548828125],
            [156.5124969482422, 2168.050048828125],
            [160.64999389648438, 2168.050048828125],
            [164.78749084472656, 2163.91259765625],
            [168.92498779296875, 2163.91259765625],
            [181.33749389648438, 2151.5],
            [185.47499084472656, 2151.5],
            [189.61248779296875, 2147.362548828125],
            [214.4375, 2147.362548828125],
            [218.5749969482422, 2151.5],
            [226.84999084472656, 2151.5],
            [230.98748779296875, 2155.637451171875],
            [235.125, 2155.637451171875],
            [239.2624969482422, 2159.77490234375],
            [251.67498779296875, 2159.77490234375],
            [255.8125, 2163.91259765625],
            [268.2250061035156, 2163.91259765625],
            [280.63751220703125, 2176.324951171875],
            [297.1875, 2176.324951171875],
            [305.4625244140625, 2168.050048828125],
            [305.4625244140625, 2155.637451171875],
            [309.6000061035156, 2151.5],
            [309.6000061035156, 2118.39990234375]
        ]

        # Преобразуем в массив NumPy
        points = np.array(points)

        line = LineString(points)
        simp_line = line.simplify(tolerance=2.0)
        simp_pts = np.array(simp_line.coords)

        # Ресемплинг по длине с U-образным распределением
        dists = np.sqrt(((points[1:]-points[:-1])**2).sum(axis=1))
        cum = np.concatenate(([0], np.cumsum(dists)))
        cum /= cum[-1]
        t_vals = beta.ppf(np.linspace(0, 1, 100), 0.5, 0.5)
        new_x = np.interp(t_vals, cum, points[:, 0])
        new_y = np.interp(t_vals, cum, points[:, 1])
        new_pts = np.vstack([new_x, new_y]).T

        # Проверяем количество точек
        # print(f"Количество точек: {len(new_pts)}")  # 141 точка

        # Берем первые 100 точек
        # points = points[:100]

        # Извлекаем x и y координаты
        xs = new_pts[:, 0]  # Первые 100 x-координат
        ys = new_pts[:, 1]  # Первые 100 y-координат

        # удалить
        # if len(xs) > MAX_POINTS*2:
        #     # Случайным образом выбираем 100 точек из имеющихся
        #     indices = np.random.choice(current_len, MAX_POINTS, replace=False)
        #     xs = points[indices]
        #     ys =
        # elif current_len < MAX_POINTS*2:
        #     # Добавляем нули в конец
        #     pad_len = MAX_POINTS - current_len
        #     padding = np.zeros((pad_len, 2))
        #     points = np.concatenate([points, padding], axis=0)

        # Преобразуем в массив NumPy
        # data = np.array(data)

        # # Формируем массивы xs (первые 100 точек) и ys (следующие 100 точек)
        # xs = data[:100]
        # ys = data[100:200]

        # xs, ys = points[:, 0], points[:, 1]

        points_result = np.concatenate([xs, ys])
        # print(len(points_result))
        # points_flat = points_result.flatten()  # (100, 2) -> (200,)

        points_input = np.expand_dims(points_result, axis=0)

        # points_flat = points_result.flatten()  # (100, 2) -> (200,)
        # points_input = np.expand_dims(points_flat, axis=0)

        model = load_model('static/neuralModels/golenDots.keras')
        predictions = model.predict(points_input)[0]

        # predictions[0] *= w  # x1
        # predictions[2] *= w  # x2
        # predictions[1] *= h  # y1
        # predictions[3] *= h
        print(predictions)
        plt.scatter(predictions[0], predictions[1],
                    color="red", marker="x", label="Pred Point 1", s=50)
        plt.scatter(predictions[2], predictions[3],
                    color="red", marker="x", label="Pred Point 1", s=50)

        plt.scatter(xs, ys, color="red", marker="o",
                    label="Pred contour", s=1)
        plt.axis("off")

        # === Сохранение ===
        os.makedirs(output_dir, exist_ok=True)
        # filename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"predicated_contour.jpg")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return output_path

    def fit_circle_to_points(points):
        """
        Аппроксимирует окружность по набору точек и возвращает центр и радиус.
        :param points: np.ndarray формы (N, 2), где каждая строка — [x, y]
        :return: (xc, yc, r) — координаты центра окружности и радиус
        """
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        A = np.c_[2*x, 2*y, np.ones_like(x)]
        b = x**2 + y**2

        # Решаем линейную систему Ax = b
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc = c[0], c[1]
        r = np.sqrt(c[2] + xc**2 + yc**2)

        return xc, yc, r

    def reverse_json_np(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        points = np.array(data['shapes'][0]['points'], dtype=np.int32)
        return np.array([points])

    def get_annotations_golen(image_path: str, workDirectory: str):
        """
        Генерирует файл JSON с разметкой, полученной из YOLO модели.
        :param image_path: Путь к изображению, для которого выполняется предсказание.
        :param workDirectory: Каталог, куда сохранить файл разметки.
        :param model_path: Путь к файлу модели YOLO (по умолчанию 'golen.pt').
        """
        # Загрузка модели
        model = YOLO(BiomecAxisPresenter.golen_model)

        # Предсказание
        results = model.predict(image_path, save=False,
                                save_txt=False, save_conf=False)
        result = results[0]

        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) <= 0:
            raise Exception(
                "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")

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
        base_name = 'golen'
        json_path = os.path.join(workDirectory, f"{base_name}.json")

        # Сохраняем JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return json_path

    def get_annotations_sustav(image_path: str, workDirectory: str):
        """
        Генерирует файл JSON с разметкой, полученной из YOLO модели.
        :param image_path: Путь к изображению, для которого выполняется предсказание.
        :param workDirectory: Каталог, куда сохранить файл разметки.
        """
        # Загрузка модели
        model = YOLO(BiomecAxisPresenter.sustav_model)

        # Предсказание
        results = model.predict(image_path, save=False,
                                save_txt=False, save_conf=False)
        result = results[0]

        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) <= 0:
            raise Exception(
                "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")

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
        base_name = "sustav"
        json_path = os.path.join(workDirectory, f"{base_name}.json")

        # Сохраняем JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return json_path
