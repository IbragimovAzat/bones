import os
from ultralytics import YOLO
import onnxruntime as ort
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import json
# from tensorflow.keras.models import load_model
from database.models import db, BioAxis
from shapely.geometry import LineString
from scipy.stats import beta
from shapely.geometry import Polygon


class BiomecAxisPresenter:
    sustav_model = 'static/neuralModels/sustav.pt'
    golen_model = 'static/neuralModels/golen.pt'

    def get_angle_description(angle):
        if 177 <= angle <= 183:
            description = BioAxis.query.get(0)
        elif (174 <= angle <= 176) or (184 <= angle <= 186):
            description = BioAxis.query.get(1)
        elif (170 <= angle <= 173) or (187 <= angle <= 190):
            description = BioAxis.query.get(2)
        elif angle < 170 or angle > 190:
            description = BioAxis.query.get(3)
        else:
            description = BioAxis.query.get(0)
        return description

    def handle_data(work_dir, img_path, json_path_golen=''):
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
        center_coordinates = []
        if (len(mask_contours_sustav[0]) > 0):
            points_sustav = mask_contours_sustav[0]  # (N, 2)
            xc, yc, r = BiomecAxisPresenter.fit_circle_to_points(points_sustav)
            center_coordinates = (int(xc), int(yc))
            # cv2.circle(img, center_coordinates, radius=5,
            #            color=(0, 0, 255), thickness=-1)
            cv2.polylines(img, mask_contours_sustav, isClosed=True,
                          color=(0, 255, 0), thickness=20)

        # Отрисовка всех 3 точек
        saved_path, angle = BiomecAxisPresenter.plot_predicted_contour(
            img_path, json_path_golen, work_dir, center_coordinates)

        description = BiomecAxisPresenter.get_angle_description(angle)

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
        angle = int(angle)
        print(output_path)
        return output_path, saved_path, description, angle

    def find_key_points(coord_array):
        # Алгоритм: находим две области высокой плотности точек по вертикали (верхняя и нижняя части контура),
        # затем вычисляем их центры масс. Алгоритм устойчив к порядку точек, учитывается только положение точек.
        # Разделяем координаты X и Y из входного списка (первые 100 значений - X, следующие 100 - Y)
        x_coords = coord_array[:100]
        y_coords = coord_array[100:]
        # Составляем список пар (x, y) для удобства обработки
        points = list(zip(x_coords, y_coords))
        # Инициализируем центры двух кластеров по оси Y:
        # верхний (начально максимальный Y) и нижний (начально минимальный Y) уровни контура
        upper_center_y = max(y_coords)
        lower_center_y = min(y_coords)
        # Итеративно уточняем разделение на два кластера по Y (метод k-means по вертикальной оси)
        for _ in range(100):  # ограничиваем число итераций для надежности
            # Разбиваем точки на два кластера в зависимости от того, к какому центру по Y они ближе
            upper_cluster_points = [pt for pt in points
                                    if abs(pt[1] - upper_center_y) < abs(pt[1] - lower_center_y)]
            lower_cluster_points = [pt for pt in points
                                    if abs(pt[1] - upper_center_y) >= abs(pt[1] - lower_center_y)]
            # Вычисляем новые центры кластеров как среднее значение Y в каждой группе
            if upper_cluster_points:
                new_upper_center_y = sum(
                    p[1] for p in upper_cluster_points) / len(upper_cluster_points)
            else:
                new_upper_center_y = upper_center_y
            if lower_cluster_points:
                new_lower_center_y = sum(
                    p[1] for p in lower_cluster_points) / len(lower_cluster_points)
            else:
                new_lower_center_y = lower_center_y
            # Проверяем изменение центров: если сдвиг очень малый, завершаем итерации (кластеризация сошлась)
            if (abs(new_upper_center_y - upper_center_y) < 1e-6 and
                    abs(new_lower_center_y - lower_center_y) < 1e-6):
                upper_center_y, lower_center_y = new_upper_center_y, new_lower_center_y
                break
            # Обновляем центры кластеров для следующей итерации
            upper_center_y, lower_center_y = new_upper_center_y, new_lower_center_y
        # После разделения на две группы по Y вычисляем центры масс каждой группы (среднее X и Y)
        upper_cluster_points = [pt for pt in points
                                if abs(pt[1] - upper_center_y) < abs(pt[1] - lower_center_y)]
        lower_cluster_points = [
            pt for pt in points if pt not in upper_cluster_points]
        # Центр масс верхней группы (если группа не пуста)
        if upper_cluster_points:
            upper_centroid_x = sum(
                p[0] for p in upper_cluster_points) / len(upper_cluster_points)
            upper_centroid_y = sum(
                p[1] for p in upper_cluster_points) / len(upper_cluster_points)
        else:
            upper_centroid_x = None
            upper_centroid_y = None
        # Центр масс нижней группы (если группа не пуста)
        if lower_cluster_points:
            lower_centroid_x = sum(
                p[0] for p in lower_cluster_points) / len(lower_cluster_points)
            lower_centroid_y = sum(
                p[1] for p in lower_cluster_points) / len(lower_cluster_points)
        else:
            lower_centroid_x = None
            lower_centroid_y = None
        # Убеждаемся, что первая точка результата - верхняя (с более высоким Y), а вторая - нижняя
        if (upper_centroid_y is not None and lower_centroid_y is not None and
                upper_centroid_y < lower_centroid_y):
            # Если по каким-то причинам верхний центр оказался ниже по Y, поменяем их местами
            upper_centroid_x, lower_centroid_x = lower_centroid_x, upper_centroid_x
            upper_centroid_y, lower_centroid_y = lower_centroid_y, upper_centroid_y
        # Формируем итоговый результат в виде списка из двух точек (x1, y1) и (x2, y2)
        return [(upper_centroid_x, upper_centroid_y), (lower_centroid_x, lower_centroid_y)]

    def postprocess_bone_contour(points, epsilon):
        # Шаг 1: Упрощение контура методом Рамера–Дугласа–Пекера (RDP) с заданным tolerance ε.
        # Преобразуем массив точек в полигон (замкнутый контур) и упрощаем с помощью shapely.
        # создаём полигон по исходным точкам (shapely автоматически замыкает контур)
        poly = Polygon(points)
        # упрощаем контур (RDP, без сохранения топологии)
        simplified_poly = poly.simplify(epsilon, preserve_topology=False)
        # получаем координаты упрощённого внешнего контура
        simplified_coords = list(simplified_poly.exterior.coords)
        if simplified_coords[0] == simplified_coords[-1]:
            # удаляем повторяющуюся последнюю точку, совпадающую с первой
            simplified_coords = simplified_coords[:-1]

        # Шаг 2: Деление высоты кости на три зоны (верхняя 15%, средняя 70%, нижняя 15%).
        # Находим минимальную и максимальную координату Y исходного контура и определяем границы зон.
        original_points = np.array(points)
        min_y = original_points[:, 1].min()
        max_y = original_points[:, 1].max()
        height = max_y - min_y
        # граница начала верхней зоны (15% снизу от вершины)
        top_threshold = max_y - 0.15 * height
        # граница окончания нижней зоны (15% сверху от низа)
        bottom_threshold = min_y + 0.15 * height

        # Функция для определения зоны по координате Y
        def get_zone(y):
            if y >= top_threshold:
                return 'top'
            elif y <= bottom_threshold:
                return 'bottom'
            else:
                return 'mid'

        # Помечаем каждую точку упрощённого контура соответствующей зоной
        simplified_coords = np.array(simplified_coords)
        zones = [get_zone(y) for y in simplified_coords[:, 1]]

        # Шаг 3: Поиск "разрывов" в верхней и нижней частях.
        # Вычисляем среднее расстояние между соседними точками в верхней и нижней зонах.
        top_dists = []
        bottom_dists = []
        n = len(simplified_coords)
        for i in range(n):
            j = (i + 1) % n  # следующий индекс (с учётом замыкания контура)
            if zones[i] == zones[j] and zones[i] in ['top', 'bottom']:
                # Сегмент i-j полностью в верхней или нижней зоне
                dx = simplified_coords[j, 0] - simplified_coords[i, 0]
                dy = simplified_coords[j, 1] - simplified_coords[i, 1]
                dist = np.hypot(dx, dy)
                if zones[i] == 'top':
                    top_dists.append(dist)
                else:  # 'bottom'
                    bottom_dists.append(dist)
        avg_top = np.mean(top_dists) if top_dists else 0.0
        avg_bottom = np.mean(bottom_dists) if bottom_dists else 0.0
        # Порог для большого разрыва: в 2 раза больше среднего расстояния
        threshold_top = 2 * avg_top if avg_top > 0 else float('inf')
        threshold_bottom = 2 * avg_bottom if avg_bottom > 0 else float('inf')

        # Шаг 3 (продолжение): Вставка дополнительных точек на крупных разрывах верхней/нижней зон.
        new_points = []
        new_zones = []
        for i in range(n):
            j = (i + 1) % n
            # Добавляем текущую точку контура
            new_points.append(
                (simplified_coords[i, 0], simplified_coords[i, 1]))
            new_zones.append(zones[i])
            # Если сегмент (i->j) лежит в верхней или нижней зоне и сильно превышает среднюю длину
            if zones[i] == zones[j] and zones[i] in ['top', 'bottom']:
                dx = simplified_coords[j, 0] - simplified_coords[i, 0]
                dy = simplified_coords[j, 1] - simplified_coords[i, 1]
                dist = np.hypot(dx, dy)
                if (zones[i] == 'top' and dist > threshold_top) or (zones[i] == 'bottom' and dist > threshold_bottom):
                    # Вычисляем количество точек для равномерного заполнения разрыва
                    segment_threshold = threshold_top if zones[i] == 'top' else threshold_bottom
                    if segment_threshold == float('inf') or segment_threshold <= 0:
                        segment_threshold = dist
                    n_insert = int(np.ceil(dist / segment_threshold)) - 1
                    if n_insert < 1:
                        n_insert = 1
                    # Линейно интерполируем n_insert точек между i и j
                    for k in range(1, n_insert + 1):
                        frac = k / (n_insert + 1)
                        new_x = simplified_coords[i, 0] + frac * dx
                        new_y = simplified_coords[i, 1] + frac * dy
                        new_points.append((new_x, new_y))
                        new_zones.append(zones[i])

        # Шаг 4: Обеспечение плотности в верхней/нижней зонах и разреженности в средней зоне.
        new_points_arr = np.array(new_points)
        new_zones_list = list(new_zones)
        # Ограничиваем количество точек в средней зоне до 10–15, удаляя лишние 'mid'-точки при необходимости.
        mid_indices = [idx for idx, z in enumerate(
            new_zones_list) if z == 'mid']
        if len(mid_indices) > 15:
            protect_indices = set()
            total_pts = len(new_points_arr)
            # Определяем граничные точки средней зоны, которые нельзя удалять (сохраняют стыки зон)
            for idx in mid_indices:
                prev_idx = (idx - 1) % total_pts
                next_idx = (idx + 1) % total_pts
                if new_zones_list[prev_idx] != 'mid' or new_zones_list[next_idx] != 'mid':
                    protect_indices.add(idx)
            removable_mid = [
                i for i in mid_indices if i not in protect_indices]
            to_remove = len(mid_indices) - 15
            if to_remove > len(removable_mid):
                to_remove = len(removable_mid)
            if to_remove > 0:
                removable_mid.sort()
                step = len(removable_mid) / \
                    to_remove if to_remove > 0 else None
                remove_set = set()
                if step:
                    for r in range(to_remove):
                        remove_idx = removable_mid[int(r * step)]
                        remove_set.add(remove_idx)
                new_points_arr = np.array(
                    [pt for idx, pt in enumerate(new_points_arr) if idx not in remove_set])
                new_zones_list = [z for idx, z in enumerate(
                    new_zones_list) if idx not in remove_set]

        # Шаг 5: Доведение общего числа точек до ровно 100.
        # Добавляем недостающие точки в верхней/нижней зонах, чтобы общее количество стало 100.
        while len(new_points_arr) < 100:
            m = len(new_points_arr)
            longest_idx = None
            longest_dist = 0.0
            # Находим самый длинный сегмент (в верхней или нижней зоне)
            for i in range(m):
                j = (i + 1) % m
                if new_zones_list[i] == new_zones_list[j] and new_zones_list[i] in ['top', 'bottom']:
                    dx = new_points_arr[j, 0] - new_points_arr[i, 0]
                    dy = new_points_arr[j, 1] - new_points_arr[i, 1]
                    dist = np.hypot(dx, dy)
                    if dist > longest_dist:
                        longest_dist = dist
                        longest_idx = i
            if longest_idx is None:
                # нет сегментов в верхней/нижней зоне (маловероятно для нормального контура)
                break
            # Вставляем новую точку в середину самого длинного сегмента
            i = longest_idx
            j = (i + 1) % len(new_points_arr)
            mid_x = (new_points_arr[i, 0] + new_points_arr[j, 0]) / 2.0
            mid_y = (new_points_arr[i, 1] + new_points_arr[j, 1]) / 2.0
            mid_zone = new_zones_list[i]
            new_points_arr = np.insert(
                new_points_arr, j, (mid_x, mid_y), axis=0)
            new_zones_list.insert(j, mid_zone)
        # Если точек оказалось больше 100, удаляем лишние (сначала из средней зоны, иначе из зон с наибольшей плотностью).
        if len(new_points_arr) > 100:
            excess = len(new_points_arr) - 100
            mid_indices = [idx for idx, z in enumerate(
                new_zones_list) if z == 'mid']
            protect_idx = set()
            total_pts = len(new_points_arr)
            for idx in mid_indices:
                prev_idx = (idx - 1) % total_pts
                next_idx = (idx + 1) % total_pts
                if new_zones_list[prev_idx] != 'mid' or new_zones_list[next_idx] != 'mid':
                    protect_idx.add(idx)
            removable_mid = [i for i in mid_indices if i not in protect_idx]
            remove_set = set()
            if len(removable_mid) >= excess:
                removable_mid.sort()
                for r in range(excess):
                    remove_set.add(removable_mid[r])
            else:
                for idx in removable_mid:
                    remove_set.add(idx)
                still_excess = excess - len(removable_mid)
                total_pts = len(new_points_arr)
                for k in range(still_excess):
                    remove_set.add(total_pts - 1 - k)
            new_points_arr = np.array(
                [pt for idx, pt in enumerate(new_points_arr) if idx not in remove_set])
            new_zones_list = [z for idx, z in enumerate(
                new_zones_list) if idx not in remove_set]
        # Приводим итоговый набор к ровно 100 точкам (гарантируем размерность 100x2).
        final_points = np.array(new_points_arr)
        if len(final_points) > 100:
            final_points = final_points[:100]
        if len(final_points) < 100:
            needed = 100 - len(final_points)
            if needed > 0:
                padding = np.tile(final_points[-1], (needed, 1))
                final_points = np.vstack([final_points, padding])
        return final_points

    def plot_predicted_contour(img_path, json_path, output_dir, center_coordinates):
        """
        Строит предсказанный контур из JSON и сохраняет изображение с точками.

        :param img_path: Путь к исходному изображению
        :param json_path: Путь к JSON-файлу с аннотацией
        :param output_dir: Каталог, куда сохранить изображение
        :return: Путь к сохранённому изображению
        """
        # === Чтение изображения ===
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(img, cmap="gray")

        # === Чтение точек из JSON ===
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not data['shapes']:
            print("Нет фигур в JSON-файле.")
            return None

        points = np.array(data['shapes'][0]['points'])

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

        new_pts = BiomecAxisPresenter.postprocess_bone_contour(new_pts, 2.0)

        # Извлекаем x и y координаты
        xs = new_pts[:, 0]  # Первые 100 x-координат
        ys = new_pts[:, 1]  # Первые 100 y-координат

        points_result = np.concatenate([xs, ys])

        predictions = BiomecAxisPresenter.find_key_points(points_result)

        # points_input = np.expand_dims(points_result, axis=0)
        # model = load_model('static/neuralModels/golenDots.keras')
        # predictions = model.predict(points_input)[0]

        # print(predictions)
        golen_bottom = [predictions[0][0], predictions[0][1]]
        golen_top = [predictions[1][0], predictions[1][1]]

        if (center_coordinates):
            angle = BiomecAxisPresenter.compute_axis_angle(
                golen_bottom, golen_top, center_coordinates)

            BiomecAxisPresenter.plot_limb_axis(
                [golen_bottom, golen_top, center_coordinates], img_path)

        plt.scatter(xs, ys, color="red", marker="o",
                    label="Pred contour", s=1)

        # === Сохранение ===
        os.makedirs(output_dir, exist_ok=True)
        # filename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"predicated_contour.jpg")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        return output_path, angle

    def compute_axis_angle(p_bottom, p_knee, p_hip):
        """
        Вычисляет угол φ между векторами (колено→лодыжка) и (колено→тазобедренный сустав).
        Возвращает значение в градусах.
        """
        # векторы
        v1 = (p_bottom[0] - p_knee[0], p_bottom[1] - p_knee[1])
        v2 = (p_hip[0] - p_knee[0], p_hip[1] - p_knee[1])
        # скалярное произведение и нормы
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)
        # защита от числовых погрешностей
        cos_phi = max(min(dot/(norm1*norm2), 1.0), -1.0)
        phi = math.degrees(math.acos(cos_phi))
        return phi

    def plot_limb_axis(coords, image_path=None, annotate=True):
        """
        Строит ломаную P1→P2 (голень) и P2→P3 (бедро) НАД изображением background.
        coords = [p_bottom, p_knee, p_hip], каждое p = (x, y) в пикселях
        image_path = путь к файлу снимка (.png/.jpg/.tif и т.п.)
        """
        # 1) Загрузить и показать изображение подложкой
        if image_path:
            img = mpimg.imread(image_path)
            # origin='upper' — пиксель (0,0) будет в левом верхнем углу
            plt.imshow(img, origin='upper', cmap='gray')
            # сохраним границы для корректного масштабирования
            plt.xlim(0, img.shape[1])
            # перевёрнутая ось Y, чтобы (0,0) — вверху
            plt.ylim(img.shape[0], 0)

        # 2) Координаты точек
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        # 3) Нарисовать линии поверх
        plt.plot(xs[0:2], ys[0:2], '-o', linewidth=2, label='Голень')
        plt.plot(xs[1:3], ys[1:3], '-o', linewidth=2, label='Бедро')

        # 4) Аннотации точек
        # if annotate:
        #     labels = [
        #         "Нижняя середина голени",
        #         "Верхняя середина голени (колено)",
        #         "Центр тазобедренного сустава"
        #     ]
        #     for (x, y), lab in zip(coords, labels):
        #         plt.annotate(lab, (x, y),
        #                      textcoords="offset points",
        #                      xytext=(5, 5),
        #                      color='green',
        #                      fontsize=2,
        #                      weight='bold')

        plt.legend(loc='upper right')
        plt.axis('off')
        # plt.title("Ось нижней конечности поверх снимка")
        plt.tight_layout()

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
                # "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")
                "Не удалось сформировать разметку для загруженного изображения. Убедитесь в полноте снимка (наличия на изображении большеберцовой кости и тазобедренного сустава)")

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
                # "Не удалось сформировать JSON разметку для изображения. Попробуйте загрузить вручную")
                "Не удалось сформировать разметку для загруженного изображения. Убедитесь в полноте снимка (наличия на изображении большеберцовой кости и тазобедренного сустава)")

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
