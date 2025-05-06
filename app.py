from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
import dismodels.dismodels as dm
from main import (getIncline, getXByY, check, readImage, getBreakPoints, getPointsOfBone,
                  getBorderPositions, getPercentHeight, getCenterPointsOfContour,
                  getBreakPointsOfGroup, getMinPointsBreak, getMinMaxCenterContourPoints,
                  getLinePoints, getBoundsNew, main)
import os
import ast
import cv2
import json
import shutil
from kb import db, Object, Cultures
from Regenerat.RegenerateRouter import RegenerateRouter
from Regenerat.RegeneratePresenter import RegeneratePresenter
import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{ROOT_DIR}/admin/diseases.db'
db.init_app(app)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "berry"
MODELS_FOLDER = ('dismodels')
UPLOAD_FOLDER = ('images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'json'}
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# функция проверки расширения файла


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# объект мониторинга

@app.route("/object", methods=['GET', 'POST'])
def menu():
    return render_template("menu.html")

# вывод на страницу доступные модели поиска заболеваний


@app.route("/models")
def models():
    try:
        spisok1 = Cultures.query.all()
        spisok = []
        for each in spisok1:
            if each.modellink is not None:
                spisok.append(each)
        return render_template("models.html", title="Каталог моделей", spisok=spisok)
    except:
        return "Ошибка чтения из БД"

# страница для применения нейросетевой модели


@app.route("/disident/", methods=['GET', 'POST'])
def disident():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('Ничего не загружено')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)
        if allowed_file(file.filename) is False:
            flash('Файл не загружен, допускаются форматы: png, jpg, jpeg')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash('Файл успешно загружен')
            filename = secure_filename(file.filename)
            file.save(os.path.join(ROOT_DIR, UPLOAD_FOLDER, filename))
            try:
                img = f"{ROOT_DIR}/{UPLOAD_FOLDER}/{filename}"
                dis = list(set(dm.detection(img)))
                # dis = list(set([2]))
                if not dis:
                    flash(f'Болезней не обнаружено')
                else:
                    return redirect("/bones")
                    # dict = ast.literal_eval(cult.clsofmodel)
                    # n = ', '.join(list(map(lambda num: dict[num], dis)))
                    # flash(f'Обнаруженные болезни: {n}')
                return render_template('modelfound.html',
                                       title="Обнаружение болезни",
                                       filename=filename)
            except:
                flash('Ошибка при использовании модели')
    else:
        return render_template('modelfound.html',
                               title="Обнаружение болезни"
                               )

    def process_image(file_path):
        # Считываем изображение
        img, _, _ = readImage(file_path)

        # Получаем контуры костей
        contours = getBoundsNew(file_path)

        # Рисуем контуры на изображении
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Сохраняем изображение с контурами
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        cv2.imwrite(output_file, img)

        return output_file

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            # Получаем загруженный файл
            file = request.files['file']

            # Сохраняем загруженный файл
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Обрабатываем изображение
            output_file = process_image(file_path)

            # Возвращаем HTML с отображением обработанного изображения
            return render_template('result.html', image_file=output_file)

        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug=True)

    # Роут для загрузки изображения и вызова функции readImage

    @app.route('/upload_image', methods=['POST'])
    def upload_image():
        file = request.files['image']
        if file:
            image = file.read()
            img, height, width = readImage(image)
            # Далее вы можете сохранить img, height и width в сессии или передать в шаблон
            return render_template('upload_success.html', height=height, width=width)
        else:
            return "Ошибка загрузки изображения"

    # Роут для вызова функции getIncline
    @app.route('/get_incline', methods=['POST'])
    def get_incline():
        # Принимайте координаты точек из формы или JSON-запроса
        point = request.form['point']
        central_point = request.form['central_point']
        incline = getIncline(point, central_point)
        return incline


@app.route("/bones/", methods=['GET', 'POST'])
def bones():
    directory = "runs/segment/"
    files = os.listdir(directory)
    print(os.listdir(f"runs/segment/{files[-1]}"))
    pp = os.listdir(f"runs/segment/{files[-1]}")
    source = f"runs/segment/{files[-1]}/{pp[0]}"
    destination = f"static/bones/{pp[0]}"
    shutil.copyfile(source, destination)
    print(source)
    return render_template('bone_result.html', img=pp[0], title="Обнаружение болезни")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Получаем контуры объектов на загруженном изображении
            contours = getBoundsNew(file_path)
            # Здесь можно использовать контуры для дальнейшей обработки или анализа
            return render_template('result.html', contours=contours)
    return render_template('upload.html')


@app.route('/')
def about():
    return render_template('main.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # проверка логина и пароля
        return 'Вы вошли в систему!'
    else:
        return render_template('login.html')


@app.route('/funcNew/', methods=['GET', 'POST'])
def funcNew():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('Файл не загружен')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)
        if allowed_file(file.filename) is False:
            flash('Файл не загружен. Поддерживаемые форматы: png, jpg, jpeg')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash('Файл успешно загружен')
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                # Вызываем основную функцию для обработки изображения
                result = main(file_path)
            # Делаем что-то с результатом, например, передаем его в шаблон
                print(result)
                return render_template('results.html', result=result)
            except Exception as e:
                flash(f'Ошибка при обработке изображения: {e}')
                # Если произошла ошибка, перенаправляем обратно
                return redirect(request.url)
    else:
        return render_template('funcNew.html')

# Поиск регенерата


@app.route('/regenerat/golen', methods=['GET', 'POST'])
def fetchRegenerateGolen():
    if request.method == "POST":
        image_files = request.files.getlist('images')
        annotation_files = request.files.getlist('annotations')

        if not image_files or all(f.filename == '' for f in image_files):
            flash('Изображения не выбраны')
            return redirect(request.url)

        # Создаем временную директорию с меткой времени
        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp = 111
        # temp_dir = os.path.join(
        #     app.config['UPLOAD_FOLDER'], f'session_{timestamp}')
        temp_dir = os.path.join(
            'static', app.config['UPLOAD_FOLDER'], f'session_{timestamp}')

        os.makedirs(temp_dir, exist_ok=True)

        annotation_map = {}

        img_path = ""
        annot_path = ""

        # Очистка временной папки перед загрузкой новых файлов
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                flash(f'Ошибка при удалении файла {filename}: {e}')

        # Сохраняем JSON-файлы с аннотациями в временную папку (если они загружены)
        for annot in annotation_files:
            if annot and annot.filename != '':
                if annot.filename.endswith('.json'):
                    annot_name = secure_filename(annot.filename)
                    annot_path = os.path.join(temp_dir, annot_name)
                    annot.save(annot_path)
                    annotation_map[os.path.splitext(
                        annot_name)[0]] = annot_path
                else:
                    flash(f'Файл {annot.filename} не является JSON')

        results = []

        # Сохраняем изображения в временную папку
        for img in image_files:
            if img and allowed_file(img.filename):
                img_name = secure_filename(img.filename)
                img_path = os.path.join(temp_dir, img_name)
                img.save(img_path)
            else:
                flash(
                    f'Файл {img.filename} не является поддерживаемым изображением')

        try:
            model_name = "golen.pt"
            result = RegenerateRouter.run_regenerate_analysis(
                temp_dir, img_path, annot_path, model_name)
            results.append(result)
        except Exception as e:
            flash(f'Ошибка при обработке: {e}')

        if results:
            flash('Файлы успешно загружены и обработаны')
            return render_template('results.html', result=results)
        else:
            return redirect(request.url)
    else:
        return render_template('regeneratTemplate.html')


@app.route('/regenerat/bedro', methods=['GET', 'POST'])
def fetchRegenerateBedro():
    if request.method == "POST":
        image_files = request.files.getlist('images')
        annotation_files = request.files.getlist('annotations')

        if not image_files or all(f.filename == '' for f in image_files):
            flash('Изображения не выбраны')
            return redirect(request.url)

        # Создаем временную директорию с меткой времени
        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp = 111
        # temp_dir = os.path.join(
        #     app.config['UPLOAD_FOLDER'], f'session_{timestamp}')
        temp_dir = os.path.join(
            'static', app.config['UPLOAD_FOLDER'], f'session_{timestamp}')

        os.makedirs(temp_dir, exist_ok=True)

        annotation_map = {}

        img_path = ""
        annot_path = ""

        # Очистка временной папки перед загрузкой новых файлов
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                flash(f'Ошибка при удалении файла {filename}: {e}')

        # Сохраняем JSON-файлы с аннотациями в временную папку (если они загружены)
        for annot in annotation_files:
            if annot and annot.filename != '':
                if annot.filename.endswith('.json'):
                    annot_name = secure_filename(annot.filename)
                    annot_path = os.path.join(temp_dir, annot_name)
                    annot.save(annot_path)
                    annotation_map[os.path.splitext(
                        annot_name)[0]] = annot_path
                else:
                    flash(f'Файл {annot.filename} не является JSON')

        results = []

        # Сохраняем изображения в временную папку
        for img in image_files:
            if img and allowed_file(img.filename):
                img_name = secure_filename(img.filename)
                img_path = os.path.join(temp_dir, img_name)
                img.save(img_path)
            else:
                flash(
                    f'Файл {img.filename} не является поддерживаемым изображением')

        try:
            model_name = "bedro.pt"
            result = RegenerateRouter.run_regenerate_analysis(
                temp_dir, img_path, annot_path, model_name)
            results.append(result)
        except Exception as e:
            flash(f'Ошибка при обработке: {e}')

        if results:
            flash('Файлы успешно загружены и обработаны')
            return render_template('results.html', result=results)
        else:
            return redirect(request.url)
    else:
        return render_template('regeneratTemplate.html')


if __name__ == '__main__':
    app.run()

# чтобы рядом маска кости отрисовывалась выгружаем 2 картиники
