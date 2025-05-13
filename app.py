import random
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from main import getBoundsNew
import os
import shutil
from Regenerat.RegenerateRouter import RegenerateRouter
from Regenerat.RegeneratePresenter import RegeneratePresenter
from kernelLine.KernelLinePresenter import KernelLinePresenter
from BiomecAxis.BiomecAxisPresenter import BiomecAxisPresenter

from database.config import Config, init_db, update_db
from database.models import db, User
# from database import config

app = Flask(__name__)

app.config.from_object(Config)
db.init_app(app)


@app.before_request
def create_tables():
    # init_db()
    # update_db()
    db.create_all()


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

            contours = getBoundsNew(file_path)

            return render_template('result.html', contours=contours)
    return render_template('upload.html')


@app.route('/')
def about():
    return render_template('main.html')


@app.route('/funcNew/', methods=['GET', 'POST'])
def getGolenLine():
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

        temp_dir = os.path.join(
            'static', app.config['UPLOAD_FOLDER'], f'session_golenLine')

        os.makedirs(temp_dir, exist_ok=True)

        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                flash(f'Ошибка при удалении файла {filename}: {e}')

        if file and allowed_file(file.filename):
            flash('Файл успешно загружен')
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            try:
                result = KernelLinePresenter.getGolenLine(temp_dir, file_path)
                return render_template('results.html', result=result)
            except Exception as e:
                flash(f'Ошибка при обработке изображения: {e}')
                return redirect(request.url)
    else:
        return render_template('funcNew.html')

# TEST .PT MODEL


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

# TEST .ONNX MODEL


@app.route('/regenerat/golenOnnx', methods=['GET', 'POST'])
def fetchRegenerateGolenOnnx():
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
            model_name = "golen.onnx"
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

# TEST SAMPLE


@app.route('/test_on_sample')
def test_on_sample():
    test_img_dir = os.path.join('static', 'test_images')
    test_annot_dir = os.path.join('static', 'test_annotations')
    model_name = "golen.pt"

    try:
        # Получаем список всех .png файлов
        img_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
        if not img_files:
            flash("Нет тестовых изображений в папке static/test_images", "danger")
            # return redirect(url_for('fetchRegenerateGolen'))
            return

        # Выбираем случайное изображение
        chosen_img = random.choice(img_files)
        img_path = os.path.join(test_img_dir, chosen_img)

        # Находим json с таким же названием
        base_name = os.path.splitext(chosen_img)[0]
        annot_path = os.path.join(test_annot_dir, f"{base_name}.json")

        if not os.path.exists(annot_path):
            flash(f"Разметка {base_name}.json не найдена", "warning")
            annot_path = None  # можно вызывать анализ без аннотации, если допустимо

        # Создаем временную директорию для вывода
        temp_dir = os.path.join(
            'static', app.config['UPLOAD_FOLDER'], f'session_test')
        os.makedirs(temp_dir, exist_ok=True)

        # Запускаем обработку
        result = RegenerateRouter.run_regenerate_analysis(
            temp_dir, img_path, annot_path, model_name)

        return render_template('results.html', result=[result])

    except Exception as e:
        flash(f"Ошибка при тестировании на данных: {e}", "danger")
        return
        # return redirect(url_for('fetchRegenerateGolen'))


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


@app.route('/bioAxis', methods=['GET', 'POST'])
def fetchBioAxis():
    if request.method == "POST":
        image_files = request.files.getlist('images')
        annotation_files = request.files.getlist('annotations')

        if not image_files or all(f.filename == '' for f in image_files):
            flash('Изображения не выбраны')
            return redirect(request.url)

        # Создаем временную директорию с меткой времени
        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp = "bio_axis"
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
            result = BiomecAxisPresenter.handle_data(
                temp_dir, img_path, annot_path)
            results.append(result)
        except Exception as e:
            flash(f'Ошибка при обработке: {e}')

        if results:
            flash('Файлы успешно загружены и обработаны')
            return render_template('results_axis.html', result=results)
        else:
            return redirect(request.url)
    else:
        return render_template('bioAxis_template.html')


if __name__ == '__main__':
    app.run()
