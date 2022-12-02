from flask import *
from flask_session import Session
from datetime import timedelta
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
from maml import Trainer
import torch
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_DIRECTORY1'] = 'uploads1/'
app.config['UPLOAD_DIRECTORY2'] = 'uploads2/'
app.config['UPLOAD_DIRECTORY3'] = 'uploads3/'
app.config['UPLOAD_DIRECTORY4'] = 'uploads4/'
app.config['UPLOAD_DIRECTORY5'] = 'uploads5/'
app.config['UPLOAD_TEST'] = 'test/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)

# The maximum number of items the session stores 
# before it starts deleting some, default 500
app.config['SESSION_FILE_THRESHOLD'] = 100
Session(app)
# TODO class names, drag and drop for the test images

@app.route('/')
def index():
    if not session.get("private_folder"):
        session['private_folder'] = "session_files/{:06}/".format(np.random.randint(1000000))
        session['learner'] = Trainer()
        os.mkdir(session['private_folder'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY2'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY3'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY4'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY5'])
        os.mkdir(session['private_folder'] + app.config['UPLOAD_TEST'])

    files = os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'])
    images1 = []

    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images1.append(file)

    files = os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY2'])
    images2 = []

    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images2.append(file)

    files = os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY3'])
    images3 = []

    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images3.append(file)

    files = os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY4'])
    images4 = []

    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images4.append(file)

    files = os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY5'])
    images5 = []

    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            images5.append(file)

    return render_template('index.html', images1=images1, images2=images2, images3=images3, images4=images4,
                           images5=images5)


@app.route('/upload1', methods=['POST'])
def upload1():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_DIRECTORY1'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')

@app.route('/upload_images1_drop', methods=['POST'])
def upload_images1_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_DIRECTORY1'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')


@app.route('/upload_images2_drop', methods=['POST'])
def upload_images2_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_DIRECTORY2'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')
@app.route('/upload2', methods=['POST'])
def upload2():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_DIRECTORY2'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')

@app.route('/upload_images3_drop', methods=['POST'])
def upload_images3_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_DIRECTORY3'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')
@app.route('/upload3', methods=['POST'])
def upload3():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_DIRECTORY3'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')


@app.route('/upload_images4_drop', methods=['POST'])
def upload_images4_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_DIRECTORY4'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')
@app.route('/upload4', methods=['POST'])
def upload4():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_DIRECTORY4'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')


@app.route('/upload_images5_drop', methods=['POST'])
def upload_images5_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_DIRECTORY5'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')
@app.route('/upload5', methods=['POST'])
def upload5():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_DIRECTORY5'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/')

@app.route('/train', methods=['POST'])
def train():
    session['learner'].fine_tune(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'],
                                 session['private_folder'] + app.config['UPLOAD_DIRECTORY2'],
                                 session['private_folder'] + app.config['UPLOAD_DIRECTORY3'],
                                 session['private_folder'] + app.config['UPLOAD_DIRECTORY4'],
                                 session['private_folder'] + app.config['UPLOAD_DIRECTORY5'])
    return redirect('/use_model')


@app.route('/serve-image1/<filename>', methods=['GET'])
def serve_image1(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'], filename)


@app.route('/serve-image2/<filename>', methods=['GET'])
def serve_image2(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_DIRECTORY2'], filename)

@app.route('/serve-image3/<filename>', methods=['GET'])
def serve_image3(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_DIRECTORY3'], filename)

@app.route('/serve-image4/<filename>', methods=['GET'])
def serve_image4(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_DIRECTORY4'], filename)

@app.route('/serve-image5/<filename>', methods=['GET'])
def serve_image5(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_DIRECTORY5'], filename)

@app.route('/serve-test-image/<filename>', methods=['GET'])
def serve_test_image(filename):
    return send_from_directory(session['private_folder'] + app.config['UPLOAD_TEST'], filename)


@app.route('/use_model')
def use_model():
    files = os.listdir(session['private_folder'] + app.config['UPLOAD_TEST'])
    images = []
    for file in files:
        if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            c = session['learner'].inference(session['private_folder'] + app.config['UPLOAD_TEST'] + '/' + file)
            images.append((file, c))

    return render_template('use_model.html', images=images)


@app.route('/download_pt', methods=['POST'])
def get_pt_model():
    torch.save(session['learner'].model, session['private_folder'] + 'model.pt')
    return send_file(session['private_folder'] + 'model.pt')


@app.route('/download_onnx', methods=['POST'])
def get_onnx_model():
    torch.onnx.export(session['learner'].model, torch.rand(1, 3, 84, 84), session['private_folder'] + 'model.onnx')
    return send_file(session['private_folder'] + 'model.onnx')

@app.route('/upload_test_file', methods=['POST'])
def upload_test_file():
    try:
        file = request.files['file']

        if file:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'File is not an image.'

            file.save(os.path.join(
                session['private_folder'] + app.config['UPLOAD_TEST'],
                secure_filename(file.filename)
            ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/use_model')

@app.route('/upload_images_test_drop', methods=['POST'])
def upload_images_test_drop():
    try:
        files = request.files.getlist('files')
        for file in files:
            # file = request.files['file']

            if file:
                extension = os.path.splitext(file.filename)[1].lower()

                if extension not in app.config['ALLOWED_EXTENSIONS']:
                    return 'File is not an image.'

                file.save(os.path.join(
                    session['private_folder'] + app.config['UPLOAD_TEST'],
                    secure_filename(file.filename)
                ))

    except RequestEntityTooLarge:
        return 'File is larger than the 16MB limit.'

    return redirect('/use_model')

@app.route('/clean1', methods=['POST'])
def clean1():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY1']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'], file))
    return redirect('/')

@app.route('/clean2', methods=['POST'])
def clean2():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY2']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY2'], file))
    return redirect('/')

@app.route('/clean3', methods=['POST'])
def clean3():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY3']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY3'], file))
    return redirect('/')

@app.route('/clean4', methods=['POST'])
def clean4():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY4']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY4'], file))
    return redirect('/')

@app.route('/clean5', methods=['POST'])
def clean5():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY5']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY5'], file))
    return redirect('/')

@app.route('/clean_all', methods=['POST'])
def clean_all():
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY1']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY1'], file))
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY2']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY2'], file))
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY3']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY3'], file))
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY4']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY4'], file))
    for file in os.listdir(session['private_folder'] + app.config['UPLOAD_DIRECTORY5']):
        os.remove(os.path.join(session['private_folder'] + app.config['UPLOAD_DIRECTORY5'], file))
    return redirect('/')
