from flask import *
from flask_session import Session
from datetime import timedelta
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_DIRECTORY1'] = 'uploads1/'
app.config['UPLOAD_DIRECTORY2'] = 'uploads2/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB
app.config['ALLOWED_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']

app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)

# The maximum number of items the session stores 
# before it starts deleting some, default 500
app.config['SESSION_FILE_THRESHOLD'] = 100  
Session(app)

@app.route('/')
def index():
  if not session.get("private_folder"):
    session['private_folder'] = "session_files/{:06}/".format(np.random.randint(1000000))
    os.mkdir(session['private_folder'])
    os.mkdir(session['private_folder']+app.config['UPLOAD_DIRECTORY1'])
    os.mkdir(session['private_folder']+app.config['UPLOAD_DIRECTORY2']) 

  files = os.listdir(session['private_folder']+app.config['UPLOAD_DIRECTORY1'])
  images1 = []

  for file in files:
    if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
      images1.append(file)

  files = os.listdir(session['private_folder']+app.config['UPLOAD_DIRECTORY2'])
  images2 = []

  for file in files:
    if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
      images2.append(file)
  
  return render_template('index.html', images1=images1, images2=images2)

@app.route('/upload1', methods=['POST'])
def upload1():
  try:
    file = request.files['file']

    if file:
      extension = os.path.splitext(file.filename)[1].lower()

      if extension not in app.config['ALLOWED_EXTENSIONS']:
        return 'File is not an image.'
        
      file.save(os.path.join(
        session['private_folder']+app.config['UPLOAD_DIRECTORY1'],
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
        session['private_folder']+app.config['UPLOAD_DIRECTORY2'],
        secure_filename(file.filename)
      ))
  
  except RequestEntityTooLarge:
    return 'File is larger than the 16MB limit.'
  
  return redirect('/')

@app.route('/train', methods=['POST'])
def train():
  # TO DO
  return

@app.route('/serve-image1/<filename>', methods=['GET'])
def serve_image1(filename):
  return send_from_directory(session['private_folder']+app.config['UPLOAD_DIRECTORY1'], filename)

@app.route('/serve-image2/<filename>', methods=['GET'])
def serve_image2(filename):
  return send_from_directory(session['private_folder']+app.config['UPLOAD_DIRECTORY2'], filename)