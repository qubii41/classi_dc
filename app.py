import os
from flask import Flask, render_template, request, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret123'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model/model_cnn2.keras')
class_names = ['KTP', 'Pasfoto', 'Rapor']
threshold = 0.75

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if 'status' not in session:
        session['status'] = {cls: False for cls in class_names}

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocessing
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            pred = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred)

            if confidence < threshold:
                prediction = "Tidak dikenali"
            else:
                prediction = predicted_class
                status = session['status']
                status[predicted_class] = True
                session['status'] = status

    return render_template('index.html', prediction=prediction, confidence=confidence,
                           filename=filename, status=session['status'])

@app.route('/reset')
def reset():
    session.pop('status', None)
    return render_template('index.html', prediction=None, confidence=None, filename=None,
                           status={cls: False for cls in class_names})

if __name__ == '__main__':
    app.run(debug=True)