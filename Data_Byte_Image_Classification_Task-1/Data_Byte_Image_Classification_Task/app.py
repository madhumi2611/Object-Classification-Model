from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the custom YOLOv8 model
model_path = 'Models/best.pt'  # Path to your custom model
model = YOLO(model_path)  # Load the model directly using ultralytics package

# Define the class names list
class_names = [
    'Akaza', 'Daki', 'Giyu Tomioka', 'Gyomei Himejima', 'Gyutaro',
    'Inosuke Hashibira', 'Kanao Tsuyuri', 'Kyojuro Rengoku', 'Mitsuri Kanroji',
    'Muichiro Tokito', 'Muzan Kibutsuji', 'Nezuko Kamado', 'Obanai Iguro', 
    'Sanemi', 'Shinobu Kocho', 'Tanjiro Kamado', 'Tengen Uzui', 'Zenitsu Agatsuma'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Perform inference
        results = model(filepath)

        # Process results
        processed_results = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # Convert the bounding box to a list
                class_index = int(box.cls.item())
                class_name = class_names[class_index]  # Get the class name
                processed_results.append({
                    'class': class_name,
                    'confidence': box.conf.item(),
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3]
                })

        return render_template('result.html', results=processed_results, filepath=file.filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
