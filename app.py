from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("final_model_1.h5")

# Define categories
categories = ["Bacterial diseases - Aeromoniasis","Bacterial gill disease","Bacterial Red disease","Fungal diseases Saprolegniasis","Healthy Fish","Parasitic diseases","Viral diseases White tail disease"]



# Configure upload folder (temporary storage)
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEMP_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "uploaded_image.png")  # Always overwrite this file

# Function to preprocess the image
def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (img_size, img_size))  # Resize to match model input
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0), img  # Return batch image and raw image for display

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save file temporarily (overwrite existing file)
            file.save(TEMP_IMAGE_PATH)
            
            # Preprocess and predict
            img_batch, img_raw = preprocess_image(TEMP_IMAGE_PATH)
            predictions = model.predict(img_batch)
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]
            predicted_label = categories[predicted_class]
            
            return render_template('result.html', image=TEMP_IMAGE_PATH, label=predicted_label)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=1000,debug=False)
