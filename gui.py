import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# --- SET YOUR MODEL AND IMAGE SIZE ---
MODEL_PATH = 'model2_catsVSdogs_10epoch.keras'  # Update with your model path
IMAGE_SIZE = (128, 128)  # Change to your input size

model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = Image.open(img_path).resize(IMAGE_SIZE).convert('RGB')
    img = ImageOps.exif_transpose(img)    
    img_arr = np.array(img) / 255.0
    if img_arr.ndim == 2:  # handle grayscale images
        img_arr = np.stack([img_arr]*3, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)[0]
    classes = ['Cat', 'Dog']
    return classes[np.argmax(pred)], pred

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        img = ImageOps.exif_transpose(img)
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        label_result.config(text='Predicting...')
        prediction, raw = predict_image(file_path)
        label_result.config(text=f"This is probably a {prediction}")

root = tk.Tk()
root.title("Cat vs Dog Classifier")

Button(root, text="Upload Image", command=upload_image).pack(pady=10)
panel = Label(root)
panel.pack()
label_result = Label(root, text="", font=('Roboto', 30))
label_result.pack(pady=10)

root.mainloop()
