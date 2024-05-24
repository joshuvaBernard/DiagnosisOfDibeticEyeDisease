import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image, ImageTk,ImageDraw
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np





class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.original_image_tk = None
        self.grayscale_image_tk = None
        self.canny_image_tk = None
        self.progress_bar = None
        self.create_widgets()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.progress_bar.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
            self.master.update()  # Update the window to show the progress bar
            self.load_model_and_predict(file_path)

    print("-------Initiating User Interface-------")
    def create_widgets(self):
        self.master.title("Fundus Image classifier")
        self.master.geometry("700x400")  # Fixed window size

        self.original_label = tk.Label(self.master, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=70, pady=10)

        self.original_image_label = tk.Label(self.master)
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)

        self.grayscale_label = tk.Label(self.master, text="Grayscale Image")
        self.grayscale_label.grid(row=0, column=1, padx=70, pady=10)

        self.grayscale_image_label = tk.Label(self.master)
        self.grayscale_image_label.grid(row=1, column=1, padx=10, pady=10)

        self.canny_label = tk.Label(self.master, text="Canny Edge Image")
        self.canny_label.grid(row=0, column=2, padx=70, pady=10)

        self.canny_image_label = tk.Label(self.master)
        self.canny_image_label.grid(row=1, column=2, padx=10, pady=10)

        self.prediction_label = tk.Label(self.master, padx=10, pady=5)
        self.prediction_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.open_file_button = tk.Button(self.master, text="Open Image", command=self.open_image)
        self.open_file_button.grid(row=3, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="s")
        print("Select an image to Predict")

        # Progress bar widget
        self.progress_bar = tk.ttk.Progressbar(self.master, length=300, mode="indeterminate")



    def load_model_and_predict(self, file_path):
        # Simulating model loading with sleep
        self.progress_bar.start(10)  # Start the progress bar
        self.master.update()  # Update the window to show the progress bar
        print("Loading Model........:)")
        model = load_model('InceptionV3(model).h5')
        self.progress_bar.stop()  # Stop the progress bar
        self.progress_bar.grid_forget()  # Remove the progress bar from the layout
        self.master.update()  # Update the window to hide the progress bar

        print("........predecting........")
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.

        predictions = model.predict(x)
        print(predictions)
        class_probabilities = predictions[0]
        predicted_class_index = np.argmax(class_probabilities)
        other_predicted_class_index = np.argmin(class_probabilities)
        class_labels = ['DR', 'Normal']
        predicted_class_label = class_labels[predicted_class_index]
        other_predicted_class_label = class_labels[other_predicted_class_index]
        predicted_class_probability = min(class_probabilities[predicted_class_index], 1.0)
        other_class_probability = 1.0 - predicted_class_probability

        original_image, grayscale_image, canny_image = self.preprocess_image(file_path)

        original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)).resize((200, 200))
        self.original_image_tk = ImageTk.PhotoImage(original_image)
        self.original_image_label.config(image=self.original_image_tk)

        grayscale_image = Image.fromarray(grayscale_image).resize((200, 200))
        self.grayscale_image_tk = ImageTk.PhotoImage(grayscale_image)
        self.grayscale_image_label.config(image=self.grayscale_image_tk)

        canny_image = Image.fromarray(canny_image).resize((200, 200))
        self.canny_image_tk = ImageTk.PhotoImage(canny_image)
        self.canny_image_label.config(image=self.canny_image_tk)

        self.prediction_label.config(
            text=f"Prediction: {predicted_class_label} with probablity of:{predicted_class_probability * 100:.2f}% || {other_predicted_class_label}  with probablity of:{other_class_probability * 100:.2f}%"  )

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny_image = cv2.Canny(grayscale_image, 150, 50)
        grayscale_image_rgb = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        return image, grayscale_image_rgb, canny_image


