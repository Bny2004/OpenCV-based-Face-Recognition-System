import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from keras.models import load_model

siamese_model = load_model('siamese_face_model.h5', compile=False)

def preprocess_image(img):
    if len(img.shape) == 2:
        img_gray = img  
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (100, 100))
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=-1)  
    img_expanded = np.expand_dims(img_expanded, axis=0)  
    return img_expanded

def generate_embeddings(img):
    img_preprocessed = preprocess_image(img)
    embeddings = siamese_model.predict(img_preprocessed)
    return embeddings

def compare_embeddings(embeddings1, embeddings2):
    distance = np.linalg.norm(embeddings1 - embeddings2)
    print(distance)
    return distance 

known_embeddings = {}
known_embeddings['Mohit'] = generate_embeddings(cv2.imread('Images/Mohit.jpeg'))
known_embeddings['siddhanta'] = generate_embeddings(cv2.imread('Images/siddhanta.jpeg'))
known_embeddings['joel'] = generate_embeddings(cv2.imread('Images/joel.jpeg'))




class VideoCaptureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)  
        self.cascadePath = 'haarcascade_frontalface_default.xml' 
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath) 

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

        self.canvas_video = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas_video.pack(side=tk.LEFT)

        self.canvas_image = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas_image.pack(side=tk.LEFT)

        self.btn_start = tk.Button(window, text="Start", width=10, command=self.start_capture)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)

        self.btn_capture = tk.Button(window, text="Capture Image", width=15, command=self.capture_image)
        self.btn_capture.pack(anchor=tk.CENTER, expand=True)

        self.btn_save = tk.Button(window, text="Save Image", width=15, command=self.save_image)
        self.btn_save.pack(anchor=tk.CENTER, expand=True)
        self.btn_save.config(state=tk.DISABLED)  

        self.btn_recapture = tk.Button(window, text="Recapture Image", width=15, command=self.recapture_image)
        self.btn_recapture.pack(anchor=tk.CENTER, expand=True)
        self.btn_recapture.config(state=tk.DISABLED)  

        self.entry_filename = tk.Entry(window)
        self.entry_filename.pack(anchor=tk.CENTER, pady=5)
        self.entry_filename.config(state=tk.DISABLED)  

        self.is_capturing = False
        self.captured_image = None
        self.update()

    def start_capture(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.btn_start.config(text="Stop")
            self.btn_capture.config(state=tk.NORMAL)
        else:
            self.is_capturing = False
            self.btn_start.config(text="Start")
            self.btn_capture.config(state=tk.DISABLED)
            self.show_captured_image()

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.captured_image = Image.fromarray(frame)
            self.show_captured_image()
            self.btn_save.config(state=tk.NORMAL)
            self.btn_recapture.config(state=tk.NORMAL)
            self.entry_filename.config(state=tk.NORMAL)

    def save_image(self):
        filename = self.entry_filename.get()
        if filename and self.captured_image:
            self.captured_image.save(f"{filename}.jpg")
            print(f"Image captured and saved as {filename}.jpg!")
            self.recapture_image()  
            self.btn_save.config(state=tk.DISABLED)
            self.btn_recapture.config(state=tk.DISABLED)
            self.entry_filename.delete(0, tk.END)
            self.entry_filename.config(state=tk.DISABLED)

    def recapture_image(self):
        self.captured_image = None
        self.canvas_image.delete("all")

    def show_captured_image(self):
        self.canvas_image.delete("all")
        if self.captured_image:
            self.photo = ImageTk.PhotoImage(image=self.captured_image)
            self.canvas_image.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(30, self.show_captured_image)  



    def update(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]

                    embeddings = generate_embeddings(face)

                    recognized_name_arr = {"Unknown": 40}
                    for name, known_embedding in known_embeddings.items():
                        recognized_name_arr[name] = compare_embeddings(embeddings, known_embedding)
                     
                    recognized_name =  max(recognized_name_arr, key=lambda k: recognized_name_arr[k])
                    
                
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    cv2.putText(frame, recognized_name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)



                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas_video.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(30, self.update)

    def close(self):
        if self.is_capturing:
            self.cap.release()
        self.window.destroy()

root = tk.Tk()
app = VideoCaptureApp(root, "face identification app")
root.protocol("WM_DELETE_WINDOW", app.close)
root.mainloop()
