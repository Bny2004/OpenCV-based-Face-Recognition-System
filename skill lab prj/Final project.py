import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

# Load known face encodings
mohit = cv2.imread(r"C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\Images\Mohit.jpeg")
mohit = cv2.cvtColor(mohit, cv2.COLOR_BGR2RGB)
mohit_encodings = face_recognition.face_encodings(mohit)[0]

joel = cv2.imread(r"C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\Images\joel.jpeg")
joel = cv2.cvtColor(joel, cv2.COLOR_BGR2RGB)
joel_encodings = face_recognition.face_encodings(joel)[0]

siddanta = cv2.imread(r"C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\Images\siddhanta.jpeg")
siddanta = cv2.cvtColor(siddanta, cv2.COLOR_BGR2RGB)
siddanta_encodings = face_recognition.face_encodings(siddanta)[0]

kaustubh = cv2.imread(r"C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\Images\kaustub.jpg")
kaustubh = cv2.cvtColor(kaustubh, cv2.COLOR_BGR2RGB)
kaustubh_encodings = face_recognition.face_encodings(kaustubh)[0]

# Create a dictionary of known face encodings
name_encoding_dict = {
    'Mohit': mohit_encodings,
    'Joel': joel_encodings,
    'Siddanta': siddanta_encodings,
    'kaustubh':kaustubh_encodings
}

class VideoCaptureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)  
        self.cascadePath = r"C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\haarcascade_frontalface_default.xml"
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
            self.captured_image.save(f"Images\{filename}.jpg")
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
            isTrue, frame = self.cap.read()
            
            if isTrue:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    # Compute face encodings for detected faces
                    face_encodings = face_recognition.face_encodings(frame)

                    for face_encoding, (x, y, w, h) in zip(face_encodings, faces):
                        # Compare face encodings with known encodings
                        matches = face_recognition.compare_faces(list(name_encoding_dict.values()), face_encoding, tolerance=0.6)

                        if True in matches:
                            match_index = matches.index(True)
                            face_name = list(name_encoding_dict.keys())[match_index]
                        else:
                            face_name = "Unknown"

                        # Draw rectangle and text on the frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
                        cv2.putText(frame, face_name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                        self.canvas_video.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(60, self.update)

    def close(self):
        if self.is_capturing:
            self.cap.release()
        self.window.destroy()

root = tk.Tk()
app = VideoCaptureApp(root, "face identification app")
root.protocol("WM_DELETE_WINDOW", app.close)
root.mainloop()