import cv2
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

# Create a dictionary of known face encodings
name_encoding_dict = {
    'Mohit': mohit_encodings,
    'Joel': joel_encodings,
    'Siddanta': siddanta_encodings
}

# Initialize face cascade classifier
cascadePath = r'C:\Users\mohit\OneDrive\Desktop\mohit desai\sem4 projects\skill lab prj\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initialize video capture
capture = cv2.VideoCapture(0)
counter = 0
face_name = "Unknown"

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
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

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Exit loop on 'd' key press
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
