import face_recognition
import cv2
import os
from datetime import datetime
from google.colab.patches import cv2_imshow

# Load known student faces
known_face_encodings = []
known_face_names = []

dataset_path = "students_dataset"

for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    student_name = os.path.splitext(image_name)[0]

    known_face_encodings.append(encoding)
    known_face_names.append(student_name)

print("Student faces encoded successfully")

# Attendance function
marked_students = []

def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        date = now.strftime("%d-%m-%Y")
        time = now.strftime("%H:%M:%S")
        f.write(f"{name},{date},{time}\n")

# Create CSV with header if not exists
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("Name,Date,Time\n")

# Load class image
test_image = face_recognition.load_image_file("class.jpg")

face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Recognize faces & mark attendance
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

        if name not in marked_students:
            mark_attendance(name)
            marked_students.append(name)

    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_bgr, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
cv2_imshow(image_bgr)
print("Attendance marked successfully")
