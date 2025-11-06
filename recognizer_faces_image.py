import face_recognition
import cv2
import os
import pickle
import argparse

app = argparse.ArgumentParser()
app.add_argument('-e', '--encodings', type=str, required=True, help='Path to input encodings file')
app.add_argument('-i', '--image', type=str, required=True, help='Path to input image')
app.add_argument('-d', '--detection_method', type=str, default='cnn', help='Face detection model to use: "hog" or "cnn"')
args = vars(app.parse_args())

print("[info] loading encodings...")
data = pickle.loads(open(args['encodings'], 'rb').read())

img = cv2.imread(args['image'])
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("[info] recognizing faces...")
box = face_recognition.face_locations(rgbImg, model=args['detection_method'])
encodings = face_recognition.face_encodings(rgbImg, box)

names = []

for ed in encodings:
    matches = face_recognition.compare_faces(data['encodings'], ed, 0.6)
    name = "Khong nhan ra mat may roi qua buon :("

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data['names'][i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    names.append(name)

for ((top, right, bottom, left), name) in zip(box, names):
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

cv2.imshow("Image", img)
cv2.waitKey(0)

