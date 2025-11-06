# Thực hiện ecoding khuôn mặt và tên tương ứng trong output_images vào trong encodings.pickle
import imutils 
from imutils import paths
import pickle
import argparse
import cv2
import os
import face_recognition


app = argparse.ArgumentParser()
app.add_argument('-i', '--output_images', type=str, required=True, help='Path to input directory of images')
app.add_argument('-e', '--encodings', type=str, required=True, help='Path to output encodings file')
app.add_argument('-d', '--detection_method', type=str, default='cnn', help='Face detection model to use: "hog" or "cnn"')
args = vars(app.parse_args())


print("[INFO] loading images...")
imgPaths = list(paths.list_images(args['output_images']))

#tao encoding va ten de so sanh
knownEncodings = []
knownNames = []

for (i, imgPath) in enumerate(imgPaths):
    print("[info] processing image {}/{}".format(i + 1, len(imgPaths)))
    name = imgPath.split(os.path.sep)[-2]  

    #load img len bang openCV va chuyen sang RGB
    img = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    box = face_recognition.face_locations(rgbImg, model=args['detection_method'])
    encodings = face_recognition.face_encodings(rgbImg, box)

    #duyet anh qua encoding
    for ed in encodings:
        knownEncodings.append(ed)
        knownNames.append(name)

print("[info] serializing encodings...")
dataPerson = {"encodings": knownEncodings, "names": knownNames}

with open(args['encodings'], 'wb') as f:
    f.write(pickle.dumps(dataPerson))
