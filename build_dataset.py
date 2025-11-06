import argparse
import os
import cv2

app = argparse.ArgumentParser(description="Build a dataset from images in a directory.")
app.add_argument('-o', '--output', type=str, required=True, help='Output directory for the dataset')
args = vars(app.parse_args())

person_name = input("Nhap ten cua ban (ten thu muc se duoc tao): ").strip()
if person_name == "":
    print("Chua nhap ten. Thoat.")
    exit(1)

person_dir = os.path.join(args["output"], person_name)  #folder riêng theo tên
os.makedirs(person_dir, exist_ok=True)


video_dir = cv2.VideoCapture(0) 
count = len(os.listdir(person_dir)) 

while True:
    message, frame = video_dir.read()
    if not message:
        print("Error reading from camera")
        break

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # enter 's' to save the frame
    if key == ord('s'):
        img_path = os.path.join(person_dir, f"{str(count).zfill(5)}.png")
        cv2.imwrite(img_path, frame)
        count += 1
    elif key == ord('e'):
        break

print("[INFO] {} images saved to {}".format(count, args["output"]))
video_dir.release()
cv2.destroyAllWindows()
