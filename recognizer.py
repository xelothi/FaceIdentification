from typing import Dict, Any
import face_recognition
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="Path to input dataset with faces encodings")
ap.add_argument("-i", "--image", required=True, help="path to image you want to be recognized")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known face
print("[INFO] loading encodings...")
dataset = pickle.loads(open(args["encodings"], "rb").read())
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rectangles = face_recognition.face_locations(rgb, model=args.get("detection-method"))
encodings = face_recognition.face_encodings(rgb, rectangles)

labels= []

for encoding in encodings:
    matches = face_recognition.compare_faces(dataset["encodings"], encoding)

    if True in matches:
        matchedID = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedID:
            label = dataset["labels"][i]
            counts[label] = counts.get(label, 0) + 1
        label = max(counts, key=counts.get)
    else:
        label = "Not-Amar"
    labels.append(label)

# loop over known faces
for ((top, right, bottom, left), label) in zip(rectangles, labels):
    # draw the rectangle on the image of predicted face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

#output
cv2.imshow("Image", image)
cv2.waitKey(0)
