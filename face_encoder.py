from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="Path to image dataset")
ap.add_argument("-e", "--encodings", required=True, help="Save encoded images as")
ap.add_argument("-d", "--detection-methods", type=str, default="cnn",
                help="which face detection model do you want to use. Options are cnn and hog")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset

print("[INFO] quantifying faces...")
Paths_to_image = list(paths.list_images(args["training"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownLabels = []

# loop over the image paths
for (i, Path) in enumerate(Paths_to_image):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(Paths_to_image)))
    label = Path.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(Path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    rectangles = face_recognition.face_locations(rgb, model=args.get("detection_method"))
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, rectangles)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownLabels.append(label)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "labels": knownLabels}
with  open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))

