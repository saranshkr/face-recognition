import argparse
import pickle
import os
import cv2
import face_recognition

from arguments import arg_map


def parse_arguments():
    parser = argparse.ArgumentParser(description='Recognise faces in test images from saved encodings')
    parser.add_argument('-d', '--test-images-dir', help='path to test images directory')
    parser.add_argument('-m', '--detection-model', choices=['hog', 'cnn'], help='face detection model to use')
    parser.add_argument('-e', '--encodings', help='path to serialized facial encodings')
    args = vars(parser.parse_args())

    test_images_dir = args['test_images_dir'] if args['test_images_dir'] else arg_map['test_images_dir']
    detection_model = args['detection_model'] if args['detection_model'] else arg_map['detection_model']
    encodings_file = args['encodings'] if args['encodings'] else arg_map['encodings']

    return (test_images_dir, detection_model, encodings_file)


def walk_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def resize_image_with_aspect_ratio(original_image, width=None, height=None, inter=cv2.INTER_AREA):
    print('Resizing image')
    dim = None
    (og_height, og_width) = original_image.shape[:2]

    if width is None and height is None:
        return original_image
    if width is None:
        ratio = height / float(og_height)
        dim = (int(og_width * ratio), height)
    else:
        ratio = width / float(og_width)
        dim = (width, int(og_height * ratio))

    return cv2.resize(original_image, dim, interpolation=inter)


if __name__ == '__main__':
    (test_images_dir, detection_model, encodings_file) = parse_arguments()

    with open(encodings_file, 'rb') as f:
        data = pickle.loads(f.read())

    for file in walk_dir(test_images_dir):
        image = cv2.imread(file)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f'Recognizing faces in test image {file}...')
        boxes = face_recognition.face_locations(rgb_image, model=detection_model)
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        # initialize the list of names for each face detected
        names = []

        for encoding in encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # show the output image
        resized_img = resize_image_with_aspect_ratio(image, width=1280)
        cv2.imshow("Image", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
