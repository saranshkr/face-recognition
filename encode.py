import argparse
import os
import pickle
import cv2
import face_recognition

from arguments import arg_map


def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect faces and encode them into a pickle file')
    parser.add_argument('-d', '--dataset-dir', help="path to the dataset directory")
    parser.add_argument('-m', '--detection-model', choices=['hog', 'cnn'], help='face detection model to use')
    parser.add_argument('-s', '--show-faces', action='store_true',
                        help='show detected face locations in each image (not recommended for large datasets)')
    parser.add_argument('-e', '--encodings', help='path to serialized facial encodings')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='increase output verbosity')
    args = vars(parser.parse_args())

    dataset_dir = args['dataset_dir'] if args['dataset_dir'] else arg_map['dataset_dir']
    detection_model = args['detection_model'] if args['detection_model'] else arg_map['detection_model']
    encodings_file = args['encodings'] if args['encodings'] else arg_map['encodings']
    verbosity = args['verbosity']
    show_faces = args['show_faces']

    if verbosity >= 2:
        print(f'Dataset directory: {dataset_dir}')
        print(f'Face detection model: {detection_model}')
        print(f'Facial encodings file: {encodings_file}')
        print(f'Verbosity level: {verbosity}')

    return (dataset_dir, detection_model, encodings_file, verbosity, show_faces)


def walk_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file), file


if __name__ == '__main__':
    (dataset_dir, detection_model, encodings_file, verbosity, show_faces) = parse_arguments()

    known_face_encodings = []
    known_face_names = []
    WINDOW_NAME = "Image"

    if verbosity == 0:
        print('Processing images...')

    for image_path, file in walk_dir(dataset_dir):
        image = cv2.imread(image_path)
        name = file.split('.')[0][:-2]
        if verbosity >= 1:
            print(f'Processing {image_path}...')

        # convert bgr (openCV default) to rgb (default for dlib)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_image, model=detection_model)
        if verbosity >= 2:
            print(f'Boxes: {boxes}')

        if show_faces:
            faces = image
            for (top, right, bottom, left) in boxes:
                faces = cv2.rectangle(faces, (left, top), (right, bottom), (0, 0, 255))
            cv2.imshow(WINDOW_NAME, faces)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        encodings = face_recognition.face_encodings(rgb_image, boxes)

        for encoding in encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    if verbosity == 0:
        print(f'Writing facial encodings to {encodings_file}')
    else:
        print(f'Writing {len(known_face_names)} facial encodings to {encodings_file}')

    data = {'encodings': known_face_encodings, 'names': known_face_names}
    with open('encodings.pickle', "wb") as f:
        f.write(pickle.dumps(data))
