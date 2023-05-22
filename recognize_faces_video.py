import argparse
import cv2
import os
import pickle
import face_recognition
import imutils

from arguments import arg_map


def parse_arguments():
    parser = argparse.ArgumentParser(description='Recognise faces in test video files from saved encodings')
    parser.add_argument('-d', '--test-videos-dir', help='path to test videos directory')
    parser.add_argument('-m', '--detection-model', choices=['hog', 'cnn'], help='face detection model to use')
    parser.add_argument('-e', '--encodings', help='path to serialized facial encodings')
    args = vars(parser.parse_args())

    test_videos_dir = args['test_videos_dir'] if args['test_videos_dir'] else arg_map['test_videos_dir']
    detection_model = args['detection_model'] if args['detection_model'] else arg_map['detection_model']
    encodings_file = args['encodings'] if args['encodings'] else arg_map['encodings']

    return (test_videos_dir, detection_model, encodings_file)


def walk_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


if __name__ == '__main__':
    (test_videos_dir, detection_model, encodings_file) = parse_arguments()

    with open(encodings_file, 'rb') as f:
        data = pickle.loads(f.read())

    for file in walk_dir(test_videos_dir):
        print(f'Recognising faces in {file}...')

        capture = cv2.VideoCapture(file)
        # writer = None
        # time.sleep(2.0)

        while True:
            ret, frame = capture.read()
            # convert the input frame from BGR to RGB
            rgb_image = frame[:, :, ::-1]
            # resize the frame (to speedup processing)
            rgb_image = imutils.resize(rgb_image, width=1024)
            r = frame.shape[1] / float(rgb_image.shape[1])

            boxes = face_recognition.face_locations(rgb_image, model='hog')
            encodings = face_recognition.face_encodings(rgb_image, boxes)
            names = []

            # loop over the facial embeddings
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
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
                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # if the video writer is None *AND* we are supposed to write
            # the output video to disk initialize the writer
            # if writer is None and args["output"] is not None:
            # 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # 	writer = cv2.VideoWriter(args["output"], fourcc, 20,
            # 		(frame.shape[1], frame.shape[0]), True)
            # if the writer is not None, write the frame with recognized
            # faces to disk
            # if writer is not None:
            # 	writer.write(frame)
            # check to see if we are supposed to display the output frame to
            # the screen

        cv2.destroyAllWindows()
        capture.release()
        # check to see if the video writer point needs to be released
        # if writer is not None:
        # 	writer.release()
