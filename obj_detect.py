import cv2.cv2 as cv2
from time import time
import numpy as np


def initialize(is_cuda):
    net = cv2.dnn.readNet("model/best.onnx")
    if is_cuda:
        print("Running on CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


capture = cv2.VideoCapture("media/rene_game.mp4")
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.6


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    predictions = net.forward()
    return predictions


def load_classes():
    with open("model/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        conf = row[4]
        if conf >= 0.93:

            classes_scores = row[5:]
            _, _, _, max_index = cv2.minMaxLoc(classes_scores)
            class_id = max_index[1]
            if classes_scores[class_id] > .25:
                confidences.append(conf)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_frame(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

#is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

class_list = load_classes()
net = initialize(True)
total_frames = 0

while True:

    _, frame = capture.read()
    if frame is None:
        print("Ended")
        break

    start_time = time()
    frame = cv2.resize(frame, (1600, 900))
    inputImage = format_frame(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    total_frames += 1

    for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[class_id], (box[0], box[1] - 10), 0, 0.5, (0, 0, 0))

    end_time = time()

    fps = 1 / np.round(end_time - start_time, 2)
    cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), 0, 1.5, (0, 0, 255), 2)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("User exit")
        break

print("Total frames: " + str(total_frames))

