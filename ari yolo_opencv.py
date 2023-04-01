import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):

    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1]
                         for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


vid = cv2.VideoCapture(0)

# image = cv2.imread(args.image)
while (True):

    ret, frame = vid.read()

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(
        frame, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # filtering multiple boxes on same object
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    class_count = dict()
    box = []

    for i in indices:
        try:
            # filtered_boxes.append(boxes[i])
            # print(str(classes[class_ids[i]]))
            class_count[str(classes[class_ids[i]])] = class_count.get(
                str(classes[class_ids[i]]), 0) + 1
            box = boxes[i]
        except:
            i = i[0]
            class_count[str(classes[class_ids[i]])] = class_count.get(
                str(classes[class_ids[i]]), 0) + 1
            box = boxes[i]
            # filtered_boxes.append(boxes[i])
            # print(str(classes[class_ids[i]]))
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h))
    print(class_count)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# cv2.imshow("object detection", image)
# cv2.waitKey()

# cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()
