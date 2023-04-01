import darknet
import cv2

# darknet helper function to run detection on image


def darknet_helper(img, width, height):
    darknet_image = darknet.make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                             interpolation=cv2.INTER_LINEAR)
    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height
    #    run model on darknet style image to get detections
    darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image)
    darknet.free_image(darknet_image)
    return detections, width_ratio, height_ratio


if __name__ == "__main__":
    network, class_names, class_colors = darknet.load_network("cfg/yolov4.cfg",
                                                              "cfg/coco.data",
                                                              "yolov4.weights")
    # network, class_names, class_colors = darknet.load_network("build/darknet/x64/cfg/yolov4.cfg",
    #                                                           "build/darknet/x64/cfg/coco.data",
    #                                                           "build/darknet/x64/yolov4.weights")
    vid = cv2.VideoCapture(0)

    while True:
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        ret, image = vid.read()

        # image = cv2.imread("data/person.jpg")
        detections, width_ratio, height_ratio = darknet_helper(image, width, height)

        for label, confidence, bbox in detections:
            left, top, right, bottom = darknet.bbox2points(bbox)
            left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
                bottom * height_ratio)
            cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors[label], 2)
            cv2.imshow('vid', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()