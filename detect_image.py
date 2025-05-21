import cv2
from yolo import YOLO

if __name__ == '__main__':

    model_path = "./models/ERBW-Bolt.onnx"
    yolov8_detector = YOLO(model_path, conf_thres=0.2, iou_thres=0.3)

    img = cv2.imread('./test/196.jpg')

    # Detect Objects
    yolov8_detector(img)

    # Draw detections
    combined_img = yolov8_detector.draw_detections(img)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite("imgs/196_test.jpg", combined_img)
    cv2.waitKey(0)
