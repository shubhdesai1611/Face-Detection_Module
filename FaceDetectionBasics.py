import cv2
import mediapipe as mp
import time

mpFace = mp.solutions.face_detection
faceDetection = mpFace.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

ptime = 0
cap = cv2.VideoCapture("videos/2.mp4")

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(imgRGB, detection, CONNECTION)
            height, width, channel = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = (
                int((bboxC.xmin) * width),
                int(bboxC.ymin * height),
                int((bboxC.width) * width),
                int(bboxC.height * height),
            )
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(detection.score[0]*100)}%",
                (bbox[0], bbox[1]),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                2,
            )

            # print(id, detection)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2
    )
    cv2.imshow("Video Frame", img)
    cv2.waitKey(1)
