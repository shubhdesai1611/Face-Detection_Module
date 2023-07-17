import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFace = mp.solutions.face_detection
        self.faceDetection = self.mpFace.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(imgRGB, detection, CONNECTION)
                height, width, channel = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = (
                    int((bboxC.xmin) * width),
                    int(bboxC.ymin * height),
                    int((bboxC.width) * width),
                    int(bboxC.height * height),
                )
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancydraw(img, bbox)
                    cv2.putText(
                        img,
                        f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1] + 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (255, 0, 255),
                        2,
                    )
        return img, bboxs

        # print(id, detection)

    def fancydraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # thickness for top left x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # thickness for top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # thickness for bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        # thickness for bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        return img


def main():
    ptime = 0
    cap = cv2.VideoCapture("videos/2.mp4")
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2
        )
        cv2.imshow("Video Frame", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
