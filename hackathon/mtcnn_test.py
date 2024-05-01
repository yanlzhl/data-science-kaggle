from mtcnn import MTCNN
import cv2

img = cv2.cvtColor(cv2.imread("ivan.png"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(img)


# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)

cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

cv2.imshow("img", img)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()