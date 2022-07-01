import urllib.request
import cv2
import urllib.request

from matplotlib import pyplot as plt

haarcascade_path = r'D:\uni\Coursera\ComputerVision1IBM\week5\eye_detector.xml'
detector = cv2.CascadeClassifier(haarcascade_path)


def plt_show(image, title="", gray=False, size=(12, 10)):
    temp = image

    # convert to grayscale images
    # if gray == False:
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

    # change image size
    # rcParams['figure.figsize'] = [10, 10]
    # remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()

# def openCam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, img = cap.read()
#         cv2.imshow('webcam', img)
#         k = cv2.waitKey(10)
#         if k == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

def detect_obj(image):
    #clean your image
    # plt_show(image)
    ## detect the car in the image
    object_list = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=4,
        minSize=(80, 70))
    print(object_list)
    #for each car, draw a rectangle around it
    for obj in object_list:
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2) #line thickness
    ## lets view the image
    plt_show(image)



image_path = r"D:\uni\Coursera\ComputerVision1IBM\week5\Cristiano.jpg"
image = cv2.imread(image_path)

# print(image)

detect_obj(image)




