import numpy as np
import cv2


# http://stackoverflow.com/a/14510607 for rasb pi

# http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
def get_frame(video, frame_number=0):
    counter = 0
    while counter < frame_number:
        video.read()
        counter += 1
    ret = video.read()[1]
    return ret


def normalize(frame):
    return cv2.cvtColor(cropped(frame), cv2.COLOR_BGR2GRAY)

def cropped(frame):
    '''
    returns half of the original frame
    :param frame:
    :return:
    '''
    # https://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial
    # print frame.ndim
    # print frame.shape
    # print frame.size
    # print frame.dtype
    return frame[:,:frame.shape[1]/2,:]


def watch_difference():
    cap = cv2.VideoCapture('moving-car.mp4')
    base_frame = normalize(get_frame(cap, 0))
    # cv2.imshow('frame', base_frame)
    # cv2.waitKey(0)

    while (True):
        # Capture frame-by-frame
        ret, new_frame = cap.read()
        new_frame = normalize(new_frame)
        # new_frame = cv2.GaussianBlur(new_frame,(11,11),0)
        new_frame -= base_frame
        new_frame = cv2.adaptiveThreshold(new_frame,255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV,455,7)
        # new_frame = cv2.threshold(new_frame,100,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


        # Display the resulting frame
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def watch_normal():
    cap = cv2.VideoCapture('moving-car.mp4')

    while (True):
        # Capture frame-by-frame
        ret, new_frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # watch_normal()
    watch_difference()