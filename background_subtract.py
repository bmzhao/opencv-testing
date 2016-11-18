import numpy as np
import cv2
import os.path


# http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html

def cropped(frame):
    '''
    returns half of the original frame (column wise)
    :param frame:
    :return:
    '''
    return frame[:, :frame.shape[1] / 2, :]


if __name__ == '__main__':
    cap = cv2.VideoCapture('moving-car.mp4')
    fgbg = cv2.BackgroundSubtractorMOG()
    output_file = 'output.avi'
    if os.path.isfile(output_file):
        os.remove(output_file)
    out = cv2.VideoWriter(output_file, cv2.cv.CV_FOURCC(*'MJPG'), 20.0, (640, 480), False)

    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        fgmask = fgbg.apply(cropped(frame))
        cv2.imshow('frame', fgmask)
        out.write(fgmask)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
