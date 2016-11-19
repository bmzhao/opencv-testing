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


def expand_blob(frame):
    '''
    blur, then threshold, then blur again
    :param frame:
    :return:
    '''

    for i in range(3):
        frame = cv2.blur(frame, (10, 10))
        # frame = cv2.medianBlur(frame,21)
        # frame = cv2.GaussianBlur(frame,(11,11),0)
        frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)[1]
    return frame


def run(output_file=None):
    cap = cv2.VideoCapture('moving-car.mp4')

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255

    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 10000

    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1
    # params.maxCircularity = 0.7

    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # params.maxInertiaRatio = 0.8

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    fgbg = cv2.BackgroundSubtractorMOG2()

    if output_file is not None:
        if os.path.isfile(output_file):
            os.remove(output_file)
        out = cv2.VideoWriter(output_file, cv2.cv.CV_FOURCC(*'MJPG'), 20.0, (640, 480), False)

    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        fgmask = fgbg.apply(cropped(frame))
        fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.medianBlur(fgmask, 41)

        keypoints = detector.detect(fgmask)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        frame_with_keypoints = cv2.drawKeypoints(
            fgmask, keypoints,
            np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('frame', frame_with_keypoints)
        if output_file is not None:
            out.write(cv2.cvtColor(frame_with_keypoints, cv2.COLOR_BGR2GRAY))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    if output_file is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
