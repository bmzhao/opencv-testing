import numpy as np
import cv2
import os.path

# http://stackoverflow.com/a/14510607 for rasb pi
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

# video training data courtesy of: http://www.cvg.reading.ac.uk/PETS2015/a.html
# http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

# http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
def get_frame(video, frame_number=0):
    counter = 0
    while counter < frame_number:
        video.read()
        counter += 1
    return video.read()[1]


def display_frame(frame):
    cv2.imshow('frame', frame)
    cv2.waitKey(0)


def normalize(frame):
    return cv2.cvtColor(cropped(frame), cv2.COLOR_BGR2GRAY)


def cropped(frame):
    '''
    returns half of the original frame (column wise)
    :param frame:
    :return:
    '''
    return frame[:, :frame.shape[1] / 2, :]


def debug(frame):
    # https://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial
    print frame.ndim
    print frame.shape
    print frame.size
    print frame.dtype


def watch_difference(base_frame=None):
    cap = cv2.VideoCapture('moving-car.mp4')

    # Define the codec and create VideoWriter object
    # http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
    # https://stackoverflow.com/questions/22482520/fourcc-in-cv2-issues
    output_file = 'output.avi'
    if os.path.isfile(output_file):
        os.remove(output_file)
    out = cv2.VideoWriter(output_file, cv2.cv.CV_FOURCC(*'MJPG'), 20.0, (640, 480), False)

    if base_frame is None:
        base_frame = normalize(get_frame(cap, 0))

    # cv2.imshow('frame', base_frame)
    # cv2.waitKey(0)

    while (True):
        # Capture frame-by-frame
        ret, new_frame = cap.read()

        if ret is False:
            break

        new_frame = normalize(new_frame)
        # new_frame = cv2.GaussianBlur(new_frame,(5,5),0)
        new_frame -= base_frame
        new_frame = cv2.adaptiveThreshold(new_frame, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 455, 10)
        # new_frame = cv2.threshold(new_frame,100,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        out.write(new_frame)

        # Display the resulting frame
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def watch_normal():
    cap = cv2.VideoCapture('moving-car.mp4')

    while (True):
        ret, new_frame = cap.read()

        if ret is False:
            break

        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def average_frames(video_source, number_of_frames):
    frame = np.array(normalize(get_frame(video_source, 1)), dtype=np.double)

    accumulator = np.zeros(frame.shape, dtype=np.double)

    accumulator += frame
    for i in range(number_of_frames - 1):
        accumulator += np.array(normalize(get_frame(video_source, 1)), dtype=np.double)

    accumulator /= number_of_frames
    return np.array(accumulator, dtype=np.uint8)


if __name__ == '__main__':
    # watch_normal()
    cap = cv2.VideoCapture('moving-car.mp4')
    base_frame = average_frames(cap, 12)
    watch_difference(base_frame=base_frame)
