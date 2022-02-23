# Extracting frames from a video file using OpenCV
# Usage: python extract_frame.py <file> <class_name> <class_index>

import os
import sys

import cv2.cv2 as cv2
import numpy as np

SAVING_FRAMES_PER_SECOND = 10


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saved_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []

    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def create_label(classifier):
    label = classifier + " 0.5" + " 0.5" + " 1.0" + " 1.0"
    return label


def main(video_file, class_name, class_index):
    filename, _ = os.path.splitext(video_file)
    filename += "-train"

    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)

    # read the video file
    cap = cv2.VideoCapture(video_file)

    # get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save

    saved_frames_duration = get_saved_frames_durations(cap, saving_frames_per_second)

    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            print("Finished extracting all the frames!")
            break

        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saved_frames_duration[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration,
            # then save the frame
            # cv2.imwrite(os.path.join(filename, f"{class_name}_{round(frame_duration, 1)}.jpg"), frame)
            cv2.imwrite(os.path.join(filename, f"{class_name}_{count}.jpg"), frame)
            # save the label
            # temp = str("labels/" + class_name + "_" + str(round(frame_duration, 1)) + ".txt")
            # with open(temp, 'w') as f:
            #     f.write(create_label(class_index))
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saved_frames_duration.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1


if __name__ == "__main__":
    video_file = sys.argv[1]
    class_name = sys.argv[2]
    class_index = sys.argv[3]
    main(video_file, class_name, class_index)

# C https://www.thepythoncode.com/article/extract-frames-from-videos-in-python
