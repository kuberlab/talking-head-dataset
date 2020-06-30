import logging
import os
import shutil
import tempfile
import threading

import cv2

import audio
import check_frame
import mlboard


class ProcessException(Exception):
    pass


def _out_video_filename(video_filename, frame_idx):
    ext = os.path.splitext(os.path.basename(video_filename))
    return "{}-{}{}".format(ext[0], frame_idx, ext[1])


def process_video(video_file, audio_file=None, output_dir=None, duration=10, ff_frames=0):
    cap = cv2.VideoCapture(video_file)
    frame_idx = 0

    if ff_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff_frames)
        frame_idx = ff_frames-1
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_duration = n_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # from facenet
    ex = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = (
            chr(ex & 0xFF) +
            chr((ex & 0xFF00) >> 8) +
            chr((ex & 0xFF0000) >> 16) +
            chr((ex & 0xFF000000) >> 24)
    )
    fourcc = cv2.VideoWriter_fourcc(*codec)

    if video_duration < duration:
        raise ProcessException("Source video's length {} seconds "
                               "it is shorter than {} seconds".format(video_duration, duration))

    logging.info("Video duration: {} seconds".format(video_duration))

    if output_dir is None:
        output_dir = "./output"

    temp_dir = tempfile.gettempdir()

    video_part_file = None
    final_file = None
    video_part_start = None
    video_writer = None
    previous_frame = None

    fragments = 0

    m = mlboard.get()
    if m is not None:
        m.update_task_info({
            "#process.frame": frame_idx,
        })

    try:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx > 0 and frame_idx % 100 == 0:
                logging.info("processed {} frames".format(frame_idx))
                if m is not None:
                    m.update_task_info({
                        "#process.frame": frame_idx,
                    })

            try:
                check_frame.is_correct(frame, previous_frame)
                if video_part_file is None:
                    video_part_file = os.path.join(temp_dir, _out_video_filename(video_file, frame_idx))
                    final_file = os.path.join(output_dir, _out_video_filename(video_file, frame_idx))
                    if os.path.exists(video_part_file):
                        os.remove(video_part_file)
                    logging.info("Start video fragment {}".format(video_part_file))
                    video_part_start = frame_idx
                    video_writer = cv2.VideoWriter(
                        video_part_file, fourcc, fps,
                        frameSize=(width, height)
                    )

                if video_writer is not None:
                    video_writer.write(frame)

                if frame_idx - video_part_start >= duration * fps:
                    logging.info("Finish video fragment {}".format(video_part_file))
                    safe_run(video_writer.release)
                    video_writer = None

                    if audio_file:
                        try:
                            audio.apply_audio_to(video_part_file, audio_file, video_part_start / fps, frame_idx / fps)
                            logging.info("Audio joined to fragment %s" % video_part_file)
                            if not os.path.isdir(output_dir):
                                os.makedirs(output_dir, 0o755)
                            shutil.move(video_part_file, final_file)
                            logging.info("File stored to %s" % final_file)
                            fragments += 1
                            if m is not None:
                                m.update_task_info({
                                    "#process.fragments": fragments,
                                })
                        except audio.ApplyAudioException as e:
                            logging.error("Join with audio error: %s, file %s removed" % (str(e), video_part_file))

                    video_part_start = None
                    video_part_file = None

            except check_frame.CheckFrameException as e:
                if video_writer is not None:
                    logging.warning("Interrupt video fragment {} on frame {} because {}".format(
                        video_part_file, frame_idx, str(e)))
                    safe_run(video_writer.release)
                    video_writer = None
                    video_part_start = None

                    if os.path.exists(video_part_file):
                        os.remove(video_part_file)
                    video_part_file = None

            frame_idx += 1
            previous_frame = frame

    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt")

    if video_writer is not None:
        logging.warning("Interrupt tailing video fragment {}".format(video_part_file))
        safe_run(video_writer.release)
        if os.path.exists(video_part_file):
            os.remove(video_part_file)

    safe_run(cap.release)


def safe_run(r):
    t = threading.Thread(target=r)
    t.start()
    t.join()
