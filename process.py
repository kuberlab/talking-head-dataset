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


def _out_video_filename(video_filename, frame_idx, duration=None):
    ext = os.path.splitext(os.path.basename(video_filename))
    d = "" if duration is None else "-{}s".format(duration)
    return "{}-{}{}{}".format(ext[0], frame_idx, d, ext[1])


def process_video(video_file, audio_file=None, output_dir=None, duration=None, ff_frames=0, check_each_frame=1):
    cap = cv2.VideoCapture(video_file)
    frame_idx = -1

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

    if duration is not None and video_duration < duration:
        raise ProcessException("Source video's length {} seconds "
                               "it is shorter than {} seconds".format(video_duration, duration))

    logging.info("Video duration: {} seconds".format(video_duration))

    if output_dir is None:
        output_dir = "./output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, 0o755)

    temp_dir = tempfile.gettempdir()

    video_part_file = None
    final_file = None
    video_part_start = None
    video_writer = None
    previous_frame = None

    fragments = 0
    frames_to_write = []
    frames_written = 0

    try:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1

            frames_to_write.append(frame)

            if frame_idx % 100 == 0:
                mlboard.update_task_info({
                    "process.frames_processed": frame_idx,
                    "youtube.frames": n_frames,
                })

            if frame_idx % check_each_frame > 0:
                continue

            finish_recording = False
            interrupt_recording = False
            interrupt_recording_reason = None

            try:
                check_frame.is_correct(frame, previous_frame)
                frame_is_correct = True
                if duration is not None \
                        and video_part_start is not None \
                        and frame_idx - video_part_start >= duration * fps:
                    finish_recording = True

            except check_frame.CheckFrameException as e:
                frame_is_correct = False
                frames_to_write = []
                if duration is not None:
                    interrupt_recording = True
                    interrupt_recording_reason = str(e)
                else:
                    finish_recording = True

            if interrupt_recording:

                if video_writer is not None:
                    logging.warning("Interrupt video fragment {} on frame {} because {}".format(
                        video_part_file, frame_idx, interrupt_recording_reason))
                    safe_run(video_writer.release)
                    video_writer = None
                    video_part_start = None

                    if os.path.exists(video_part_file):
                        os.remove(video_part_file)
                    video_part_file = None

            elif finish_recording:

                if video_writer is not None:

                    frames_written += flush_video(video_writer, frames_to_write)
                    frames_to_write = []

                    fragments = finalize_video(
                        video_writer, video_part_file, audio_file, frames_written,
                        video_part_start, frame_idx, fps, final_file, fragments,
                    )

                    video_writer = None
                    video_part_start = None
                    video_part_file = None

            elif frame_is_correct:

                if video_part_file is None:
                    video_part_start = frame_idx
                    ovf = _out_video_filename(video_file, video_part_start, duration)
                    video_part_file = os.path.join(temp_dir, ovf)
                    final_file = os.path.join(output_dir, ovf)
                    if os.path.exists(video_part_file):
                        os.remove(video_part_file)
                    logging.info("Start video fragment {} from frame {}".format(
                        video_part_file, video_part_start))
                    video_writer = cv2.VideoWriter(
                        video_part_file, fourcc, fps,
                        frameSize=(width, height)
                    )
                    frames_written = 0
                    frames_to_write = []

                frames_written += flush_video(video_writer, frames_to_write)
                frames_to_write = []

            previous_frame = frame

    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt")

    if video_writer is not None:
        if duration is None:
            fragments = finalize_video(
                video_writer, video_part_file, audio_file, frames_written,
                video_part_start, frame_idx, fps, final_file, fragments,
            )
        else:
            logging.warning("Interrupt tailing video fragment {}".format(video_part_file))
            safe_run(video_writer.release)
            if os.path.exists(video_part_file):
                os.remove(video_part_file)

    safe_run(cap.release)

    return fragments


def safe_run(r):
    t = threading.Thread(target=r)
    t.start()
    t.join()


def finalize_video(
        video_writer, video_part_file, audio_file, frames_written,
        video_part_start, frame_idx, fps, final_file, fragments,
):
    logging.info("Finish video fragment {}: {}-{}, frames written {}".format(
        video_part_file, video_part_start, frame_idx, frames_written))
    safe_run(video_writer.release)

    if audio_file:
        try:
            audio.apply_audio_to(video_part_file, audio_file, video_part_start / fps, frame_idx / fps)
            logging.info("Audio joined to fragment %s: %s-%s, %s-%s sec" % (
                video_part_file, video_part_start, frame_idx,
                video_part_start / fps, frame_idx / fps,
            ))
        except audio.ApplyAudioException as e:
            os.remove(video_part_file)
            logging.error("Join with audio error: %s, file %s removed" % (str(e), video_part_file))
            return fragments

    shutil.move(video_part_file, final_file)
    logging.info("File stored to %s" % final_file)

    fragments += 1
    mlboard.update_task_info({
        "process.fragments": fragments,
    })

    return fragments


def flush_video(video_writer, frames_to_write):
    frames_written = 0
    if video_writer is not None:
        for frame in frames_to_write:
            video_writer.write(frame)
            frames_written += 1
    return frames_written


# if __name__ == '__main__':
#     check_frame.initialize("./models")
#     process_video("./video-2AFpAATHXtc.mp4", "./audio-2AFpAATHXtc.mp4", ff_frames=1000, duration=1, check_each_frame=3)
