import argparse
import logging
import os

import check_frame
import process
import youtube
from log import init_logging


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--link', type=str, default=None, help='Process single YouTube video specified in link')
    parser.add_argument('--csv', type=str, default=None, help='Process several YouTube videos, link in CSV')
    parser.add_argument('--duration', type=int, default=10, help='Processed video duration in seconds')
    parser.add_argument('--output-dir', type=str, default=None, help='Output dir')
    parser.add_argument('--face-detect-model', type=str, default=None, help='Face detection model')
    parser.add_argument('--face-detect-threshold', type=float, default=.5, help='Face detect threshold')
    parser.add_argument('--change-scene-threshold', type=float, default=.5, help='Change scene threshold')
    args = parser.parse_args()

    init_logging()

    if not args.link and not args.csv or args.link and args.csv:
        raise RuntimeError("Should be only --youtube or only --csv parameter")

    links = []
    if args.csv:
        with open(args.csv) as f:
            for link in f:
                link = link.strip()
                if link:
                    links.append(link)
    else:
        links.append(args.link)

    logging.info("Going to process {} YouTube links".format(len(links)))

    check_frame.initialize(
        face_detect_model_path=args.face_detect_model,
        face_detect_threshold=args.face_detect_threshold,
        change_scene_threshold=args.change_scene_threshold,
    )

    for link in links:
        video_file, audio_file = youtube.download(link)
        process.process_video(video_file, audio_file, duration=args.duration)
        os.remove(video_file)
        if audio_file != video_file:
            os.remove(audio_file)


if __name__ == '__main__':
    main()