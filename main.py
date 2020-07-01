import argparse
import logging
import os

import check_frame
import mlboard
import process
import youtube
from log import init_logging


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--link', type=str, default=None, help='Process single YouTube video specified in link')
    parser.add_argument('--csv', type=str, default=None, help='Process several YouTube videos, link in CSV')
    parser.add_argument('--duration', type=int, default=None, help='Processed video duration in seconds, not crop if not set')
    parser.add_argument('--check-each-frame', type=int, default=1, help='Check each N frame for correct')
    parser.add_argument('--output-dir', type=str, default=None, help='Output dir')
    parser.add_argument('--models-dir', type=str, default=None, help='Models dir')
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
        models_dir=args.models_dir,
        face_detect_threshold=args.face_detect_threshold,
        change_scene_threshold=args.change_scene_threshold,
    )

    total_fragments = 0

    mlboard.update_task_info({
        "total.count": len(links),
    })

    for n, link in enumerate(links):
        mlboard.update_task_info({
            "youtube.link": link,
            "youtube.downloaded": "false",
        })
        video_file, audio_file = youtube.download(link)
        mlboard.update_task_info({
            "youtube.downloaded": "true",
        })
        fragments = process.process_video(
            video_file, audio_file,
            output_dir=args.output_dir,
            duration=args.duration,
            check_each_frame=args.check_each_frame,
        )
        total_fragments += fragments
        mlboard.update_task_info({
            "total.fragments_done": total_fragments,
            "total.processed": n + 1,
        })
        os.remove(video_file)
        if audio_file != video_file:
            os.remove(audio_file)

    mlboard.update_task_info({
        "youtube.processing": "-",
        "youtube.processed": len(links),
    })


if __name__ == '__main__':
    main()
