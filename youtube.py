import logging
import tempfile

from pytube import YouTube, exceptions


class YouTubeDownloadException(Exception):
    pass


def download(link, temp_dir=None) -> (str, str):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    try:
        video = YouTube(link)
    except exceptions.ExtractError:
        raise YouTubeDownloadException("Extract YouTube video error, invalid link '{}'".format(link))
    logging.info("Processing YouTube video: {} ({}, {} seconds)".format(
        video.player_response.get("videoDetails", {}).get("title"), link, video.length))

    stream_video = video.streams.filter(file_extension="mp4").order_by("resolution").last()

    if not stream_video:
        raise YouTubeDownloadException("Unable to get video stream from {}".format(link))

    logging.info("Get video stream {} ({}), {}, {}fps".format(
        stream_video.mime_type, stream_video.video_codec, stream_video.resolution, stream_video.fps))

    if stream_video.audio_codec is None:
        logging.info("Video stream has no audio, get audio")
        separated_audio = True
        stream_audio = video.streams.get_audio_only()
    else:
        logging.info("Video stream has audio")
        separated_audio = False
        stream_audio = stream_video

    if not stream_video:
        raise YouTubeDownloadException("Unable to get audio stream from {}".format(link))

    logging.info("Get audio stream {} {}".format(stream_audio.mime_type, stream_audio.abr))

    logging.info("Downloading video...")
    video_file = stream_video.download(output_path=temp_dir, filename="./video-" + video.video_id)
    logging.info("Downloading video complete: {}".format(video_file))

    if separated_audio:
        logging.info("Downloading audio...")
        audio_file = stream_audio.download(output_path=temp_dir, filename="./audio-" + video.video_id)
        logging.info("Downloading audio complete: {}".format(audio_file))
    else:
        audio_file = video_file

    return video_file, audio_file
