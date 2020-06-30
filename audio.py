import os
import shlex
import shutil
import subprocess


class ApplyAudioException(Exception):
    pass


def apply_audio_to(cropped_video_file, source_audio_file, audio_start, audio_end):
    video_ext = os.path.splitext(cropped_video_file)

    # get cropped audio
    audio_file = video_ext[0] + ".aac"
    if os.path.exists(audio_file):
        os.remove(audio_file)
    cmd = 'ffmpeg -y -i "%s" -ss %s -to %s -vn -acodec copy %s' % (source_audio_file, audio_start, audio_end, audio_file)
    code = subprocess.call(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if code != 0:
        raise ApplyAudioException("Failed run %s: exit code %s" % (cmd, code))

    # join cropped audio with video
    video_with_audio_file = video_ext[0] + ".audio" + video_ext[1]
    if os.path.exists(video_with_audio_file):
        os.remove(video_with_audio_file)
    cmd = 'ffmpeg -i "%s" -i "%s" -c:v copy -c:a aac "%s"' % (cropped_video_file, audio_file, video_with_audio_file)
    code = subprocess.call(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if code != 0:
        raise ApplyAudioException("Failed run %s: exit code %s" % (cmd, code))

    os.remove(audio_file)
    shutil.move(video_with_audio_file, cropped_video_file)
