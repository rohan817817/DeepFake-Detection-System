from moviepy import VideoFileClip

def extract_audio(video_path, output_audio):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio)
    return output_audio

