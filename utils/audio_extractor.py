from moviepy import VideoFileClip

def extract_audio(video_path, output_audio):
    video = VideoFileClip(video_path)

    if video.audio is None:
        print(f"No audio track found: {video_path}")
        return None
    
    video.audio.write_audiofile(output_audio)
    return output_audio

