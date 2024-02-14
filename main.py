import whisperx
import gc 
import torch
import freetype
import math
import glob
import random
import subprocess as sp
import gradio as gr
from utils.configservice import CONFIG_DICT
from moviepy.editor import *

def create_wordinfo_arr(title, st, et, total_chr):
    arr = [title]
    new_word_arr = []
    new_start_time_arr = []
    new_end_time_arr = []
    i = 0
    while(i < len(arr)):
        if len(arr[i]) > total_chr :
            word = arr[i][0 : total_chr-1] + '-'      #add - with remaining word
            new_word_arr.append(word)
            arr[i] = arr[i][total_chr-1::]
        else:
            new_word_arr.append(arr[i])
            i += 1
            
    duration  = (et-st)/len(new_word_arr)
    start = st
    
    for i in new_word_arr:
        
        new_start_time_arr.append(start)
        new_end_time_arr.append(start + duration)
        
        start += duration  
        
    return new_word_arr, new_start_time_arr,  new_end_time_arr
    
def create_text_clip(text, color, font_ttf_path, font_size = 45, start = 0, end = 2, x_pos = 0, y_pos = 0):
    stroke_width = font_size/50
    clip = TextClip(text, color = color, fontsize = font_size, font = font_ttf_path, stroke_width = stroke_width,
                    stroke_color = "black")
    clip = clip.set_position((x_pos, y_pos))
    clip = clip.set_start(start)
    clip = clip.set_end(end)
    return clip

def find_max_width(title, font_size, font_ttf_path):
    max_width = 0
    max_height = 0
    for i in title:
        if i == ' ':
            continue
        character = i
        font_ttf_path = font_ttf_path
        font = freetype.Face(font_ttf_path)
        font.set_char_size(int(font_size * 64))
        font.load_char(character)
        width = font.glyph.bitmap.width
        height = font.glyph.bitmap.rows
        max_width = max(max_width,width)
        max_height = max(max_height,height)
    return max_width, max_height

def clip_formation(Str, video, new_word_arr, new_start_time_arr, new_end_time_arr, text_index_arr, clip_arr, 
                   highlight_clip_arr, color_clip_arr, font_ttf_path, font_size, width_of_chr, height_of_chr, rgb_highlight_color):
    video_w = video.size[0]
    y_pos =  video.size[1] * 0.75
    text_clip_test = create_text_clip(Str, "white", font_ttf_path, font_size)
    start_x_pos = (video_w - text_clip_test.size[0]) / 2
    x_pos = start_x_pos

    start_time_index = text_index_arr[0]
    end_time_index = text_index_arr[-1]

    start_time = new_start_time_arr[start_time_index]
    end_time = new_end_time_arr[end_time_index]

    for j in range(0, len(text_index_arr)):
        if j == len(text_index_arr)-1:
            clip_word = new_word_arr[text_index_arr[j]]
        else:
            clip_word = new_word_arr[text_index_arr[j]] + " "

        clip = create_text_clip(clip_word, "white", font_ttf_path, font_size, start_time, end_time, x_pos, y_pos)
        clip_w, clip_h = clip.size

        shadow_shift_x = 1.4 * (width_of_chr / 9) 
        shadow_shift_y = 1.5 * (height_of_chr / 10)
        shadow_x_pos =  x_pos + shadow_shift_x
        shadow_y_pos = y_pos + shadow_shift_y
        shadow_clip = create_text_clip(clip_word, "black", font_ttf_path, font_size, start_time, end_time, shadow_x_pos, shadow_y_pos)
       
        clip_word_start = new_start_time_arr[text_index_arr[j]]
        clip_word_end = new_end_time_arr[text_index_arr[j]]

        highlight_clip = create_text_clip(clip_word, rgb_highlight_color, font_ttf_path, font_size, clip_word_start, clip_word_end, x_pos, y_pos)
        
        
        
        color_clip = ColorClip(size = (clip_w + int(clip_w * 0.01), clip_h - int(clip_h * 0.01)), color = (60, 60, 60))
        color_clip = color_clip.set_start(clip_word_start)
        color_clip = color_clip.set_end(clip_word_end)
        color_clip = color_clip.set_position((x_pos, y_pos))
        color_clip = color_clip.set_opacity(0.3)
        

        x_pos += clip_w

        color_clip_arr.append(color_clip)
        clip_arr.append(shadow_clip)
        clip_arr.append(clip)
        highlight_clip_arr.append(highlight_clip)

def sentence_formation(new_word_arr, new_start_time_arr, new_end_time_arr, total_chr, video, width_of_chr, 
                       height_of_chr, font_ttf_path):
    
    rgb_color_arr = CONFIG_DICT["RGB_COLOR_ARR"]
    rgb_highlight_color = random.choice(rgb_color_arr)
    
    video_w = video.size[0]
    y_pos =  video.size[1]*0.75
    total_size = 0
    Str = ""
    text_index_arr = []
    font_size = 50
    clip_arr = []
    highlight_clip_arr = []
    color_clip_arr = []
    
    for i in range(0, len(new_word_arr)):
        word = new_word_arr[i]
        if total_size + len(word) + 1 < total_chr:
            Str += word + " "  
            text_index_arr.append(i)
            total_size += len(word) + 1
                            
        elif total_size + len(word) + 1 == total_chr:
            Str += word
            text_index_arr.append(i)
            total_size += len(word)
            
        else:  
            
            clip_formation(Str, video, new_word_arr, new_start_time_arr, new_end_time_arr, text_index_arr, clip_arr, 
                           highlight_clip_arr, color_clip_arr, font_ttf_path, font_size, width_of_chr, height_of_chr, rgb_highlight_color)
                
            Str = word
            total_size = len(word)
            text_index_arr = [i]
            
    clip_formation(Str, video, new_word_arr, new_start_time_arr, new_end_time_arr, text_index_arr, clip_arr,
                   highlight_clip_arr, color_clip_arr, font_ttf_path, font_size, width_of_chr, height_of_chr, rgb_highlight_color)



    return clip_arr, highlight_clip_arr, color_clip_arr 

def download_audio(url, audio_file):
    """
    download files for processing
    -----
    Arguments
    url: String - url of file
    -----
    Returns
    None: mp4 and mp3 variants to save
    """
    sp.call(f"wget {url} -O ./{audio_file}", shell = True, stdout = sp.DEVNULL, stderr = sp.DEVNULL)
    print(f'audio saved at {audio_file} done')

def find_total_character_one_frame(TITLE, font_size, font_ttf_path, video_w):
    width_of_chr, height_of_chr = find_max_width(TITLE, font_size, font_ttf_path)
    x_margin = video_w * 0.98
    total_chr_one_line = math.ceil((x_margin) / width_of_chr)
    return width_of_chr, height_of_chr, total_chr_one_line
          
    
def extract_word_and_timestamps(segments):
    start_time_arr = []
    end_time_arr = []
    word_arr = []
    count = 0

    for segment in segments:
        for word_data in segment["words"]:
            if(len(word_data)) < 3:
                if count != 0:
                    print(start_time_arr[-1], end_time_arr[-1])
                    half_duration = (end_time_arr[-1] - start_time_arr[-1]) / 2

                    end_time_arr[-1] -= half_duration
                    print(start_time_arr[-1], end_time_arr[-1])
                    word = word_data["word"]
                    st = end_time_arr[-1]
                    et = st + half_duration

                    start_time_arr.append(st)
                    end_time_arr.append(et)
                    word_arr.append(word)

                else:
                    continue
            else:
                word = word_data["word"]
                st = word_data["start"]
                et = word_data["end"]
                start_time_arr.append(st)
                end_time_arr.append(et)
                word_arr.append(word)
                count += 1
            
    return start_time_arr, end_time_arr, word_arr

def word_fit_one_line(start_time_arr, end_time_arr, word_arr, total_chr_one_line):
    new_start_time_arr = []
    new_end_time_arr = []
    new_word_arr = []
    for i in range(0, len(word_arr)):
        word = word_arr[i]
        st = start_time_arr[i]
        et = end_time_arr[i]
        a, b, c = create_wordinfo_arr(word, st, et, total_chr_one_line)
        new_word_arr += a
        new_start_time_arr += b
        new_end_time_arr += c
    return new_start_time_arr, new_end_time_arr, new_word_arr

def preprocess_the_generated_data(segments, total_chr_one_line):
    start_time_arr, end_time_arr, word_arr = extract_word_and_timestamps(segments)
    new_start_time_arr, new_end_time_arr, new_word_arr = word_fit_one_line(start_time_arr, end_time_arr, word_arr, total_chr_one_line)
    return new_start_time_arr, new_end_time_arr, new_word_arr

def load_model(audio_file):
    device = CONFIG_DICT["DEVICE"]
    batch_size = CONFIG_DICT["BATCH_SIZE"]  # reduce if low on GPU mem
    compute_type = CONFIG_DICT["COMPUTE_TYPE"]  # change to "int8" if low on GPU mem (may reduce accuracy)
    size = CONFIG_DICT["WHISPER_MODEL_TYPE"]
    language = CONFIG_DICT["LANGUAGE"]

    model = whisperx.load_model(size, device, compute_type = compute_type,language = language)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size = batch_size,language = language)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    model_a, metadata = whisperx.load_align_model(language_code = result["language"], device = device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments = CONFIG_DICT["BOOLEAN"][0])

    del model_a
    torch.cuda.empty_cache()
    gc.collect()
    
    return result 

def add_subtitle(video_file, font_size):
    if len(font_size) == 0:
        font_size = CONFIG_DICT["FONT_SIZE"]
    else:
        font_size = eval(font_size)
    url = video_file
    print(f"url {url}")
    output_path = CONFIG_DICT["SUBTITLE_VIDEO_PATH"]
    ttf_file_arr = glob.glob('./ttf_file/*')
    index = random.randint(0, len(ttf_file_arr)-1)
    font_ttf_path = ttf_file_arr[index]
    video = VideoFileClip(url)
    video_w, video_h = video.size
    audio_file = CONFIG_DICT["AUDIOFILEPATH"]
    sp.call(f"rm -rf {audio_file}", shell = True)
    
    audio = video.audio
    audio.write_audiofile(audio_file)
    
    # download_audio(url, audio_file)
    
    TITLE = "W"
    width_of_chr, height_of_chr, total_chr_one_line = find_total_character_one_frame(TITLE, font_size, font_ttf_path, video_w)
    result = load_model(audio_file)
    segments = result["segments"]
    new_start_time_arr, new_end_time_arr, new_word_arr = preprocess_the_generated_data(segments, total_chr_one_line)
    
    clip_arr, highlight_clip_arr, color_clip_arr = sentence_formation(new_word_arr, new_start_time_arr, new_end_time_arr, total_chr_one_line, video,width_of_chr,
                                                      height_of_chr, font_ttf_path)
    
    EFFECTS_ARR = CONFIG_DICT["EFFECTS_ARR"]
    EFFECTS = random.choice(EFFECTS_ARR)
    
    if EFFECTS == "color_clip":
        final_clip = CompositeVideoClip([video, *color_clip_arr, *clip_arr])
    elif EFFECTS == "highlight_clip":
        final_clip = CompositeVideoClip([video, *clip_arr, *highlight_clip_arr])
    else:
        final_clip = CompositeVideoClip([video, *color_clip_arr, *clip_arr, *highlight_clip_arr])
    
    
    sp.call(f"rm -rf {output_path}", shell = True)
    final_clip.write_videofile(output_path)
    
    return output_path

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Subtitle Video Generation
        
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                font_size = gr.Text(label = "Font Size", lines = 2, interactive=True)
            with gr.Row():
                input_video = gr.PlayableVideo(show_label = "Upload your video file here", sources = "upload", 
                                               interactive=True)
            with gr.Row():
                submit = gr.Button("Upload")
        with gr.Column():
            file_output = gr.Video(label = "Subtitle video(mp4)")
        
    submit.click(add_subtitle, inputs = [input_video, font_size],outputs = file_output)


if __name__ == "__main__":
    demo.queue(max_size = 10)
    demo.launch(share = True,debug = True) 