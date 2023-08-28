import os
import warnings
import argparse
from PIL import ImageDraw

import gradio as gr

DEBUG_MODE = False
if not DEBUG_MODE:
    import torch
    import transformers
    from transformers import AutoTokenizer
    from huggingface_hub import login, snapshot_download
    from utils.data_preprocess import build_transform, RAHuskyCaptionCollator
    from custom_models.all_seeing_model import AllSeeingModelForCaption
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = './assets/All-Seeing-Model-FT-V0'

    warnings.warn(f'torch version: {torch.__version__}')
    warnings.warn(f'transformers version: {transformers.__version__}')

    # if not os.path.exists(MODEL_PATH):
    print('begin to download model ckpt')
    login(token=os.environ['HF_TOKEN'])
    snapshot_download(repo_id='Weiyun1025/All-Seeing-Model-FT-V0', cache_dir='./cache', local_dir=MODEL_PATH, resume_download=True)

    warnings.warn(f'Files in {MODEL_PATH}: {os.listdir(MODEL_PATH)}')
    model = AllSeeingModelForCaption.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    image_processor = build_transform(model.config.vision_config.image_size)
    collator = RAHuskyCaptionCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_queries=model.config.num_query_tokens,
        input_size=model.config.vision_config.image_size,
        train_mode=False,
    )

    generation_config = dict(
        do_sample=False,
        temperature=0.7,
        max_new_tokens=512,
        num_beams=1,
    )


TEXT_PLACEHOLDER_BEFORE_UPLOAD = 'Please upload your image first'
TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT = 'Please select two points on the image to determine the position of the box'
TEXT_PLACEHOLDER_AFTER_UPLOAD = 'Type and press Enter'

POINT_RADIUS = 16
POINT_COLOR = (255, 0, 0)

BBOX_WIDTH = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--name', default='')
    args = parser.parse_args()
    return args


def gradio_reset(user_state: dict):
    user_state = {}

    return (
        gr.update(value=None, interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=None),
        gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD),
        user_state,
    )


def point_reset(user_state: dict):
    user_state.pop('points', None)
    user_state.pop('boxes', None)
    user_state.pop('boxes_mask', None)
    user_state['image'] = user_state['original_image'].copy()
    return (
        user_state['original_image'],
        gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        user_state,
    )


def text_reset(user_state: dict):
    user_state.pop('input_ids', None)
    user_state.pop('attention_mask', None)

    interactive = len(user_state['points']) == 2 if 'points' in user_state else False
    return (
        gr.update(value=None),
        gr.update(interactive=interactive, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD if interactive else TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        user_state,
    )


def upload_img(image, user_state):
    if image is None:
        return (
            None,
            None,
            None,
            gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD),
            user_state,
        )

    user_state['image'] = image.copy()
    user_state['original_image'] = image.copy()
    if not DEBUG_MODE:
        user_state['pixel_values'] = image_processor(image)

    return (
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        user_state,
    )


def upload_point(user_state: dict, evt: gr.SelectData):
    if 'image' not in user_state:
        raise gr.Error('Please click Upload & Start Chat button before pointing at the image')

    image = user_state['image']
    new_point = (evt.index[0], evt.index[1])

    if len(user_state.get('points', [])) >= 2:
        return (
            image,
            gr.update(interactive=True, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD),
            user_state,
        )
    if 'points' in user_state:
        user_state['points'].append(new_point)
        assert len(user_state['points']) == 2

        point1, point2 = user_state['points']
        x1 = min(point1[0], point2[0])
        y1 = min(point1[1], point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        bbox = (x1, y1, x2, y2)
        user_state['bbox'] = bbox
        user_state['image'] = user_state['original_image'].copy()

        image = user_state['image']
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, width=BBOX_WIDTH, outline=POINT_COLOR)

    else:
        user_state['points'] = [new_point]

        x, y = new_point
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS), fill=POINT_COLOR)

    interactive = len(user_state['points']) == 2
    return (
        image,
        gr.update(interactive=interactive, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD if interactive else TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        user_state,
    )


def ask_and_answer(chatbot: list, text_input: str, user_state: dict):
    if 'bbox' in user_state:
        bbox = user_state['bbox']
    else:
        raise gr.Error('Please select 2 points.')

    if DEBUG_MODE:
        outputs = 'hello world'
    else:
        inputs = {
            'query': text_input,
            'bbox': bbox,
            'image': user_state['original_image'],
            'pixel_values': user_state['pixel_values'],

            # useless
            'image_id': -1,
            'label': '',
        }
        inputs = collator([inputs])
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['boxes'][0] = inputs['boxes'][0].to(device)
        inputs['boxes_mask'] = inputs['boxes_mask'].to(device)
        inputs.pop('labels', None)

        outputs = model.generate(**inputs, **generation_config)['sequences']
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # chatbot.append([text_input, outputs])
    chatbot = [[text_input, outputs]]
    return (
        chatbot,
        None,
        user_state,
    )


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div align='center'>
            <div style="display: inline-block;">
                <h1 style="">The All-Seeing-Model (ASM) Demo</h>
            </div>
            <div style="display: inline-block; vertical-align: bottom;">
                <img width='60' src="/file=./assets/logo.png">
            </div>
            <div style='display:flex; align-items: center; justify-content: center; gap: 0.25rem; '>
                <a href='https://github.com/OpenGVLab/all-seeing'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
                <a href='https://arxiv.org/abs/2308.01907'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
            </div>
        </div>
        """,
    )

    user_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil", height=450)
            with gr.Row():
                clear_points = gr.Button("Clear points", interactive=False)
                clear_text = gr.Button("Clear text", interactive=False)
            clear_all = gr.Button("Restart", variant='primary')

        with gr.Column():
            chatbot = gr.Chatbot(label='All-Seeing-Model', height=460)
            text_input = gr.Textbox(label='User', interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD)

    image.select(upload_point, [user_state], [image, text_input, user_state])
    image.upload(upload_img, [image, user_state], [image, clear_text, clear_points, text_input, user_state])

    text_input.submit(ask_and_answer, [chatbot, text_input, user_state], [chatbot, text_input, user_state])

    clear_points.click(point_reset, [user_state], [image, text_input, user_state], queue=False)
    clear_text.click(text_reset, [user_state], [chatbot, text_input, user_state], queue=False)
    clear_all.click(gradio_reset, [user_state], [image, clear_text, clear_points, chatbot, text_input, user_state], queue=False)

    gr.HTML(
        """
        <div align='left' style='font-size: large'>
            <h2 style='font-size: x-large'> User Manual: </h2>
            <ol>
                <li> Upload your image.  </li>
                <li> Select two points on the image to determine the position of the box. </li>
                <li> Begin to chat! </li>
            </ol>
            NOTE: if you want to upload another image, please click the Restart button.
        </div>
        """
    )

server_port = 10010
for i in range(10010, 10100):
    cmd = f'netstat -aon|grep {i}'
    with os.popen(cmd, 'r') as file:
        if '' == file.read():
            server_port = i
            break

print('server_port:', server_port)
# demo.queue().launch(server_name="0.0.0.0", server_port=server_port)
demo.queue().launch()
