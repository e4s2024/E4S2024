import os
import uuid
import glob
import shutil
from pathlib import Path
from multiprocessing.pool import Pool

import gradio as gr
import torch
from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from Face_swap_with_two_imgs import FaceSwap
from face_swap_video_pipeline import FaceSwapVideoPipeline
from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


def load_image_pipeline():
    pipeline = FaceSwap()
    return pipeline


def load_video_pipeline():
    video_opts = OurSwapFacePipelineOptions().parse()
    pipeline = FaceSwapVideoPipeline(video_opts, use_time_subfolder=False)
    return pipeline


def swap_image(pipeline,
               source_path,
               target_path,
               out_path,
               verbose: bool = False,
               ) -> Image:
    swap_result = pipeline.face_swap_pipeline(
        source=Image.open(source_path).convert("RGB"),
        target=Image.open(target_path).convert("RGB"),
        save_dir=out_path,
        target_mask=None,
        crop_mode="both",
        verbose=verbose,
        optimize_W=False,
        finetune_net=False,
        copy_face=False,
        pose_drive="faceVid2Vid",
        ct_mode="blender",
        warp_target=False,
        face_inpainting=True,
    )
    return swap_result


def swap_image_gr(img1, img2, use_post=False, use_gpen=False, gpu_mode=True):
    root_dir = make_abs_path("./gradio_tmp_data")
    req_id = uuid.uuid1().hex
    data_dir = os.path.join(root_dir, req_id)
    os.makedirs(data_dir, exist_ok=True)

    source_path = os.path.join(data_dir, "source.png")
    target_path = os.path.join(data_dir, "target.png")
    cv2.imwrite(source_path, img1[:, :, ::-1])  # RGB to BGR
    cv2.imwrite(target_path, img2[:, :, ::-1])  # RGB to BGR

    if global_holder.get("image") is None:
        global_holder["image"] = load_image_pipeline()
    result = swap_image(
        global_holder["image"],
        source_path,
        target_path,
        data_dir,
    )

    return result


def swap_video_gr(img1, target_video, frames_cnt, use_crop, use_pti,
                  pti_steps, pti_lr, pti_recolor_lambda, pti_resume_weight_path
                  ):
    root_dir = make_abs_path("./gradio_tmp_data")
    req_id = uuid.uuid1().hex
    data_dir = os.path.join(root_dir, req_id)  # main exp dir
    os.makedirs(data_dir, exist_ok=True)

    source_path = os.path.join(data_dir, "source.png")
    target_path = os.path.join(data_dir, "target.mp4")
    result_path = os.path.join(data_dir, "output.mp4")
    cv2.imwrite(source_path, img1[:, :, ::-1])
    os.system(f"cp {target_video} {target_path}")

    if global_holder.get("video") is None:
        global_holder["video"] = load_video_pipeline()
    video_swap_pipeline = global_holder["video"]
    video_swap_pipeline.out_dir = data_dir
    video_swap_pipeline.e4s_opt.exp_dir = data_dir
    video_swap_pipeline.e4s_opt.max_pti_steps = pti_steps
    video_swap_pipeline.e4s_opt.pti_learning_rate = pti_lr
    video_swap_pipeline.e4s_opt.recolor_lambda = pti_recolor_lambda

    video_swap_pipeline.forward(
        target_path, source_path,
        target_frames_cnt=frames_cnt,
        use_crop=use_crop,
        use_pti=use_pti,
        pti_resume_weight_path=pti_resume_weight_path,
    )
    return result_path


if __name__ == "__main__":
    global_holder = {}

    with gr.Blocks() as demo:
        gr.Markdown("E4S: Fine-Grained Face Swapping via Regional GAN Inversion")

        with gr.Tab("Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="source")
                    image2_input = gr.Image(label="target")
                with gr.Column(scale=2):
                    image_output = gr.Image(label="result")
                    image_button = gr.Button("Run")

        with gr.Tab("Video"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    image3_input = gr.Image(label="source image")
                    video_input = gr.Video(label="target video")
                with gr.Column(scale=3):
                    video_output = gr.Video(label="result")
                    video_button = gr.Button("Run")
            with gr.Accordion("Advanced Video Swapping Options", open=False):
                frames_cnt = gr.Slider(label="Target Max Frames Count (-1: use all frames)",
                                       minimum=-1, maximum=9999, value=-1, step=1)
                use_crop = gr.Checkbox(label='Crop Inputs? (crop and align the faces)', value=True)
                use_pti = gr.Checkbox(label='Enable PTI Tuning (finetuning the generator to obtain more stable video result',
                                      value=True)
            with gr.Accordion("Advanced PTI Tuning Options", open=False):
                pti_steps = gr.Slider(label="Max PTI Steps", minimum=0, maximum=999, value=80, step=1)
                pti_lr = gr.Slider(label="PTI Learning Rate", minimum=0.0, maximum=1e-1, value=1e-3, step=0.0001)
                pti_recolor_lambda = gr.Slider(label="Recolor Lambda", minimum=0.0, maximum=20.0, value=5.0, step=0.1)
                pti_resume_weight_path = gr.Textbox(label="PTI Resume Weight Path",
                                                    value='/Your/Path/To/PTI_G_lr??_iters??.pth')

        image_button.click(
            swap_image_gr,
            inputs=[image1_input, image2_input],
            outputs=image_output,
        )
        video_button.click(
            swap_video_gr,
            inputs=[image3_input, video_input,
                    frames_cnt, use_crop, use_pti,
                    pti_steps, pti_lr, pti_recolor_lambda, pti_resume_weight_path,
                    ],
            outputs=video_output,
        )

    demo.launch(server_name="0.0.0.0", server_port=7868)
