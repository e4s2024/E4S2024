import argparse
import copy
import os
import datetime
import glob
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions
from utils import torch_utils
from models.networks import Net3
from datasets.dataset import get_transforms, TO_TENSOR, NORMALIZE
from gradio_utils.face_swapping import (
    read_video_as_frames,
    save_frames_as_video,
    crop_and_align_face,
    logical_or_reduce,
    create_masks,
    get_facial_mask_from_seg19,
    get_edge,
    blending_two_images_with_mask,
    SoftErosion,
)
from swap_face_fine.face_vid2vid.drive_demo import init_facevid2vid_pretrained_model, drive_source_demo
from swap_face_fine.face_parsing.face_parsing_demo import (
    init_faceParsing_pretrained_model,
    faceParsing_demo,
    vis_parsing_maps
)
from swap_face_fine.gpen.gpen_demo import GPENInfer
from swap_face_fine.inference_codeformer import CodeFormerInfer
from swap_face_fine.realesr.image_infer import RealESRBatchInfer
from training.video_swap_ft_coach import VideoSwapPTICoach
from swap_face_fine.swap_face_mask import swap_head_mask_hole_first, swap_comp_style_vector
from swap_face_fine.multi_band_blending import blending
from swap_face_fine.Blender.inference import BlenderInfer


class FaceSwapVideoPipeline(object):
    def __init__(self,
                 e4s_opt: argparse.Namespace,
                 use_time_subfolder: bool = True,
                 ):
        self.exp_root = e4s_opt.exp_dir
        self.out_dir = e4s_opt.exp_dir
        self.pti_save_fn = None
        self.use_time_subfolder = use_time_subfolder

        self.e4s_opt = e4s_opt
        self.e4s_model = None
        self.device = e4s_opt.device

        # models are lazy loaded
        self.face_reenact_model = {}
        self.face_parsing_model = {}
        self.face_enhance_model = {}
        self.face_recolor_model = {}
        self.mask_softer_model = {}

        self.num_seg_cls = 12  # fixed

    def forward(self,
                target_video_path: str,
                source_image_path: str,
                result_video_fn: str = "output.mp4",
                use_crop: bool = True,
                target_frames_cnt: int = -1,
                use_pti: bool = True,
                pti_resume_weight_path: str = "./video_outputs/finetuned_G_lr0.001000_iters80.pth",
                ):
        """
        @param target_video_path:
        @param source_image_path:
        @param result_video_fn:
        @param use_crop:
        @param target_frames_cnt:
        @param use_pti:
        @param pti_resume_weight_path: if opt.max_pti_steps == 0, the pipeline will use this pre-trained weight file
        """
        # 0. update time, used as output directory
        self._update_out_dir()

        # 1. prepare input target and source
        target_paths, source_paths = self._prepare_inputs(
            target_video_path, source_image_path, target_frames_cnt=target_frames_cnt,
        )
        target_frames_cnt = len(target_paths)

        # 2. crop and align
        crop_results = self._process_crop_align(
            target_paths, source_paths, use_crop=use_crop,
        )
        T = crop_results["targets_crop"]
        S = crop_results["source_crop"]
        T_ori = crop_results["targets_ori"]
        T_inv_trans = crop_results["targets_inv_trans"]

        # 3. face reenactment
        drivens, drivens_recolor = self._process_face_reenact(
            T, S, use_recolor=True
        )

        # 4. face enhancement
        # drivens = self._process_face_enhance(
        #     drivens, model_name="codeformer",
        # )
        # if drivens_recolor[0] is not None:
        #     drivens_recolor = self._process_face_enhance(
        #         drivens_recolor, model_name="codeformer", save_prefix="D_recolor_"
        #     )

        # 5. face parsing
        parsing_results = self._process_face_parsing(
            T, S, drivens
        )
        T_mask = parsing_results["targets_mask"]
        S_mask = parsing_results["source_mask"]
        D_mask = parsing_results["drivens_mask"]

        # 6. extract initial style vectors
        self._process_extract_init_style_vectors(
            drivens, T, drivens_mask=D_mask, targets_mask=T_mask
        )

        # 7. PTI tuning
        if use_pti:
            self._process_pti_tuning(
                pti_resume_weight_path, target_frames_cnt=target_frames_cnt,
            )

        # 8. face swapping
        swap_results = self._process_face_swapping(
            target_frames_cnt, T_inv_trans, T_ori,
        )
        swaps_face = swap_results["swaps_face"]  # each is: PIL.Image
        swaps_mask = swap_results["swaps_mask"]  # each is: np.ndarray(512,512), in {0,...,9}

        # 9. prepare outputs
        self._prepare_outputs(
            result_video_fn, target_video_path
        )

    def _update_out_dir(self):
        if not self.use_time_subfolder:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = os.path.join(self.exp_root, now)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.e4s_opt.exp_dir = out_dir
        print(f"[FaceSwapVideoPipeline] out directory changed to: {self.out_dir}")
        return

    def _prepare_inputs(self, target_video_path: str, source_image_path: str,
                        target_frames_cnt: int = 120,
                        ):
        in_target_frames_folder = os.path.join(self.out_dir, "in_frames/")

        t_frames, t_paths = read_video_as_frames(target_video_path, in_target_frames_folder)
        t_frames = t_frames[:target_frames_cnt]
        t_paths = t_paths[:target_frames_cnt]  # many targets

        s_paths = [source_image_path]  # only 1 source

        # save inputs
        target_save_path = os.path.join(self.out_dir, "target.mp4")
        source_save_path = os.path.join(self.out_dir, "source.png")
        os.system(f"cp {target_video_path} {target_save_path}")
        os.system(f"cp {source_image_path} {source_save_path}")
        return t_paths, s_paths

    def _process_crop_align(self, t_paths: list, s_paths: list, use_crop: bool):
        if use_crop:
            target_files = [(os.path.basename(f).split('.')[0], f) for f in t_paths]
            source_files = [(os.path.basename(f).split('.')[0], f) for f in s_paths]

            target_crops, target_orig_images, target_quads, target_inv_transforms = crop_and_align_face(
                target_files, image_size=1024, scale=1.0, center_sigma=1.0, xy_sigma=3.0, use_fa=False
            )
            T = [crop.convert("RGB") for crop in target_crops]

            source_crops, source_orig_images, source_quads, source_inv_transforms = crop_and_align_face(
                source_files, image_size=1024, scale=1.0, center_sigma=0, xy_sigma=0, use_fa=False
            )
            S = source_crops[0].convert("RGB")

            T_ori = target_orig_images
            T_inv_trans = target_inv_transforms

        else:
            T = [Image.open(t).convert("RGB").resize((1024, 1024)) for t in t_paths]
            S = Image.open(s_paths[0]).convert("RGB").resize((1024, 1024))
            T_ori = T
            T_inv_trans = None

        return {
            "targets_crop": T,
            "source_crop": S,
            "targets_ori": T_ori,
            "targets_inv_trans": T_inv_trans
        }

    def _process_face_parsing(self, targets, source, drivens):
        self._load_face_parsing_model()
        face_parsing_model = self.face_parsing_model["model"]

        print("[FaceSwapVideoPipeline] face parsing...")
        T_mask = [faceParsing_demo(face_parsing_model, frm, convert_to_seg12=True) for frm in targets]  # 12
        S_mask = faceParsing_demo(face_parsing_model, source, convert_to_seg12=True)
        D_mask = [faceParsing_demo(face_parsing_model, d, convert_to_seg12=True) for d in drivens]

        save_img_dir = os.path.join(self.out_dir, "imgs")
        save_mask_dir = os.path.join(self.out_dir, "mask")
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)
        for i in range(len(T_mask)):
            targets[i].save(os.path.join(save_img_dir, "T_%04d.png" % i))
            Image.fromarray(T_mask[i]).save(os.path.join(save_mask_dir, "T_mask_%04d.png" % i))
            Image.fromarray(D_mask[i]).save(os.path.join(save_mask_dir, "D_mask_%04d.png" % i))
            D_mask_vis = vis_parsing_maps(drivens[i], D_mask[i])
            Image.fromarray(D_mask_vis).save(os.path.join(save_mask_dir, "D_mask_vis_%04d.png" % i))
        Image.fromarray(S_mask).save(os.path.join(save_mask_dir, "S_mask.png"))

        return {
            "targets_mask": T_mask,
            "source_mask": S_mask,
            "drivens_mask": D_mask,
        }

    def _process_face_reenact(self, targets, source,
                              use_recolor: bool = False):
        self._load_face_reenact_model()
        generator = self.face_reenact_model["generator"]
        kp_detector = self.face_reenact_model["kp_detector"]
        he_estimator = self.face_reenact_model["he_estimator"]
        estimate_jacobian = self.face_reenact_model["estimate_jacobian"]

        print("[FaceSwapVideoPipeline] face reenacting...")
        targets_256 = [resize(np.array(im) / 255.0, (256, 256)) for im in targets]
        source_256 = resize(np.array(source) / 255.0, (256, 256))

        predictions = drive_source_demo(source_256, targets_256,
                                        generator, kp_detector, he_estimator, estimate_jacobian)
        predictions = [(pred * 255).astype(np.uint8) for pred in predictions]  # RGB

        predictions = self._process_face_enhance(
            predictions, model_name="gpen",
        )  # fixed as gpen

        ''' color transfer before pasting back '''
        predictions_recolor = [None] * len(predictions)
        if use_recolor:
            predictions_recolor = [None] * len(predictions)
            face_parsing_model = self._load_face_parsing_model()["model"]
            face_enhance_model = self._load_face_enhance_model("codeformer")["codeformer"]
            recolor_save_dir = os.path.join(self.out_dir, "recolor_before_rgi")
            os.makedirs(recolor_save_dir, exist_ok=True)
            face_recolor_model = self._load_face_recolor_model()["model"]
            mask_softer_model = self._load_mask_softer()["model"]

            for i in range(len(predictions)):
                # swapped_face_image = Image.fromarray(predictions[i])
                swapped_face_image = predictions[i]
                swapped_face_image.save(os.path.join(recolor_save_dir, "recolor_input_%04d.png" % i))
                T = targets[i].resize(swapped_face_image.size)

                swap_mask_19 = faceParsing_demo(face_parsing_model, swapped_face_image, convert_to_seg12=False)
                target_mask_19 = faceParsing_demo(face_parsing_model, T, convert_to_seg12=False)
                recolor: Image = face_recolor_model.infer_image(
                    swapped_face_image, T,
                    Image.fromarray(swap_mask_19), Image.fromarray(target_mask_19)
                )
                recolor.save(os.path.join(recolor_save_dir, "recolor_gen_%04d.png" % i))
                recolor = recolor.resize(swapped_face_image.size)
                recolor = face_enhance_model.infer_image(recolor)  # no need to super-res?
                recolor = recolor.resize((512, 512)).resize(recolor.size)  # resize down to avoid too high-res in video
                recolor.save(os.path.join(recolor_save_dir, "gen_enhance_%04d.png" % i))

                # only copy low-frequency parts
                # blending_mask = get_facial_mask_from_seg19(
                #     torch.LongTensor(swap_mask_19[None, None, :, :]),
                #     target_size=recolor.size, edge_softer=mask_softer_model, is_seg19=True
                # )
                # edge = get_edge(swapped_face_image)
                # edge = np.array(edge).astype(np.float32) / 255.
                # blending_mask = (blending_mask - edge).clip(0., 1.)
                # Image.fromarray((blending_mask.squeeze() * 255.).astype(np.uint8)).save(
                #     os.path.join(recolor_save_dir, "blend_mask_%04d.png" % i)
                # )
                # recolor = blending_two_images_with_mask(
                #     swapped_face_image, recolor, up_ratio=0.95, up_mask=blending_mask.copy()
                # )
                # recolor.save(os.path.join(recolor_save_dir, "recolor_blend_%04d.png" % i))

                predictions_recolor[i] = np.array(recolor)  # RGB

            imgs_save_dir = os.path.join(self.out_dir, "imgs")
            os.makedirs(imgs_save_dir, exist_ok=True)
            for i in range(len(predictions_recolor)):
                Image.fromarray(predictions_recolor[i]).save(
                    os.path.join(imgs_save_dir, "%s%04d.png" % ("D_recolor_", i)))
        ''' end '''

        self._free_face_reenact_model()
        return predictions, predictions_recolor

    def _process_face_enhance(self, lq_images: list,
                              model_name: str = "gpen",
                              save_prefix: str = "D_",
                              ):
        self._load_face_enhance_model(model_name)
        enhance_model = self.face_enhance_model[model_name]
        print("[FaceSwapVideoPipeline] face enhancing...")
        hq_images = [enhance_model.infer_image(Image.fromarray(lq)) for lq in lq_images]

        save_dir = os.path.join(self.out_dir, "imgs")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(hq_images)):
            hq_images[i].save(os.path.join(save_dir, "%s%04d.png" % (save_prefix, i)))
        return hq_images

    @torch.no_grad()
    def _process_extract_init_style_vectors(self, drivens, targets, drivens_mask, targets_mask):
        save_dir = os.path.join(self.out_dir, "styleVec")
        os.makedirs(save_dir, exist_ok=True)
        net = self._load_e4s_model()

        for i, (d, t) in enumerate(zip(drivens, targets)):
            driven = transforms.Compose([TO_TENSOR, NORMALIZE])(d)
            driven = driven.to(self.device).float().unsqueeze(0)
            driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(drivens_mask[i]))
            driven_mask = (driven_mask * 255).long().to(self.device).unsqueeze(0)
            driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls=self.num_seg_cls)

            target = transforms.Compose([TO_TENSOR, NORMALIZE])(t)
            target = target.to(self.device).float().unsqueeze(0)
            target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(targets_mask[i]))
            target_mask = (target_mask * 255).long().to(self.device).unsqueeze(0)
            target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls=self.num_seg_cls)

            driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot)
            torch.save(driven_style_vector, os.path.join(save_dir, "D_style_vec_%04d.pt" % i))

            target_style_vector, _ = net.get_style_vectors(target, target_onehot)
            torch.save(target_style_vector, os.path.join(save_dir, "T_style_vec_%04d.pt" % i))

    def _process_pti_tuning(self,
                            pti_resume_weight_path: str = "./video_outputs/finetuned_G_lr0.001000_iters80.pth",
                            target_frames_cnt: int = -1,
                            ):
        opts = self.e4s_opt
        pti_steps = opts.max_pti_steps

        if pti_steps > 0:  # needs PTI
            finetune_coach = VideoSwapPTICoach(
                opts,
                e4s_net=self._load_e4s_model(),
                num_targets=target_frames_cnt, erode=True)
            finetune_coach.train()

            # save tuned weights
            save_dict = finetune_coach.get_save_dict()
            self.pti_save_fn = "PTI_G_lr%f_iters%d.pth" % (opts.pti_learning_rate, pti_steps)
            save_path = os.path.join(opts.exp_dir, self.pti_save_fn)
            torch.save(save_dict, save_path)

            net = finetune_coach.net
            print(f"[FaceSwapVideoPipeline] PTI training finished, model saved to: {save_path}")

        else:  # load PTI tuned weights
            if not os.path.exists(pti_resume_weight_path):
                print(f"Tuned PTI weights not found! Load the latest tuned PTI weight: ({self.pti_save_fn})")
                pti_resume_weight_path = os.path.join(opts.exp_dir, self.pti_save_fn)
            net = self._load_e4s_model()
            pti_tuned_weights = torch.load(pti_resume_weight_path)
            net.latent_avg = pti_tuned_weights['latent_avg'].to(opts.device)
            net.load_state_dict(torch_utils.remove_module_prefix(pti_tuned_weights["state_dict"], prefix="module."))
            print(f"[FaceSwapVideoPipeline] Load pre-trained PTI weights from: {pti_resume_weight_path}")

        self.e4s_model = net
        return

    def _process_face_swapping(self,
                               target_frames_cnt: int = 120,
                               targets_inv_trans: list = None,
                               targets_ori: list = None,
                               ):
        out_dir = self.out_dir
        opts = self.e4s_opt
        swap_save_dir = os.path.join(self.out_dir, "swapped")  # paste back
        os.makedirs(swap_save_dir, exist_ok=True)

        net = self.e4s_model

        swaps_face = []
        swaps_mask = []
        for i in tqdm(range(target_frames_cnt), desc="Swapping"):
            D = Image.open(os.path.join(opts.exp_dir, "imgs", "D_%04d.png" % i)).convert(
                "RGB").resize((1024, 1024))
            T = Image.open(os.path.join(opts.exp_dir, "imgs", "T_%04d.png" % i)).convert(
                "RGB").resize((1024, 1024))

            D_mask = np.array(
                Image.open(os.path.join(opts.exp_dir, "mask", "D_mask_%04d.png" % i)))

            T_mask = np.array(
                Image.open(os.path.join(opts.exp_dir, "mask", "T_mask_%04d.png" % i)))
            # T_mask, _ = erode_mask(T_mask, T, radius=1, verbose=True)

            # swapped_msk, hole_map, eyebrows_line = swap_head_mask_revisit(D_mask, T_mask)  # 换头
            swapped_msk, hole_mask, hole_map, eye_line = swap_head_mask_hole_first(D_mask, T_mask)
            cv2.imwrite(os.path.join(out_dir, "mask", "swappedMask_%04d.png" % i),
                        swapped_msk)
            swaps_mask.append(swapped_msk)

            swappped_one_hot = torch_utils.labelMap2OneHot(
                torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12).to(opts.device)
            # torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(opts.exp_dir,"swappedMaskVis.png"))

            # 保留 target_style_vectors 的background, hair, 其余的全用 driven_style_vectors
            D_style_vector = torch.load(
                os.path.join(out_dir, "styleVec", "D_style_vec_%04d.pt" % i)).to(
                opts.device).float()
            T_style_vector = torch.load(
                os.path.join(out_dir, "styleVec", "T_style_vec_%04d.pt" % i)).to(
                opts.device).float()
            comp_indices = set(range(opts.num_seg_cls)) - {0, 4, 11}  # 9 mouth
            swapped_style_vectors = swap_comp_style_vector(T_style_vector, D_style_vector, list(comp_indices),
                                                           belowFace_interpolation=False)

            with torch.no_grad():
                swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
                swapped_face, _, structure_feats = net.gen_img(torch.zeros(1, 512, 32, 32).to(opts.device),
                                                               swapped_style_codes, swappped_one_hot)  # in [-1,1]

                ''' save images '''
                swapped_face_image = torch_utils.tensor2im(swapped_face)
                swapped_face_image = swapped_face_image.resize((512, 512)).resize((1024, 1024))
                swapped_m = transforms.Compose([TO_TENSOR])(swapped_msk)
                swapped_m = (swapped_m * 255).long().to(opts.device).unsqueeze(0)
                swapped_face_image.save(os.path.join(swap_save_dir, "pti_gen_%04d.png" % i))

                swaps_face.append(swapped_face_image)

                outer_dilation = 5  # 这个值可以调节
                mask_bg = logical_or_reduce(*[swapped_m == clz for clz in [0, 11, 7, 4,
                                                                           8]])  # 4,8,7  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分, 11 earings 4 hair 8 neck 7 ear
                is_foreground = torch.logical_not(mask_bg)
                hole_index = hole_mask[None][None]
                is_foreground[hole_index[None]] = True
                foreground_mask = is_foreground.float()

                # foreground_mask = dilation(foreground_mask, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=foreground_mask.device), engine='convolution')
                content_mask, border_mask, full_mask = create_masks(foreground_mask, operation='expansion', radius=5)

                # past back
                content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
                content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
                border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
                border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
                border_mask = np.repeat(border_mask, 3, axis=-1)

                swapped_and_pasted = swapped_face_image * content_mask + T * (1 - content_mask)
                swapped_and_pasted = Image.fromarray(blending(np.array(T), swapped_and_pasted, mask=border_mask))
                pasted_image = swapped_and_pasted

                if targets_inv_trans is None:  # op1. directly paste
                    pasted_image.save(os.path.join(swap_save_dir, "swap_face_%04d.png"%i))
                else:  # op2. paste back
                    swapped_and_pasted = swapped_and_pasted.convert('RGBA')
                    pasted_image = targets_ori[i].convert('RGBA')
                    swapped_and_pasted.putalpha(255)
                    projected = swapped_and_pasted.transform(targets_ori[i].size, Image.PERSPECTIVE,
                                                             targets_inv_trans[i],
                                                             Image.BILINEAR)
                    pasted_image.alpha_composite(projected)
                    pasted_image.save(os.path.join(swap_save_dir, "swap_face_%04d.png" % i))

        return {
            "swaps_face": swaps_face,
            "swaps_mask": swaps_mask,
        }

    def _prepare_outputs(self, result_video_fn: str, target_video_path: str):
        out_dir = self.out_dir
        swap_save_dir = os.path.join(self.out_dir, "swapped")

        save_frames_as_video(
            frames=swap_save_dir,
            video_save_dir=out_dir,
            video_save_fn=result_video_fn,
            frame_template="swap_face_%04d.png",
            audio_from=target_video_path,
            delete_tmp_frames=False,
        )

        return

    def _load_face_reenact_model(self):
        if len(self.face_reenact_model.items()) > 0:
            return self.face_reenact_model
        face_vid2vid_cfg = "./pretrained/faceVid2Vid/vox-256.yaml"
        face_vid2vid_ckpt = "./pretrained/faceVid2Vid/00000189-checkpoint.pth.tar"
        generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(
            face_vid2vid_cfg,
            face_vid2vid_ckpt
        )
        self.face_reenact_model["generator"] = generator
        self.face_reenact_model["kp_detector"] = kp_detector
        self.face_reenact_model["he_estimator"] = he_estimator
        self.face_reenact_model["estimate_jacobian"] = estimate_jacobian
        print("[FaceSwapVideoPipeline] Face reenactment model loaded.")
        return self.face_reenact_model

    def _free_face_reenact_model(self):
        keys = self.face_reenact_model.keys()
        for k in tuple(keys):
            del self.face_reenact_model[k]
        print("[FaceSwapVideoPipeline] Face reenactment model free memory.")
        self.face_reenact_model = {}

    def _load_face_parsing_model(self):
        if len(self.face_parsing_model.items()) > 0:
            return self.face_parsing_model
        face_parsing_ckpt = "./pretrained/faceseg/79999_iter.pth"
        self.face_parsing_model["model"] = init_faceParsing_pretrained_model(face_parsing_ckpt)
        print("[FaceSwapVideoPipeline] Face parsing model loaded.")
        return self.face_parsing_model

    def _load_face_enhance_model(self, name: str = "gpen"):
        if self.face_enhance_model.get(name) is not None:
            return self.face_enhance_model
        if name == "gpen":
            self.face_enhance_model["gpen"] = GPENInfer()
        elif name == "codeformer":
            self.face_enhance_model["codeformer"] = CodeFormerInfer()
        elif name == "realesr":
            self.face_enhance_model["realesr"] = RealESRBatchInfer()
        else:
            raise KeyError(f"Not supported face enhancement model: {name}")
        print(f"[FaceSwapVideoPipeline] Face enhancing model loaded: {name}")
        return self.face_enhance_model

    def _load_e4s_model(self):
        if not self.e4s_model is None:
            return self.e4s_model
        opts = self.e4s_opt
        net = Net3(opts)
        net = net.to(opts.device)
        save_dict = torch.load(opts.checkpoint_path)
        net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
        net.latent_avg = save_dict['latent_avg'].to(opts.device)
        print("[FaceSwapVideoPipeline] Load E4S pre-trained model success!")
        self.e4s_model = net
        return self.e4s_model

    def _load_face_recolor_model(self):
        if len(self.face_recolor_model.items()) > 0:
            return self.face_recolor_model
        blender = BlenderInfer()
        self.face_recolor_model["model"] = blender
        print("[FaceSwapVideoPipeline] Face recolor model loaded.")
        return self.face_recolor_model

    def _load_mask_softer(self):
        if len(self.mask_softer_model.items()) > 0:
            return self.mask_softer_model
        softer = SoftErosion()
        self.mask_softer_model["model"] = softer
        print("[FaceSwapVideoPipeline] Mask softer model loaded.")
        return self.mask_softer_model


if __name__ == "__main__":
    test_opts = OurSwapFacePipelineOptions().parse()
    test_opts.max_pti_steps = 80
    test_opts.recolor_lambda = 5.
    test_opts.exp_dir = './video_outputs/'
    os.makedirs(test_opts.exp_dir, exist_ok=True)

    pipeline = FaceSwapVideoPipeline(
        test_opts,
    )

    # test_target_video_path = "/home/yuange/Documents/E4S_v2/figs/video_infer/target_NKXqc5vAN8_0.mp4"
    test_target_video_path = "/home/yuange/datasets/STIT/datasets/jim.mp4"
    test_source_image_path = "/home/yuange/Documents/E4S_v2/sota_method_results/infoswap/source/00012.jpg"

    pipeline.forward(
        test_target_video_path, test_source_image_path,
        use_crop=True,
        target_frames_cnt=-1,
        pti_resume_weight_path="./video_outputs/2023-10-14T01-43-31/PTI_G_lr0.001000_iters80.pth",
    )
