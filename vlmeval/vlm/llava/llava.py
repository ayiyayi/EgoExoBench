import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="liuhaotian/llava_v1.5_7b", **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as err:
            logging.critical(
                "Please install llava from https://github.com/haotian-liu/LLaVA"
            )
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        if model_path == "Lin-Chen/ShareGPT4V-7B":
            model_name = "llava-v1.5-7b"
        elif model_path == "Lin-Chen/ShareGPT4V-13B":
            model_name = "llava-v1.5-13b"
        else:
            model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="cpu",
                )
            )
        except Exception as err:
            if "ShareGPT4V" in model_path:
                import llava

                logging.critical(
                    "Please manually remove the encoder type check in "
                    f"{llava.__path__[0]}/model/multimodal_encoder/builder.py "
                    "Line 8 to use the ShareGPT4V model. "
                )
            else:
                logging.critical("Unknown error when loading LLaVA model.")
            raise err

        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            content, images_sub = self.concat_tilist(utter["content"])
            prompt += content
            images.extend(images_sub)
            prompt += " " if utter["role"] == "user" else self.stop_str
        assert message[-1]["role"] == "user", message
        prompt += "ASSISTANT: "

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, self.image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-v1.6-vicuna-7b-hf", **kwargs):
        import transformers
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            AutoProcessor,
            LlavaForConditionalGeneration,
        )

        self.model_path = model_path
        if "34b" in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_path, use_fast=False
            )
        elif "interleave" in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        flash_attn_flag = False
        try:
            import flash_attn

            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
        else:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def apply_prompt_template(self, prompt):
        model_path = self.model_path.lower()
        if "mistral" in model_path:
            template = "[INST] PLACEHOLDER [/INST]"
        elif "vicuna" in model_path:
            template = (
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                "USER: PLACEHOLDER ASSISTANT:"
            )
        elif "34b" in model_path:
            template = (
                "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>"
                "<|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError(
                f"Prompt template for {model_path} not implemented."
            )

        prompt = template.replace("PLACEHOLDER", f"<image>\n{prompt}")
        return prompt

    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to(
            "cuda", torch.float16
        )
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        return answer


class LLaVA_Next2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llama3-llava-next-8b", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=None
        )
        model.cuda().eval()
        model.tie_weights()

        if "llama3" in model_path.lower():
            conv_mode = "llava_llama_3"
        elif "qwen" in model_path.lower():
            conv_mode = "qwen_1_5"
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors="pt")["pixel_values"][0].half().cuda()
            for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )  # noqa: E501
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        video_kwargs_default = dict(
            overwrite=True, mm_spatial_pool_mode="average", force_sample=True
        )
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs[
                    "mm_spatial_pool_mode"
                ]

        rank, world_size = get_rank_and_world_size()
        model_name = get_model_name_from_path(model_path)
        import warnings
        # filter warning align with official code
        warnings.filterwarnings("ignore")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="auto",
            overwrite_config=overwrite_config,
        )
        model.eval()
        model.tie_weights()

        if "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(
            images, self.image_processor, self.model.config
        )
        image_tensor = [
            _image.to(dtype=torch.float16, device="cuda") for _image in image_tensor
        ]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        if dataset == "EgoExoBench_MCQ":
            return self.generate_inner_multi_video(message, dataset)
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError(
                "LLaVA-OneVision does not support multiple videos as input."
            )

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, 1, self.force_sample
        )

        time_instruciton = (
            f"The video lasts for {video_time:.2f} seconds,"
            f"and {len(video_frames[0])} frames are uniformly sampled from it."
            f"These frames are located at {frame_time}."
            f"Please answer the following questions related to this video.\n"
        )

        if self.force_sample:
            content = visual_content + time_instruciton + text_content
        else:
            content = visual_content + text_content

        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_multi_video(self, message, dataset=None):
        # only test on EgoExoBench_MCQ
        content, text_content, visual_content, videos = "", "", "", []
        video_nframes = []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"
                video_nframes.append(msg.get('nframes', 1))
                text_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        content = text_content

        image_tensors = []
        for nframes, video in zip(video_nframes, videos):
            if nframes == 1:
                # process image type
                video_frames = Image.open(video).convert("RGB")
                video_frames = [video_frames]  # Wrap in a list for consistency
            else:
                video_frames, frame_time, video_time = self.load_video(
                    video, nframes, 3, self.force_sample
                )
            frames = (
                self.image_processor.preprocess(video_frames, return_tensors="pt")[
                    "pixel_values"
                ]
                .half()
                .cuda()
            )
            image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities=["video"] * len(image_tensors)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, sample_fps, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames, frame_time, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)

class EgoGPT(LLaVA_OneVision):
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -300
    DEFAULT_IMAGE_TOKEN = "<image>"
    
    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", **kwargs):
        assert model_path is not None
        try:
            from egogpt.model.builder import load_pretrained_model
            from egogpt.conversation import conv_templates
            from egogpt.mm_utils import (
                process_images,
            )  # noqa: E501
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/EvolvingLMMs-Lab/EgoLife`"
            )
            raise err

        rank, world_size = get_rank_and_world_size()
        import warnings
        # filter warning align with official code
        warnings.filterwarnings("ignore")
        tokenizer, model, _ = load_pretrained_model(
            model_path,
            device_map="auto",
        )
        model.eval()

        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = model.get_vision_tower().image_processor
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )

    def split_text(self, text, keywords):
        import re
        pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
        parts = re.split(pattern, text)
        parts = [part for part in parts if part]
        return parts

    def generate_inner_video(self, message, dataset=None):
        # only test on EgoExoBench_MCQ dataset
        content = ""

        image_tensors = []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                content += self.DEFAULT_IMAGE_TOKEN + "\n"
                media_path = msg["value"]
                if msg['type'] == 'image':
                    video_frames = Image.open(media_path).convert("RGB")
                    video_frames = [video_frames]
                else:
                    video_frames, frame_time, video_time = self.load_video(
                        media_path, msg['nframes'], 3, True
                    )
                frames = (
                    self.image_processor.preprocess(video_frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()
                )
                image_tensors.append(frames)
                
        image_tensors = [t.to(self.model.device) for t in image_tensors]

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(self.conv_templates[conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        parts = self.split_text(prompt_question, ["<image>"])
        input_ids = []
        for part in parts:
            if part == "<image>":
                input_ids.append(self.IMAGE_TOKEN_INDEX)
            elif part == "<speech>":
                input_ids.append(self.SPEECH_TOKEN_INDEX)
            else:
                input_ids.extend(self.tokenizer(part).input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.model.device)
        
        speech = torch.zeros(3000, 128)
        speech = torch.stack([speech]).to(self.model.device).half()
        speech_lengths = torch.LongTensor([3000])
        
        image_sizes = [frame.size for frame in video_frames]
        generate_kwargs = {"eos_token_id": self.tokenizer.eos_token_id}

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            speech=speech,
            speech_lengths=speech_lengths,
            do_sample=False,
            temperature=0.5,
            max_new_tokens=2048,
            modalities=["video"] * len(image_tensors),
            **generate_kwargs,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

class LLaVA_OneVision_HF(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1
        self.model_path = model_path

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError("LLaVA-OneVision does not support multiple videos as input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, fps=1, force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": content}, {"type": "video"}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_fps = round(avg_fps / fps)
        frame_idx = list(range(0, total_frame_num, effective_fps))
        frame_time = [i / avg_fps for i in frame_idx]

        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])
        video_frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frame_num / avg_fps

        return video_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO":
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)
