import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import torchvision.transforms as T
from torchvision import transforms
import pandas as pd
import os
import re
from .utils.egoexobench import *
import torch

FAIL_MSG = 'Failed to obtain answer via API.'

class EgoExoBench_MCQ(VideoBaseDataset):
    MD5 = 'b5c3cfe5d316f1a67a4076991f16ca9c'
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='EgoExoBench_MCQ'):
        super().__init__(dataset=dataset, nframe=64)
        self.frame_fps = 2

    @classmethod
    def supported_datasets(cls):
        return ['EgoExoBench_MCQ']

    def prepare_dataset(self, dataset_name='EgoExoBench_MCQ', repo_id='Heleun/EgoExoBench_MCQ'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            video_dir = osp.join(pth, "processed_video")
            frames_dir = osp.join(pth, "processed_frames")

            if not osp.exists(data_file) or not osp.exists(video_dir) or not osp.exists(frames_dir):
                return False
            
            if md5(data_file) != self.MD5:
                return False
            
            return True
        cache_path = get_cache_path(repo_id)
        self.video_root = LMUDataRoot()
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return
                json_data_dir = os.path.join(dataset_path, 'MCQ')
                task_types = ['Ego-Exo-Relation', 'Ego-Exo-View-Transition', 'Ego-Exo-Temporal-Reasoning']
                self.data_list = []
                def add_media(value, content, medias):
                    if 'prefix' in value:
                        content += '\n' + value['prefix']
                    if 'video_path' in value:
                        content += '\n' + '<video>'
                        medias.append(
                            {
                                "type": "video",
                                "video_path": value["video_path"],
                                "video_start": value.get("video_start", None),
                                "video_end": value.get("video_end", None),
                                "nframes": value.get("nframes", None),
                            }
                        )
                    elif 'image_paths' in value:
                        content += '\n' + '<video>'
                        type = 'image' if len(value['image_paths']) == 1 else 'frames'
                        medias.append(
                            {
                                "type": type,
                                "image_paths": value["image_paths"],
                            }
                        )
                    return content
        
                def process_item(item):
                    question = ''
                    options = []
                    medias = []
                    answer = ''
                    response_format = 'Please respond with a single label (e.g., A, B, etc.). The answer is: '
                    for key, value in item:
                        if key == 'answer':
                            answer = value
                            continue
                        elif key == 'options':
                            for opt_idx, oitem in enumerate(value):
                                if isinstance(oitem, str):
                                    options.append('ABCD'[opt_idx] + '. ' + oitem)
                                elif isinstance(oitem, dict):
                                    opt = 'ABCD'[opt_idx] + '. '
                                    opt = add_media(oitem, opt, medias)
                                    options.append(opt)
                            continue
                        elif key == 'candidates' and isinstance(value, list):
                            for cand_idx, citem in enumerate(value):
                                question = add_media(citem, question, medias)
                            continue
                        elif key == 'response_format':
                            response_format = value
                            continue
                        elif isinstance(value, str):
                            if key == 'question':
                                question += 'Question: ' + value
                            else:
                                question += '\n' + value
                        elif isinstance(value, dict) and ('video_path' in value or 'image_paths' in value):
                            question = add_media(value, question, medias)
                    
                    options = '\n'.join(options)
                    return question, options, response_format, medias, answer


                for task_type in task_types:
                    task_dir = osp.join(json_data_dir, task_type)
                    subtask_files = [f for f in os.listdir(task_dir) if f.endswith('.json')]
                    subtask_files.sort()
                    
                    for subtask_file in subtask_files:
                        subtask_path = os.path.join(task_dir, subtask_file)

                        with open(subtask_path, 'r') as f:
                            data = json.load(f)
                    
                        for key, value in data.items():
                            question, options, response_format, medias, answer = process_item(value.items())
                            
                            self.data_list.append({
                                'task_type': task_type,
                                'subtask_type': subtask_file.replace('.json', ''),
                                'question': question,
                                'options': options,
                                'response_format': response_format,
                                'medias': medias,
                                'answer': answer,
                                'video': None,
                            })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            os.environ['HUGGINGFACE_TOKEN'] = ''
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            huggingface_hub.login(hf_token)
            
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        # transform
        self.transform = T.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])
        
        return dict(root=dataset_path, data_file=data_file)

    def load_into_video_and_process(self, media, mcq_idx):
        video_root = self.video_root

        if media['type'] in ['frames']:
            media['nframes'] = len(media['image_paths'])//2*2
            processed_video_path = f'{mcq_idx}.mp4'
            processed_video_path = osp.join(video_root, 'processed_video', processed_video_path)
        elif media['type'] in ['image']:
            processed_image_path = osp.join(video_root, 'processed_video', f'{mcq_idx}.jpg')
            return dict(type='image', value=processed_image_path)
        elif media['type'] in ['video']:
            processed_video_path = f'{mcq_idx}.mp4'
            processed_video_path = osp.join(video_root, 'processed_video', processed_video_path)

        return dict(type='video', value=processed_video_path, nframes=media.get('nframes', 8))
    
    def save_video_into_images(self, media, mcq_idx):
        video_root = self.video_root
        frame_base_path = osp.join(video_root, 'processed_frames', str(mcq_idx))
        
        input_images = sorted([
            osp.join(frame_base_path, f) for f in os.listdir(frame_base_path)
            if osp.isfile(osp.join(frame_base_path, f))
        ])

        return input_images
    
    def process_text_and_media(self, text, media_list, video_llm, mcq_idx):

        message = []
        chunks = re.split(r'(<image>|<video>)', text)
        media_index = 0
        media_list = eval(media_list)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])
        assert placeholder_count == len(media_list), \
            f"Placeholder count {placeholder_count} does not match media list length {len(media_list)}."
            

        for chunk in chunks:
            if chunk in ['<image>', '<video>']:
                if video_llm:
                    media_content = self.load_into_video_and_process(media_list[media_index], f'question{mcq_idx}_video{media_index}')
                    message.append(media_content)
                else:
                    # Save the video as individual image frames for processing
                    img_frame_paths = self.save_video_into_images(media_list[media_index], f'question{mcq_idx}_video{media_index}')
                    for im in img_frame_paths:
                        message.append(dict(type='image', value=im))
                    
                media_index += 1
            elif chunk.strip():
                message.append(dict(type='text', value=chunk.strip()))

        return message
    
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            mcq_idx = line
            assert line < len(self)
            line = self.data.iloc[line]
            mcq_idx = int(line['index'])
        
        text = line['question'] + '\nOptions:\n' + line['options'] + '\n' + line['response_format']
        message = self.process_text_and_media(text, line['medias'], video_llm, mcq_idx)
        return message
    
    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.egoexobench import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'EgoExoBench_MCQ',
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
    
