# EgoExoBench

  
This is the official repository of **EgoExoBench** [Paper]()

## 📊 Benchmark Overview

**EgoExoBench** is a large-scale benchmark designed to evaluate cross-view video understanding in multimodal large language models (MLLMs). It contains paired egocentric–exocentric videos and over **7,300 multiple-choice questions** across **11 subtasks**, covering three key dimensions of ego–exo reasoning:

* **Ego-Exo Relation**
* **Ego-Exo View Transition**
* **Ego-Exo Temporal Reasoning**

## 📝 Data Preparation
EgoExoBench builds upon six publicly available ego–exo datasets. Please download the videos from the following sources:

* [Ego-Exo4D](https://ego-exo4d-data.org/)
* [LEMMA](https://sites.google.com/view/lemma-activity)
* [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn)
* [TF2023](https://github.com/facebookresearch/Ego-Exo)
* [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe)
* [CVMHAT](https://github.com/RuizeHan/CVMHT)

Place all datasets under the `data/` directory. The dataset structure is as follows:
```
EgoExoBench/
├── data/
│   ├── CVMHAT
│   	├── data
│   ├── Ego-Exo4D
│   	├── takes
│   ├── EgoExoLearn
│   ├── EgoMe
│   ├── LEMMA
│   ├── TF2023
│   	├── data
```

Next, download the EgoExoBench **multiple-choice questions (MCQs)** file and place it in the `MCQ/` directory.

## Installation
```
git clone https://github.com/ayiyayi/EgoExoBench.git
cd EgoExoBench
```

Please note that different VLMs require specific environment configurations (e.g., different versions of transformers). We recommend consulting the official documentation of each VLM to ensure an accurate evaluation and proper setup.
[Qwen2.5VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL3](https://github.com/OpenGVLab/InternVL), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [LLaVA-NeXT-Video](https://github.com/LLaVA-VL/LLaVA-NeXT)

## 🚀 Model Evaluation
Evaluation is built upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
```
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=1 run.py --data EgoExoBench_MCQ --model Qwen2.5-VL-7B-Instruct-ForVideo

# for very large VLMs
python run.py --data EgoExoBench_MCQ --model Qwen2.5-VL-72B-Instruct-ForVideo
```

## 🙏 Acknowledgements
This codebase is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
EgoExoBench builds upon publicly available ego–exo datasets: [Ego-Exo4D](https://ego-exo4d-data.org/), [LEMMA](https://sites.google.com/view/lemma-activity), [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn),  [TF2023](https://github.com/facebookresearch/Ego-Exo), [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe), [CVMHAT](https://github.com/RuizeHan/CVMHT).
Thanks for open-sourcing!
