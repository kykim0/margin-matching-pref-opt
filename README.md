# Margin Matching Preference Optimization

PyTorch implementation for the paper:

**Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback** <br>
[Kyuyoung Kim](https://kykim0.github.io/)\*, Ah Jeong Seo\*, [Hao Liu](https://www.haoliu.ai/), [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html), [Kimin Lee](https://sites.google.com/view/kiminlee/home) <br>
In EMNLP 2024 Findings

<!--![Overview of MASN](model_overview.jpg)-->
<img src="./assets/concept.png" width="90%" align="middle">


Setup
--------
```
conda create -n mmpo python=3.10
# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# check gpu
import torch
torch.cuda.is_available()

# install the remaining package dependencies
python -m pip install -e .
pip install -r requirements.txt

# install flash attention
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# for deepspeed
conda install -c conda-forge mpi4py mpich
```


Dataset
--------
We evaluate MMPO using both human and AI feedback data to assess their performance on feedback of varying qualities:

### 1. [UltraFeedback](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned)
- For more precise evaluation, we used the filtered version released by AllenAI. Filtering details can be found in the dataset card.<br>

### 2. [SHP](https://huggingface.co/datasets/Ahjeong/SHP_filtered_for_MMPO)
- To build this [filtered version of SHP](https://huggingface.co/datasets/Ahjeong/SHP_filtered_for_MMPO), we sample uniformly across score differences to evaluate the methods across diverse quality margins.
- Following [KTO](https://github.com/ContextualAI/HALOs), we ensured that the same prompt appears no more than five times to prevent overfitting. More details can be found in `scripts/run_dpo_shp.py`.


Training
--------
Use the following commands to train models:

### DPO

- Gemma 2b:
```
accelerate launch --num_processes=2 --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_dpo.py recipes/gemma-2b/dpo.yaml --bt_beta=2.2
```

- Gemma 7b:
```
accelerate launch --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/gemma-7b/dpo.yaml --bt_beta=0.3
```
To train on the SHP dataset:
```
accelerate launch --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo_shp.py recipes/gemma-7b/dpo_shp.yaml --bt_beta=1.1
```

- Llama 8b:
```
accelerate launch --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml run_dpo.py recipes/llama-3-8b/dpo.yaml --bt_beta=2.2
```


Evaluation
--------
### 1. MT-bench

To evaluate on MT-bench, follow the instructions in [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). Below is the MT-bench results for the models trained with MMPO and DPO on UltraFeedback:

  Model  |  Gemma-2b  |  Gemma-7b  | LLaMA-8b  | 
------- | ------ | ------ | ------ | 
DPO | 6.09 | 7.40 | 7.41 |
MMPO | 6.10 | 7.53 | 7.58 |

<img src="./assets/mt-bench.png" width="50%" align="middle">

### 2. RewardBench

To evaluate on RewardBench, follow the instructions in [RewardBench](https://github.com/allenai/reward-bench) repository. As of September 2024, MMPO achieves state-of-the-art performance compared to other models at the same scale (see [leaderboard](https://huggingface.co/spaces/allenai/reward-bench)):

<img src="./assets/rbench_leaderboard.png" width="90%" align="middle">

More experimental details can be found in the paper. <br>


## Citation

```bibtex
```
