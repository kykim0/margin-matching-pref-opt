# Margin Matching Preference Optimization

Pytorch Implementation for the paper:

**Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback** <br>
[Kyuyoung Kim*](https://kykim0.github.io/), Ah Jeong Seo*, [Hao Liu](https://www.haoliu.ai/), [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html), [Kimin Lee](https://sites.google.com/view/kiminlee/home) <br>
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
1. **[UltraFeedback](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned)**: the filtered version released by AllenAI <br>
2. **[SHP](https://huggingface.co/datasets/Ahjeong/SHP_filtered_for_MMPO)**:
- To build this [filtered version of SHP](https://huggingface.co/datasets/Ahjeong/SHP_filtered_for_MMPO), we sample uniformly across score differences to evaluate the methods over diverse quality margins.
- Following [KTO](https://github.com/ContextualAI/HALOs), we ensured that the same prompt appears no more than five times to prevent overfitting. More details can be found in `scripts/run_dpo_shp.py`.


Training
--------
Our best checkpoint is released at Huggingface: [`MMPO_Gemma_7b_gamma1.1_epoch3`](https://huggingface.co/Ahjeong/MMPO_Gemma_7b_gamma1.1_epoch3)

### DPO

- Gemma 2b:
```
accelerate launch --main_process_port=2380 --num_processes=2 --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_dpo.py recipes/gemma-2b/dpo.yaml --bt_beta=2.2
```

- Gemma 7b:
```
accelerate launch --main_process_port=2382 --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/gemma-7b/dpo.yaml --bt_beta=0.3
```
To training with SHP data, run:
```
accelerate launch --main_process_port=2381 --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo_shp.py recipes/gemma-7b/dpo_shp.yaml --bt_beta=1.1
```

- Llama 8b:
```
accelerate launch --main_process_port=2383 --num_processes=4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml run_dpo.py recipes/llama-3-8b/dpo.yaml --bt_beta=2.2
```


Evaluation
--------
- MT-Bench


- RewardBench: [leaderboard](https://huggingface.co/spaces/allenai/reward-bench)
<img src="./assets/rbench_leaderboard.png" width="90%" align="middle">
As of Sep. 2024


## Citation

```bibtex
```

License
--------
Apache License

