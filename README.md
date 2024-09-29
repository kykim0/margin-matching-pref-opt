# Margin Matching Preference Optimization

Pytorch Implementation for the paper:

**[Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback][1]** <br>
[Kyuyoung Kim*](https://kykim0.github.io/), Ah Jeong Seo*, [Hao Liu](https://www.haoliu.ai/), [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html), [Kimin Lee](https://sites.google.com/view/kiminlee/home) <br>
In EMNLP 2024 Findings

<!--![Overview of MASN](model_overview.jpg)-->
<img src="./assets/concept.png" width="90%" align="middle">

Requirements
--------
python 3.7, pytorch 1.2.0


Dataset
--------
- [UltraFeedback](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned)
- [SHP](https://huggingface.co/datasets/Ahjeong/SHP_filtered_for_MMPO)


Training
--------
- best ckpt: https://huggingface.co/Ahjeong/MMPO_Gemma_7b_gamma1.1_epoch3


Evaluation
--------
- MT-Bench
- RewardBench: [leaderboard](https://huggingface.co/spaces/allenai/reward-bench)


## Citation

```bibtex
```

License
--------
MIT License

[1]: 
