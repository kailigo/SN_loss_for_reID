# Support Neighbor Loss for Person Re-Identification

This repository is for the paper introduced in the following paper

Kai Li, Zhengming Ding, Kunpeng Li, Yulun Zhang, and Yun Fu, "Support Neighbor Loss for Person Re-Identification", ACM Multimedia (ACM MM) 2018, [[arXiv]](https://arxiv.org/abs/1808.06030) 

## Environment
Python 3 + PyTorch 4.0

## Train

## Test

## Results

### Quantitative Results

Results on Market1501
![Retrieval](/figs/quantitative.png)

<!-- Analysis on the impact of gallery size
![Retrieval](/figs/gal_size_analysis.png) -->


### Visual Results

Pedestrian retrieval results
![Retrieval](/figs/retrieval.png)

Pedestrian feature embedding visualization
![Embedding visualization](/figs/embedding.png)

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{li2018support,
  title={Support neighbor loss for person re-identification},
  author={Li, Kai and Ding, Zhengming and Li, Kunpeng and Zhang, Yulun and Fu, Yun},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={1492--1500},
  year={2018},
  organization={ACM}
}
```

## Acknowledgements
This code is built on [this](https://github.com/huanghoujing/person-reid-triplet-loss-baseline) repository, developed by Houjing Huang.
