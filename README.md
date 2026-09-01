# multi-view-gte

### [Paper](https://arxiv.org/pdf/2508.05857) | [Project page](https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html) 

This is the official code implementation of the paper "Multi-view Gaze Target Estimation" (ICCV 2025)

## Multi-view Gaze Target (MVGT) Dataset
Please go to the [project webpage](https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html) to download the dataset. Please check the Readme file of in the dataset regarding the data organization and annotation format.

We also release the depth maps estimated from Metric3D ([link](https://drive.google.com/file/d/1xbSstpy7MYP9oHisVhDQMGMQdAH9liTb/view?usp=drive_link)) and the estimated scale shifts for Cross-view GTE ([link](https://drive.google.com/file/d/1X5W55ktz8ee7gk_rgeBBZxDmAJ2AnQEC/view?usp=drive_link)).

## Run Code

We used pytorch 2.1.2, pytorch-lightning=2.3.1, and pytorch-cuda=11.8 in our experiments.
We performed leave-one-scene-out cross-validation. The results when validated on each of the 4 scenes are averaged and reported. We provide the code for both evaluating for each head/target visibility combination, and evaluating for cross-view GTE task.

In the multi-view experiments, the model is fine-tuned from the weights pretrained on Gazefollow dataset, while in the cross-view experiments the model is fine-tuned from the weights trained in the multi-view experiments. 

We provide the examplar model checkpoints for [multi-view](https://github.com/cvlab-stonybrook/multi-view-gte/releases/tag/multiview_ckpts) and [cross-view](https://github.com/cvlab-stonybrook/multi-view-gte/releases/tag/crossview_ckpts). The model weights after pretraining on Gazefollow is provided [here](https://github.com/cvlab-stonybrook/multi-view-gte/releases/tag/gf_trained_weights).

## License
The code is under [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Citation
If you find our code useful for your research, please cite
```
@article{miao2025multi,
  title={Multi-view Gaze Target Estimation},
  author={Miao, Qiaomu and Golani, Vivek Raju and Xu, Jingyi and Dutta, Progga Paromita and Hoai, Minh and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2508.05857},
  year={2025}
}
