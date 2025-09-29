# AFRD
Official implementation of the paper "Attention Fusion Reverse Distillation for Multi-Lighting Image Anomaly Detection", targeting the Multi-Lighting Image Anomaly Detection (MLIAD) task—where multi-lighting conditions are leveraged to improve imaging quality and anomaly detection performance.

## Abstract
This study focuses on Multi-Lighting Image Anomaly Detection (MLIAD), where multiple lighting conditions are used to improve imaging quality and anomaly detection performance. Since existing image anomaly detection methods cannot handle multiple inputs (like multi-lighting images) for a single sample, it proposes the Attention Fusion Reverse Distillation (AFRD) method. Specifically, AFRD uses a pre-trained teacher network to extract features from multiple inputs, fuses these features via an attention module, and then uses a student network to regress the fused features—with regression errors serving as anomaly scores during inference. Experiments on the Eyecandies dataset show that AFRD outperforms other MLIAD alternatives and highlights the benefit of multi-lighting conditions for anomaly detection.
## Framework
<img width="700" height="766" alt="image" src="https://github.com/user-attachments/assets/c1cfe7b0-4750-4031-8ea3-e99ce9cae570" />
<img width="1495" height="736" alt="image" src="https://github.com/user-attachments/assets/b7818a31-1655-46ae-a054-a344dd6e899b" />

## Result
<img width="1395" height="618" alt="image" src="https://github.com/user-attachments/assets/3d9ec693-049f-4279-bb81-9e6058acc561" />
<img width="1468" height="471" alt="image" src="https://github.com/user-attachments/assets/37591d4d-d5a8-4a15-bf44-0c6a26f1cc21" />

## Citation
If you use this dataset in your research, please cite the following paper:
[Attention Fusion Reverse Distillation for Multi-Lighting Image Anomaly Detection](https://ieeexplore.ieee.org/document/10711818)
```bibtex
@INPROCEEDINGS{10711818,
  author={Zhang, Yiheng and Cao, Yunkang and Zhang, Tianhang and Shen, Weiming},
  booktitle={2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)}, 
  title={Attention Fusion Reverse Distillation for Multi-Lighting Image Anomaly Detection}, 
  year={2024},
  volume={},
  number={},
  pages={2134-2139},
  keywords={Accuracy;Computer aided software engineering;Automation;Attention mechanisms;Lighting;Imaging;Production;Feature extraction;Anomaly detection},
  doi={10.1109/CASE59546.2024.10711818}}
