
# Multi-Modal Deepfake Detection Using Attention-Based Fusion Framework
[Research Paper](https://github.com/vcbsl/Audio-Visual-Deepfake-Detection-Localization/)
## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
## Overview
In the digital age, the rise of deepfakes and synthetic media presents significant threats to societal and political integrity. Deepfakes that manipulate multiple modalities, such as audio and visual, are especially concerning due to their increased realism. Our project addresses this challenge by proposing a novel multi-modal attention framework based on recurrent neural networks (RNNs). This framework leverages contextual information to enhance deepfake detection and localization across audio-visual modalities.
### Figure 1: Overview of the Proposed Model
![Model Architecture](images/teaser.png)


## Methodology
Our approach focuses on the attention-based fusion of heterogeneous data streams from different modalities, specifically audio and visual signals. The primary challenge lies in bridging the distributional modality gap, which can hinder effective data fusion. We address this challenge by applying attention mechanisms to multi-modal multisequence representations, allowing the model to learn the most relevant features for deepfake detection and localization.
### Figure 2: Illustration of the proposed Multi-Modal Multi-Sequence Bi-modal Attention (MMMS-BA) model for audio-visual deepfake detection and localization
![FakeAVCeleb Results](images/CCMA.png)
## Key Features
- **Multi-Modal Data Fusion**: Utilizes attention mechanisms to integrate audio and visual data effectively.
- **RNN-Based Framework**: Leverages the sequential nature of audio-visual data using recurrent neural networks.
- **Improved Detection and Localization**: Demonstrates superior performance compared to existing methods, with a 3.47% increase in detection accuracy and a 2.05% increase in localization precision.

## Datasets
We conducted thorough experimental validations on the following audio-visual deepfake datasets:

| Dataset | Year | Tasks | Manipulated Modality | Manipulation Method | 
|---------|------|-------|----------------------|---------------------|
| [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) | 2021 | Detection | Audio and Visual | Re-enactment |
| [LAV-DF](https://github.com/ControlNet/LAV-DF) | 2022 | Detection and Localization | Audio and Visual | Content-driven | 
| [TVIL](https://github.com/ymhzyj/UMMAFormer) | 2023 | Detection and Localization | Audio and Visual | Inpainting forgery | 
| [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M) | 2023 | Detection and Localization | Audio and Visual | Content-driven | 

## Results
Our framework's effectiveness was validated through comprehensive experiments on the aforementioned datasets. The results demonstrate our approach's superiority in both deepfake detection and localization, achieving state-of-the-art performance.

## Citation

## Acknowledgements

### License
This project is licensed under the terms of the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license. The copyright of the images remains with the original owners.



