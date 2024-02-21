# Human Visual Attention [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
This repository contains a curated list of research papers and resources focusing on saliency and scanpath prediction, human attention, human visual search.


❗ Latest Update: 21 February 2024.
❗This repo is a work in progress. New updates coming soon, stay tuned!! :construction: :construction:

## 📚 Table of Contents
- [Human Attention Modelling](#human-attention-modelling)
    - <details>
        <summary>Saliency prediction</summary>
        
        | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
|:--------:|:--------------:|:----------------------------------------------------|:---------------------|:---------:|
|   2023   |      CVPR      | Learning from Unique Perspectives: User-aware Saliency Modeling | *Shi Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com//content/CVPR2023/papers/Chen_Learning_From_Unique_Perspectives_User-Aware_Saliency_Modeling_CVPR_2023_paper.pdf) 
|   2023   |      CVPR      | TempSAL - Uncovering Temporal Information for Deep Saliency Prediction | *Bahar Aydemir et al.*    | [📜 Paper](https://arxiv.org/abs/2301.02315) / [Code :octocat:](https://github.com/IVRL/Tempsal)
|   2023   |      BMVC      | Clustered Saliency Prediction | *Rezvan Sherkat et al.*    | [📜 Paper](https://arxiv.org/abs/2207.02205)
|   2023   |      NeurIPS      | What Do Deep Saliency Models Learn about Visual Attention? | *Shi Chen et al.*    | [📜 Paper](https://arxiv.org/abs/2310.09679) / [Code :octocat:](https://github.com/szzexpoi/saliency_analysis)
|   2023   |      NeurIPS      | What Do Deep Saliency Models Learn about Visual Attention? | *Shi Chen et al.*    | [📜 Paper](https://arxiv.org/abs/2310.09679) / [Code :octocat:](https://github.com/szzexpoi/saliency_analysis)
|   2018   |      IEEE Transactions on Image Processing      | Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model | *Marcella Cornia et al.*    | [📜 Paper](https://arxiv.org/pdf/1611.09571.pdf) / [Code :octocat:](https://github.com/marcellacornia/sam)
|   2015   |      CVPR      | SALICON: Saliency in Context | *Ming Jiang et al.*    | [📜 Paper](https://www-users.cse.umn.edu/~qzhao/publications/pdf/salicon_cvpr15.pdf) / [Project Page](http://salicon.net/)

    </details>
    
    - <details>
        <summary>Scanpath Prediction</summary>
    
        | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
        |:--------:|:--------------:|:---------:|:-----------:|:---------:|
        |   2023   |      arXiv      | Contrastive Language-Image Pretrained Models are Zero-Shot Human Scanpath Predictors | *Dario Zanca et al.*    | [📜 Paper](https://arxiv.org/abs/2305.12380) / [Code + Dataset :octocat:](https://github.com/mad-lab-fau/CapMIT1003)
        |   2023   |      CVPR      | Gazeformer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention | *Sounak Mondal et al.*    | [📜 Paper](https://arxiv.org/abs/2303.15274) / [Code :octocat:](https://github.com/cvlab-stonybrook/Gazeformer/)
        |   2022   |      ECCV      | Target-absent Human Attention | *Zhibo Yang et al.*    | [📜 Paper](https://arxiv.org/abs/2207.01166) / [Code :octocat:](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention)
        |   2021   |      CVPR      | Predicting Human Scanpaths in Visual Question Answering | *Xianyu Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Predicting_Human_Scanpaths_in_Visual_Question_Answering_CVPR_2021_paper.pdf) / [Code :octocat:](https://github.com/chenxy99/Scanpaths)

    </details>
- [Integrating Human Attention in AI models](#integrating-human-attention-in-ai-models)
    - [Image and Video Processing](#image-and-videoprocessing)
        - [Visual Recognition](#visual-recognition)
        - [Graphic Design](#graphic-design)
        - [Image Enhancement and Manipulation](#image-enhancement-and-manipulation)
        - [Image Quality Assessment](#image-quality-assessment)
    - [Vision-and-Language Applications](#vision-and-language)
        - [Automatic Captioning](#automatic-captioning)
        - [Visual Question Answering](#visual-question-answering)
    - [Language Modelling](#language-modelling)
        - [Machine Reading Comprehension](#machine-reading-comprehension)
        - [Natural Language Understanding](#natural-language-understanding)
    - [Domain-Specific Applications](#domain-specific-applications)
        - [Robotics](#robotics)
        - [Autonomous Driving](#autonomous-driving)
        - [Medicine](#medicine)
- [How to contribute](#how-to-contribute)

# Human Attention Modelling
<details>
    <summary>Saliency Prediction</summary>

## Saliency Prediction
| **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
|:--------:|:--------------:|:----------------------------------------------------|:---------------------|:---------:|
|   2023   |      CVPR      | Learning from Unique Perspectives: User-aware Saliency Modeling | *Shi Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com//content/CVPR2023/papers/Chen_Learning_From_Unique_Perspectives_User-Aware_Saliency_Modeling_CVPR_2023_paper.pdf) 
|   2023   |      CVPR      | TempSAL - Uncovering Temporal Information for Deep Saliency Prediction | *Bahar Aydemir et al.*    | [📜 Paper](https://arxiv.org/abs/2301.02315) / [Code :octocat:](https://github.com/IVRL/Tempsal)
|   2023   |      BMVC      | Clustered Saliency Prediction | *Rezvan Sherkat et al.*    | [📜 Paper](https://arxiv.org/abs/2207.02205)
|   2023   |      NeurIPS      | What Do Deep Saliency Models Learn about Visual Attention? | *Shi Chen et al.*    | [📜 Paper](https://arxiv.org/abs/2310.09679) / [Code :octocat:](https://github.com/szzexpoi/saliency_analysis)
|   2023   |      NeurIPS      | What Do Deep Saliency Models Learn about Visual Attention? | *Shi Chen et al.*    | [📜 Paper](https://arxiv.org/abs/2310.09679) / [Code :octocat:](https://github.com/szzexpoi/saliency_analysis)
|   2018   |      IEEE Transactions on Image Processing      | Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model | *Marcella Cornia et al.*    | [📜 Paper](https://arxiv.org/pdf/1611.09571.pdf) / [Code :octocat:](https://github.com/marcellacornia/sam)
|   2015   |      CVPR      | SALICON: Saliency in Context | *Ming Jiang et al.*    | [📜 Paper](https://www-users.cse.umn.edu/~qzhao/publications/pdf/salicon_cvpr15.pdf) / [Project Page](http://salicon.net/)

</details>

## Scanpath Prediction
| **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
|:--------:|:--------------:|:---------:|:-----------:|:---------:|
|   2023   |      arXiv      | Contrastive Language-Image Pretrained Models are Zero-Shot Human Scanpath Predictors | *Dario Zanca et al.*    | [📜 Paper](https://arxiv.org/abs/2305.12380) / [Code + Dataset :octocat:](https://github.com/mad-lab-fau/CapMIT1003)
|   2023   |      CVPR      | Gazeformer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention | *Sounak Mondal et al.*    | [📜 Paper](https://arxiv.org/abs/2303.15274) / [Code :octocat:](https://github.com/cvlab-stonybrook/Gazeformer/)
|   2022   |      ECCV      | Target-absent Human Attention | *Zhibo Yang et al.*    | [📜 Paper](https://arxiv.org/abs/2207.01166) / [Code :octocat:](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention)
|   2021   |      CVPR      | Predicting Human Scanpaths in Visual Question Answering | *Xianyu Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Predicting_Human_Scanpaths_in_Visual_Question_Answering_CVPR_2021_paper.pdf) / [Code :octocat:](https://github.com/chenxy99/Scanpaths)

# Integrating Human Attention in AI Models
## Image and Video Processing
### Visual Recognition
| **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
|:--------:|:--------------:|:---------:|:-----------:|:---------:|
|   2019   |      CVPR      | Shifting more attention to video salient object detection | *Deng-Ping Fan et al.*    | [📜 Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Shifting_More_Attention_to_Video_Salient_Object_Detection_CVPR_2019_paper.pdf) / [Code :octocat:](https://github.com/DengPingFan/DAVSOD)
### Graphic Design
| **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
|:--------:|:--------------:|:---------:|:-----------:|:---------:|
|   2017   | ACM Symposium on UIST (User Interface Software and Technology) | Learning Visual Importance for Graphic Designs and Data Visualizations |      Zoya Bylinskii       | [📜 Paper](https://arxiv.org/pdf/1708.02660.pdf) / [Code :octocat:](https://github.com/cvzoya/visimportance/tree/master?tab=readme-ov-file) |                                                    |
### Image Enhancement and Manipulation
### Image Quality Assessment
## Vision-and-Language Applications
### Automatic Captioning
### Visual Question Answering
## Language Modelling
### Machine Reading Comprehension
### Natural Language Understanding
## Domain-Specific Applications
### Robotics
### Autonomous Driving
### Medicine

# How to Contribute

1. Fork this repository and clone it locally.
2. Create a new branch for your changes: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Description of the changes'`.
4. Push to your fork: `git push origin feature-name`.
5. Open a pull request on the original repository by providing a description of your changes.

This project is in constant development, and we welcome contributions to include the latest research papers in the field or report issues 💥💥.
