# Human Visual Attention [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
This repository contains a curated list of research papers and resources focusing on saliency and scanpath prediction, human attention, human visual search.


❗ Latest Update: 21 February 2024.
❗This repo is a work in progress. New updates coming soon, stay tuned!! :construction: :construction:

## Our Survey on Human Visual Attention 👀

🔥🔥 [*Trends, Applications, and Challenges in Human Attention Modelling*]() 🔥🔥\
\
**Authors:** 
[**Giuseppe Cartella**](https://scholar.google.com/citations?hl=en&user=0sJ4VCcAAAAJ)
[**Marcella Cornia**](https://scholar.google.com/citations?user=DzgmSJEAAAAJ&hl=it&oi=ao)
[**Vittorio Cuculo**](https://scholar.google.com/citations?user=usEfqxoAAAAJ&hl=it&oi=ao)
[**Alessandro D'Amelio**](https://scholar.google.com/citations?user=chkawtoAAAAJ&hl=it&oi=ao)
[**Dario Zanca**](https://scholar.google.com/citations?user=KjwaSXkAAAAJ&hl=it&oi=ao)
[**Giuseppe Boccignone**](https://scholar.google.com/citations?user=LqM0uJwAAAAJ&hl=it&oi=ao)
[**Rita Cucchiara**](https://scholar.google.com/citations?user=OM3sZEoAAAAJ&hl=it&oi=ao)

<p align="center">
    <img src="figure.jpg" style="max-width:500px">
</p>

# 📚 Table of Contents
- **Human Attention Modelling**
    - <details>
        <summary>Saliency Prediction</summary>
        
        | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
        |:--------:|:--------------:|:----------------------------------------------------|:---------------------|:---------:|
        |   2023   |      CVPR      | Learning from Unique Perspectives: User-aware Saliency Modeling | *Shi Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com//content/CVPR2023/papers/Chen_Learning_From_Unique_Perspectives_User-Aware_Saliency_Modeling_CVPR_2023_paper.pdf) 
        |   2023   |      CVPR      | TempSAL - Uncovering Temporal Information for Deep Saliency Prediction | *Bahar Aydemir et al.*    | [📜 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Aydemir_TempSAL_-_Uncovering_Temporal_Information_for_Deep_Saliency_Prediction_CVPR_2023_paper.pdf) / [Code :octocat:](https://github.com/IVRL/Tempsal)
        |   2023   |      BMVC      | Clustered Saliency Prediction | *Rezvan Sherkat et al.*    | [📜 Paper](https://arxiv.org/pdf/2207.02205.pdf)
        |   2023   |      NeurIPS      | What Do Deep Saliency Models Learn about Visual Attention? | *Shi Chen et al.*    | [📜 Paper](https://arxiv.org/pdf/2310.09679.pdf) / [Code :octocat:](https://github.com/szzexpoi/saliency_analysis)
        |   2022   |      Neurocomputing      | TranSalNet: Towards perceptually relevant visual saliency prediction | *Jianxun Lou et al.*    | [📜 Paper](https://www.sciencedirect.com/science/article/pii/S0925231222004714?via%3Dihub) / [Code :octocat:](https://github.com/LJOVO/TranSalNet?tab=readme-ov-file)
      |   2018   |      IEEE Transactions on Image Processing      | Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model | *Marcella Cornia et al.*    | [📜 Paper](https://arxiv.org/pdf/1611.09571.pdf) / [Code :octocat:](https://github.com/marcellacornia/sam)
        |   2015   |      CVPR      | SALICON: Saliency in Context | *Ming Jiang et al.*    | [📜 Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Jiang_SALICON_Saliency_in_2015_CVPR_paper.pdf) / [Project Page](http://salicon.net/)
        |   2009   |      ICCV      | Learning to Predict Where Humans Look | *Tilke Judd et al.*    | [📜 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459462)
        |   1998   |      TPAMI      | A Model of Saliency-Based Visual Attention for Rapid Scene Analysis | *Laurent Itti et al.*    | [📜 Paper](https://forums.cs.tau.ac.il/~hezy/Vision%20Seminar/koch%20attention%20pami.pdf)
    </details>
    
    - <details>
        <summary>Scanpath Prediction</summary>
    
        | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
        |:--------:|:--------------:|:---------:|:-----------:|:---------:|
        |   2023   |      arXiv      | Contrastive Language-Image Pretrained Models are Zero-Shot Human Scanpath Predictors | *Dario Zanca et al.*    | [📜 Paper](https://arxiv.org/pdf/2305.12380.pdf) / [Code + Dataset :octocat:](https://github.com/mad-lab-fau/CapMIT1003)
        |   2023   |      CVPR      | Gazeformer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention | *Sounak Mondal et al.*    | [📜 Paper](https://arxiv.org/pdf/2303.15274.pdf) / [Code :octocat:](https://github.com/cvlab-stonybrook/Gazeformer/)
        |   2022   |      ECCV      | Target-absent Human Attention | *Zhibo Yang et al.*    | [📜 Paper](https://arxiv.org/pdf/2207.01166.pdf) / [Code :octocat:](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention)
        |   2022   |      Journal of Vision      | DeepGaze III: Modeling free-viewing human scanpaths with deep learning | *Matthias Kümmerer et al.*    | [📜 Paper](https://jov.arvojournals.org/article.aspx?articleid=2778776) / [Code :octocat:](https://github.com/matthias-k/DeepGaze)
        |   2021   |      CVPR      | Predicting Human Scanpaths in Visual Question Answering | *Xianyu Chen et al.*    | [📜 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Predicting_Human_Scanpaths_in_Visual_Question_Answering_CVPR_2021_paper.pdf) / [Code :octocat:](https://github.com/chenxy99/Scanpaths)
        |   2019   |      TPAMI      | Gravitational Laws of Focus of Attention | *Dario Zanca et al.*    | [📜 Paper](https://ieeexplore.ieee.org/abstract/document/8730418) / [Code :octocat:](https://github.com/dariozanca/G-Eymol)
        |   2015   |      Vision Research      | Saccadic model of eye movements for free-viewing condition | *Olivier Le Meur et al.*    | [📜 Paper](https://www.sciencedirect.com/science/article/pii/S0042698915000504)
    </details>

- **Integrating Human Attention in AI models**
    - ***Image and Video Processing***
        - <details>
            <summary>Visual Recognition</summary>
            
            | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
            |:--------:|:--------------:|:---------:|:-----------:|:---------:|
            |   2019   |      CVPR      | Shifting more attention to video salient object detection | *Deng-Ping Fan et al.*    | [📜 Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Shifting_More_Attention_to_Video_Salient_Object_Detection_CVPR_2019_paper.pdf) / [Code :octocat:](https://github.com/DengPingFan/DAVSOD)
          </details>
        - <details>
            <summary>Graphic Design</summary>
            
            | **Year** | **Conference / Journal** | **Title** | **Authors** | **Links** |
            |:--------:|:--------------:|:---------:|:-----------:|:---------:|
            |   2017   | ACM Symposium on UIST (User Interface Software and Technology) | Learning Visual Importance for Graphic Designs and Data Visualizations |      Zoya Bylinskii       | [📜 Paper](https://arxiv.org/pdf/1708.02660.pdf) / [Code :octocat:](https://github.com/cvzoya/visimportance/tree/master?tab=readme-ov-file) 
        </details>
    
        - <details>
            <summary>Image Enhancement and Manipulation</summary>
        </details>
        
        - <details>
            <summary>Image Quality Assessment</summary>
        </details>
    - ***Vision-and-Language Applications***
        - <details>
            <summary>Automatic Captioning</summary>
        </details>
        
        - <details>
            <summary>Visual Question Answering</summary>
        </details>
    - ***Language Modelling***
        - <details>
            <summary>Machine Reading Comprehension</summary>
        </details>
        
        - <details>
            <summary>Natural Language Understanding</summary>
        </details>
    - ***Domain-Specific Applications***
        - <details>
            <summary>Robotics</summary>
        </details>
        
        - <details>
            <summary>Autonomous Driving</summary>
        </details>
        
        - <details>
            <summary>Medicine</summary>
        </details>



# How to Contribute 🚀

1. Fork this repository and clone it locally.
2. Create a new branch for your changes: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Description of the changes'`.
4. Push to your fork: `git push origin feature-name`.
5. Open a pull request on the original repository by providing a description of your changes.

This project is in constant development, and we welcome contributions to include the latest research papers in the field or report issues 💥💥.
