# Medical World Model
<div align="center">
<br>
<h3>Medical World Model: Generative Simulation of Tumor Evolution for Treatment Planning</h3>

[Yijun Yang](https://yijun-yang.github.io/)<sup>1</sup>&nbsp;
[Zhao-Yang Wang](https://scholar.google.com/citations?hl=zh-CN&user=L_4sVVYAAAAJ)<sup>2</sup>&nbsp;
[Qiuping Liu](https://github.com/scott-yjyang/MeWM)<sup>3</sup>&nbsp;
[Shuwen Sun](https://github.com/scott-yjyang/MeWM)<sup>3</sup>&nbsp;
[Kang Wang](https://github.com/scott-yjyang/MeWM)<sup>4</sup>&nbsp;
[Rama Chellappa](https://scholar.google.com/citations?user=L60tuywAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>&nbsp;
[Zongwei Zhou](https://scholar.google.com/citations?user=JVOeczAAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup><br>
[Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ&hl=zh-CN)<sup>2</sup>&nbsp;
[Lei Zhu](https://sites.google.com/site/indexlzhu/home)<sup>1,5</sup>&nbsp;
[Yu-Dong Zhang](https://github.com/scott-yjyang/MeWM)<sup>3</sup>&nbsp;
[Jieneng Chen](https://beckschen.github.io/)<sup>2</sup>&nbsp;

<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou) &nbsp; <sup>2</sup> Johns Hopkins University &nbsp; <sup>3</sup> The First Affiliated Hospital of Nanjing Medical University &nbsp; <sup>4</sup> University of California, San Francisco <br> <sup>5</sup> The Hong Kong University of Science and Technology 
 
[![ArXiv](https://img.shields.io/badge/ArXiv-<2503.xx>-<COLOR>.svg)](https://arxiv.org/pdf/2503.xx) 
  
</div>


## [<a href="https://yijun-yang.github.io/MeWM/" target="_blank">Project Page</a>]

[//]: # (### Abstract)

Providing effective treatment and making informed clinical decisions are essential goals of modern medicine and clinical care.
We are interested in simulating disease dynamics for clinical decision-making, leveraging recent advances in large generative models.
To this end, we introduce the Medical World Model (MeWM), the first world model in medicine that visually predicts future disease states based on clinical decisions. 
MeWM comprises (i) vision-language models to serve as policy models, and (ii) tumor generative models as dynamics models. The policy model generates action plans, such as clinical treatments, while the dynamics model simulates tumor progression or regression under given treatment conditions. 
Building on this, we propose the inverse dynamics model that applies survival analysis to the simulated post-treatment tumor, enabling the evaluation of treatment efficacy and the selection of the optimal clinical action plan. As a result, the proposed MeWM simulates disease dynamics by synthesizing post-treatment tumors, with state-of-the-art specificity in Turing tests evaluated by radiologists. 
Simultaneously, its inverse dynamics model outperforms medical-specialized GPTs in optimizing individualized treatment protocols across all metrics.
Notably, MeWM improves clinical decision-making for interventional physicians, boosting F1-score in selecting the optimal TACE protocol by 13\%, paving the way for future integration of medical world models as the second readers.

For more info see the [project webpage](https://yijun-yang.github.io/MeWM/).

## 🔥 Latest News

* The project is quickly updated
* June 3, 2025: 👋 We release the Codebase of **MeWM** 
* June 3, 2025: 👋 We release the arXiv manuscript of **MeWM** 
* June 3, 2025: 👋 We release the project page of **MeWM** 



## Data

The raw public dataset could be found at [TCIA](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/).


## Citation


## Acknowledgement
The code is developed upon [DiffTumor](https://github.com/MrGiovanni/DiffTumor), [TextoMorph](https://github.com/MrGiovanni/TextoMorph). If you have issues with code and paper content, please contact [Yijun Yang](https://yijun-yang.github.io/).








