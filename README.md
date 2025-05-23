# Medical World Model
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

## Code



## Data

The raw public dataset could be found at [TCGA](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/).
