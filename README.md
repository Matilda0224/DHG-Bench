# DHG-Bench

DHG-Bench is an open and unified pipline for Deep Hypergraph Learning (DHGL) based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://www.pyg.org/). The pipline includes **20** benchmark datasets, **16(16/16)** HNN algorithms, **3(3/3)** hypergraph benchmark tasks.

## 1. Datasets

* Homophily Hypergraph Datasets

| Datasets | Nodes | Hyperedges | Features | Classes | Description  |Support|
| ---------- | ------------------------ |  ---------- | ------------------------ | ------------------- |  ------------------- |  :-------------------: | 
| Cora | 2708 | 1579 | 1433 | 7 | co-citation |Y|
| Citeseer |  3312 | 1079 | 3703 | 6 |co-citation |Y|
| Pubmed | 19717 | 7963 | 500 | 3 |co-citation | Y|
| Cora-CA |  2708 | 1072 | 1433 | 7 |co-authorship |Y|
| DBLP-CA |  41302 | 22363 | 1425 | 6 |co-authorship |Y|
| Zoo | 101 | 43 | 16| 7 | UCI | Y |
| 20Newsgroups|16242|100|100|4|UCI | Y |
| Mushroom |8124|298|22|2|UCI | Y |
| NTU2012 | 2012 |2012|100|67|graphics| Y |
|ModelNet40|12311|12311|100|40|graphics| Y |
|Yelp|50758|679302|1862|9|recommendation|Y |
|House|1290|341|100|2|co-purchasing|Y |
|Warmart| 88860 |699062|100|11|co-purchasing| Y |
|Trivago|172,738 | 233,202 | 300 | 160 | hotel |Y|
|ogbn-mag|736,389|1,134,649|128|349| co-authorship |Y|
|Amazon|2,268,083| 4,285,295| 1,000 |15|co-purchasing|Y |
|MAG-PM|2,353,996|1,082,711 |1,000 | 22|co-authorship|Y |

* Heterophily Hypergraph Datasets

| Datasets | Nodes | Hyperedges | Features | Classes | Description  |Support|
| ---------- | ------------------------ |  ---------- | ------------------------ | ------------------- |  ------------------- | :-------------------: | 
|Actor|16,255|10,164|50|3| co-occurence |Y|
|Amazon-ratings|22,299|2,090|111|5| co-purchasing |Y|
|Twitch-gamers| 16,812|2,627|7|2| co-create |Y|
|Pokec|14,998|2,406|65|2| co-friendship |Y|

* Edge-dependent Node Classification Datasets

| Datasets | Nodes | Hyperedges | Features | Classes |  Description  |
| ---------- | ------------------------ | ------------------- | ---------- | ------------------------ | ------------------- | 
| Coauth-DBLP |108,484| 91,266||3|co-authorship |
| Coauth-AMiner | 1,712,433| 2,037,605 ||3|co-authorship |
| Email-Enron | 21,251| 101,124 ||3|co-occurence |
| Email-Eu | 986 | 209,508 ||3| co-occurence |
| Stack-Biology |15,490| 26,823 ||3|co-create |
| Stack-Physics | 80,936| 200,811 ||3|co-create |

* Sensitive Attribute Datasets
 
| Datasets | Nodes | Hyperedges | Features | Sens |  Label  |Support|
| ---------- | ------------------------ | ------------------- | ---------- | ------------------------ | ------------------- | :-------------------: | 
| German Credit|1,000|1,000|27|Gender|Credit status| Y |
| Bail |18,876|18,876|18|Race|Bail decision| Y |
| Credit Defaulter|30,000|30,000|13|Age|Future default| Y |

## 2. Algorithms

| **ID** | **Method** | **Paper** |     **Venue**     |  **Completed** |  **Edge Embedding** |
| :----: | :----: | :----: |:----: |  :----: |  :----: | 
|1 | HGNN | [Hypergraph neural networks](https://cdn.aaai.org/ojs/4235/4235-13-7289-1-10-20190705.pdf) | AAAI 2019 | Y | Y |
|2|HyperGCN|[HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://proceedings.neurips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf)|NeurIPS 2019| Y | N |
|3| HNHN | [HNHN: Hypergraph Networks with Hyperedge Neurons](https://grlplus.github.io/papers/40.pdf) | ICML WS 2020 | Y | Y |
|4| HCHA | [Hypergraph convolution and hypergraph attention](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304404?via=ihub) | PR 2020 | Y | Y |
|5| UniGNN | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://www.ijcai.org/proceedings/2021/0353.pdf) | IJCAI 2021 | Y | Y |
|6| AllSetformer | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/forum?id=hpBTIv2uy_E) | ICLR 2022 | Y |N|
|7|HyperND|[Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a/prokopchik22a.pdf)|ICML 2022|Y|N|
|8| EHNN|[Equivariant hypergraph neural networks](https://arxiv.org/pdf/2208.10428)|ECCV 2022|Y|Y|
|9| LEGCN|[Semi-supervised Hypergraph Node Classification on Hypergraph Line Expansion](https://arxiv.org/pdf/2005.04843)|CIKM 2022|Y| N |
|10| ED-HNN | [Equivariant Hypergraph Diffusion Neural Operators](https://openreview.net/forum?id=RiTjKoscnNd) | ICLR 2023 | Y | Y |
|11|PhenomNN|[From Hypergraph Energy Functions to Hypergraph Neural Networks](https://proceedings.mlr.press/v202/wang23d/wang23d.pdf)|ICML 2023|Y|N|
|12|SheafHyperGNN|[Sheaf Hypergraph Networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/27f243af2887d7f248f518d9b967a882-Paper-Conference.pdf)|NeurIPS 2023|Y |N|
|13|HJRL|[Hypergraph Joint Representation Learning for Hypervertices and Hyperedges via Cross Expansion](https://openreview.net/forum?id=fxLaL5s6UH)|AAAI 2024|Y|Y|
|14| DPHGNN |[DPHGNN: A Dual Perspective Hypergraph Neural Networks](https://arxiv.org/pdf/2405.16616)|KDD 2024| Y |Y|
|15|T-HyperGNNs |[T-HyperGNNs: Hypergraph Neural Networks via Tensor Representations](https://ieeexplore.ieee.org/document/10462516)| TNNLS 2024|Y|N|
|16|TF-HNN |[Training-Free Message Passing for Learning on Hypergraphs](https://openreview.net/pdf?id=4AuyYxt7A2)| ICLR 2025|Y|N|

## 3. Evaluations

### 3.1. Evaluation Tasks

* Tranductive Setting Only

| **Task** | **Type** |  **Effectiveness Metrics** |     
| ---------- | ---------------- | ---------------- | 
|Node Classification | node-level | **Accuracy**, F1-score, AUC-ROC|
|Hyperedge Prediction | hyperedge-level |**Accuracy**, F1-score, AUC-ROC|
|Hypergraph Classification | hypergraph-level |**Accuracy, Macro-F1**|

### 3.2. Efficiency and Scalability

* Training/Inference Time Cost

* Memory Usage during Training and Inference

* Complexity Analysis

### 3.3. Robustness

* Supervision-robustness

* Structure-robustness

* Feature-robustness

### 3.4. Fairness

* Demographic Parity (DP)

* Equalized Odds. (EO)