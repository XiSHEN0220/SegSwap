# SegSwap
Pytorch implementation of paper "Learning Co-segmentation by Segment Swapping for Retrieval and Discovery" 
 
Present in CVPR 2022 [Image Matching Workshop](https://image-matching-workshop.github.io/) and  [Transformers for Vision Workshop](https://sites.google.com/view/t4v-cvpr22/papers)

[[arXiv]](http://arxiv.org/abs/2110.15904) [[Project page]](http://imagine.enpc.fr/~shenx/SegSwap/) [[Supplementary material]](http://imagine.enpc.fr/~shenx/SegSwap/suppMat.pdf) [[Youtube Video]](https://youtu.be/9pKwNGZPDr8)[[Slides]](http://imagine.enpc.fr/~shenx/SegSwap/slides.pdf)


<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train.jpg" width="800px" alt="teaser">
</p>

<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/arch.jpg" width="800px" alt="teaser">
</p>



If our project is helpful for your research, please consider citing : 
``` 
@article{shen2021learning,
  title={Learning Co-segmentation by Segment Swapping for Retrieval and Discovery},
  author={Shen, Xi and Efros, Alexei A and Joulin, Armand and Aubry, Mathieu},
  journal={arXiv},
  year={2021}
```



## Table of Content
* [0. Quickstart](#0-quickstart)
* [1. Installation](#1-installation)
* [2. Training Data Generation](#2-training-data-generation)
* [3. Evaluation](#3-evaluation)
    * [3.1 One-shot Art Detail Detection on Brueghel Dataset](#31-one-shot-art-detail-detection-on-brueghel-dataset)
    * [3.2 Place Recognition on Tokyo247 Dataset](#32-place-recognition-on-tokyo247-dataset)
    * [3.3 Place Recognition on Pitts30K Dataset](#33-place-recognition-on-pitts30k-dataset)
    * [3.4 Discovery on Internet Dataset](#34-discovery-on-internet-dataset)
* [4. Train](#4-train)
* [5. Acknowledgement](#5-acknowledgement)
* [6. ChangeLog](#6-changelog)
* [7. License](#7-license)


## 0. Quickstart

A quick start guide of how to use our code is available in [demo/demo.ipynb](https://github.com/XiSHEN0220/SegSwap/tree/main/demo/demo.ipynb)

<p align="center">
<a href="https://github.com/XiSHEN0220/SegSwap/tree/main/demo/demo.ipynb"><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/demo/demo.png" width="600px" alt="notebook"></a>
</p>

## 1. Installation

### 1.1. Dependencies

Our model can be learnt on a **a single GPU Tesla-V100-16GB.**
The code has been tested in [Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/#v171) + cuda 10.2
 
Other dependencies can be installed via (tqdm, kornia, opencv-python, scipy) : 
``` Bash
bash requirement.sh
```


### 1.2. Pre-trained MocoV2-resnet50 + cross-transformer (~300M)

Quick download : 

``` Bash
cd model/pretrained
bash download_model.sh
```


## 2. Training Data Generation

### 2.1. Download COCO (~20G)

This command will download coco2017 training set + annotations (~20G). 
``` Bash
cd data/COCO2017/download_coco.sh
bash download_coco.sh
```

### 2.2. Image Pairs with One Repeated Object 

#### 2.2.1 Generating 100k pairs (~18G)

This command will generate 100k image pairs with one repeated object. 
``` Bash
cd data/
python generate_1obj.py --out-dir pairs_1obj_100k 
```

#### 2.2.1 Examples of image pairs 

<p align="center">
<table>
  <tr>
    <th>Source</th>
    <th>Blended Obj + Background</th>
    <th>Stylised Source</th>
    <th>Stylised Background</th>
  </tr>
  
  <tr>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_1_a.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_1_b.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_1_as.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_1_bs.jpg" width="200px"></td>
  </tr>
  
  <tr>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_2_a.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_2_b.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_2_as.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/1obj_2_bs.jpg" width="200px"></td>
  </tr>
  
  
</table>
</p>

#### 2.2.2 Visualizing correspondences and masks of the generated pairs 

This command will generate 10 pairs and visualize correspondences and masks of the pairs. 

``` Bash
cd data/
bash vis_pair.sh
```

These pairs can be illustrated via **vis10_1obj/vis.html**


### 2.3. Image Pairs with Two Repeated Object 

#### 2.3.1 Generating 100k pairs (~18G)

This command will generate 100k image pairs with one repeated object. 
``` Bash
cd data/
python generate_2obj.py --out-dir pairs_2obj_100k 
```

#### 2.3.1 Examples of image pairs 

<p align="center">
<table>
  <tr>
    <th>Source</th>
    <th>Blended Obj + Background</th>
    <th>Stylised Source</th>
    <th>Stylised Background</th>
  </tr>
  
  <tr>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_1_a.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_1_b.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_1_as.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_1_bs.jpg" width="200px"></td>
  </tr>
  
  <tr>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_2_a.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_2_b.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_2_as.jpg" width="200px"></td>
    <td><img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/train_data/2obj_2_bs.jpg" width="200px"></td>
  </tr>
  
  
</table>
</p>

#### 2.3.2 Visualizing correspondences and masks of the generated pairs 

This command will generate 10 pairs and visualize correspondences and masks of the pairs. 

``` Bash
cd data/
bash vis_pair.sh
```

These pairs can be illustrated via **vis10_2obj/vis.html**


## 3. Evaluation

### 3.1 One-shot Art Detail Detection on Brueghel Dataset

#### 3.1.1 Visual results: top-3 retrieved images

<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/vis_brueghel.jpg" width="800px" alt="teaser">
</p>


#### 3.1.2 Data 

Brueghel dataset has been uploaded [in this repo](https://github.com/XiSHEN0220/SegSwap/tree/main/data/Brueghel) 



### 3.1.3 Quantitative results

The following command conduct evaluation on Brueghel with pre-trained cross-transformer:

``` Bash
cd evalBrueghel
python evalBrueghel.py --out-coarse out_brueghel.json --resume-pth ../model/hard_mining_neg5.pth --label-pth ../data/Brueghel/brueghelTest.json
```

Note that this command will save the features of Brueghel(~10G).

### 3.2 Place Recognition on Tokyo247 Dataset

#### 3.2.1 Visual results: top-3 retrieved images

<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/vis_tokyo.jpg" width="800px" alt="teaser">
</p>


#### 3.2.2 Data 

Download Tokyo247 from [its project page](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)

Download the [top-100 results](https://drive.google.com/file/d/1mavpK9gHpCM8QBGp-J2faCJkd32BhdCp/view?usp=sharing) used by patchVlad(~1G). 

The data needs to be organised: 

```
./SegSwap/data/Tokyo247
                    ├── query/
                        ├── 247query_subset_v2/
                    ├── database/
...

./SegSwap/evalTokyo
                    ├── top100_patchVlad.npy
```

### 3.2.3 Quantitative results

The following command conduct evaluation on Tokyo247 with pre-trained cross-transformer:
``` Bash
cd evalTokyo
python evalTokyo.py --qry-dir ../data/Tokyo247/query/247query_subset_v2 --db-dir ../data/Tokyo247/database --resume-pth ../model/hard_mining_neg5.pth
```


### 3.3 Place Recognition on Pitts30K Dataset

#### 3.3.1 Visual results: top-3 retrieved images

<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/vis_pitts.jpg" width="800px" alt="teaser">
</p>


#### 3.3.2 Data 

Download Pittsburgh dataset from [its project page](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)

Download the [top-100 results](https://drive.google.com/file/d/1Rrorci_SAxZpNoRERtIwEIXzORcyMaIr/view?usp=sharing) used by patchVlad (~4G). 

The data needs to be organised: 

```
./SegSwap/data/Pitts
                ├── queries_real/
...

./SegSwap/evalPitts
                    ├── top100_patchVlad.npy
```

### 3.3.3 Quantitative results

The following command conduct evaluation on Pittsburgh30K with pre-trained cross-transformer:
``` Bash
cd evalPitts
python evalPitts.py --qry-dir ../data/Pitts/queries_real --db-dir ../data/Pitts --resume-pth ../model/hard_mining_neg5.pth
```

### 3.4 Discovery on Internet Dataset

#### 3.4.1 Visual results

<p align="center">
<img src="https://github.com/XiSHEN0220/SegSwap/blob/main/fig/vis_int.jpg" width="800px" alt="teaser">
</p>


#### 3.4.2 Data 

Download Internet dataset from its [project page](https://people.csail.mit.edu/mrub/ObjectDiscovery/)

We provide a script to quickly download and preprocess the data (~400M): 

``` Bash
cd data/Internet
bash download_int.sh
```

The data needs to be organised: 

```
./SegSwap/data/Internet
                ├── Airplane100
                    ├── GroundTruth                
                ├── Horse100
                    ├── GroundTruth                
                ├── Car100
                    ├── GroundTruth                                
```

### 3.4.3 Quantitative results

The following commands conduct evaluation on Internet with pre-trained cross-transformer
``` Bash
cd evalInt
bash run_pair_480p.sh
bash run_best_only_cycle.sh
```

## 4. Training 

### Stage 1: standard training

Supposing that the generated pairs are saved in `./SegSwap/data/pairs_1obj_100k` and `./SegSwap/data/pairs_2obj_100k`. 

Training command can be found in `./SegSwap/train/run.sh`. 

Note that this command should be able to be launched on a single GPU with 16G memory. 

``` Bash
cd train
bash run.sh
```

### Stage 2: hard mining

In `train/run_hardmining.sh`, replacing `--resume-pth` by the model trained in the 1st stage, than running: 

``` Bash
cd train
bash run_hardmining.sh
```



## 5. Acknowledgement

We appreciate helps from :  

* authors of [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD) who share their top-100 lists on Tokyo247 and Pitts30K with us. 

* [Dr. Relja Arandjelović](http://www.relja.info/) for providing Tokyo247 and Pitts30K datasets.

* public code like [Kornia](https://github.com/kornia/kornia)


Part of code is borrowed from our previous projects: [ArtMiner](https://github.com/XiSHEN0220/ArtMiner) and [Watermark](https://github.com/XiSHEN0220/WatermarkReco)



## 6. ChangeLog

* **27/10/22**, add a demo
* **21/10/21**, model, evaluation + training released
* **01/11/21**, update arxiv link + supplmentary material

## 7. License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including Kornia, Pytorch, and uses datasets which each have their own respective licenses that must also be followed.

