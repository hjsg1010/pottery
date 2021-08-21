# AIRI Pottery
 
3D classification and segmentation (location prediction) of points cloud data
* Each pottery(label) has various size of shards.  
* Each Shards has various size of points.  

We aim to classify shards into different pottery classes and predict a relative location of shards within the pottery using point cloud representation. 


<img width="867" alt="abstract_figure" src="https://user-images.githubusercontent.com/17421673/130309350-9c01962c-4d32-4664-9d7c-03f3cfbfd3e4.png">


---

## Data Preparing

We used 3D scanned pottery data in .npy format. 

The figure below is an example of the pottery prototype and shards that we actually used.
<img width="480" alt="original_pottery_shards" src="https://user-images.githubusercontent.com/17421673/130309340-635b0317-fc44-4287-bafb-9c45ef656107.png">

Also, we first generated deep learning models to generate synthetic data using different methods as shown in below figure
<img width="686" alt="data_generation" src="https://user-images.githubusercontent.com/17421673/130309404-eedbca66-f805-4eec-a9d3-d7469b9dad1c.png">


#### Data Preprocessing
1. run make_filelist.ipynb 
   * generated shards list per each pottery (filelist_label_numofshards_randomseed_F.txt)  
2. run edit_h5_seglabel.ipynb 
   * shards' *CENTRALIZED* point cloud data per pottery, pottery labels(ID), segmentation_label 생성(label_numofshards_randomseed.h5),
   * segmentation_label: labelling (y_max, y_mean, y_min) of each shard
   * generate train data, test data list (train_files.txt, test_files.txt)
3. move h5 format data, train_files.txt and test_files.txt into same folder (please refer the path in provider.py and train_pottery_combined.py)  

---

## Model

<!-- ![model](./images/model.png) -->

<img width="952" alt="model_figure" src="https://user-images.githubusercontent.com/17421673/130309265-51801cda-9757-4794-8674-83c307cb3975.png">
- Simultaneous learning of pottery type classification and relative position prediction

  
- train: ./train_pottery_combined.py  
- model: ./models/dgcnn+skipdense.py   

```bash
python train_pottery_combined.py
```

---

## Result Figure (ex)
#### 1. Classification  
<!-- ![classification](./images/classification.png) -->
<img width="1121" alt="classification" src="https://user-images.githubusercontent.com/17421673/130309158-0bde2a29-8d8c-4fd7-8cce-eb0fd8ab46c2.png">


#### 2. Location Prediction  
<!-- ![segmentation](./images/segmentation.png) -->
<img width="987" alt="segmentation" src="https://user-images.githubusercontent.com/17421673/130309167-923f6247-ed85-4d4b-8d71-d326c52c8d46.png">

---

## Experiment
- Experiment settings: ubuntu 16.04, 64Gmemory, 16core, GPU Tesla V100-SXM2(16G) (used 1 GPU)  
- Tensorflow
 
---

## Visualization

run *pottery_demo.ipynb*

- required: tetgen library  

```bash
$ conda insatll -c conda-forge tetgen
```

---


## Acknowledgement
This code is heavily borrowed from [dgcnn](https://github.com/WangYueFt/dgcnn), and [pointnet](https://github.com/charlesq34/pointnet)

