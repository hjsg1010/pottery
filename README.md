# AIRI_Pottery
  
3D classification and segmentation (location prediction) of points cloud data
* Each pottery(label) has various size of shards.  
* Each Shards has various size of points.  

---

## DATA PREPARING  

We used 3D scanned pottery data in .npy format. 
```
data 예시 쓰기
기영님 논문 data generation 참고
```

1. run make_filelist.ipynb 
   * generated shards list per each pottery (filelist_label_numofshards_randomseed_F.txt)  
2. run edit_h5_seglabel.ipynb 
   * shards' *CENTRALIZED* point cloud data per pottery, pottery labels(ID), segmentation_label 생성(label_numofshards_randomseed.h5),
   * segmentation_label: labelling (y_max, y_mean, y_min) of each shard
   * generate train data, test data list (train_files.txt, test_files.txt)
3. move h5 format data, train_files.txt and test_files.txt into same folder (please refer the path in provider.py and train_pottery_combined.py)  

---

## Model

![model](./images/model.png)

  
- train: ./train_pottery_combined.py  
- model: ./models/dgcnn+skipdense.py   

```bash
python train_pottery_combined.py
```

---

## RESULT FIGURE (ex)
#### 1. classification  
<!-- ![classification](./images/classification.png) -->
<img width="1121" alt="classification" src="https://user-images.githubusercontent.com/17421673/130309158-0bde2a29-8d8c-4fd7-8cce-eb0fd8ab46c2.png">


#### 2. location prediction  
<!-- ![segmentation](./images/segmentation.png) -->
<img width="987" alt="segmentation" src="https://user-images.githubusercontent.com/17421673/130309167-923f6247-ed85-4d4b-8d71-d326c52c8d46.png">

---

## EXPERIMENT
- Experiment settings: ubuntu 16.04, 64Gmemory, 16core, GPU Tesla V100-SXM2(16G) (used 1 GPU)  
- Tensorflow
 
---

## VISUALIZATION
```
run pottery_demo.ipynb
```
- required: tetgen library  

```bash
$ conda insatll -c conda-forge tetgen
```

---


## Acknowledgement
This code is heavily borrowed from [dgcnn](https://github.com/WangYueFt/dgcnn), and [pointnet](https://github.com/charlesq34/pointnet)

