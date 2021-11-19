# image-segmentation
## Table of Contents

 * [專案描述](#專案描述)
 * [執行專案](#執行專案)

## 專案描述
1. 架構:u-net
2. backbone: resnet50,resnet101

### 圖片分割
1. 訓練模型
2. 模型預測
3. 顯示結果

## 執行專案

### 安裝套件

```bash
$ pip install -r requirements.txt 
```

### 訓練模型

```bash
#到專案目錄下
$ cd path_to_dir/imagesegmentation

# 訓練模型
$ python train.py
#optional
#train_dir:訓練圖片路徑,內需要img和mask路徑
#categories:分類數量
#prev_weight:前模型參數檔案路徑
#input_length:模型輸入圖片長度
#input_width:模型輸入圖片寬度
#epochs:模型訓練週期
#batch_size:一批次的圖片數量
```

### 訓練模型
```bash
#到專案目錄下
$ cd path_to_dir/imagesegmentation

# 模型預測
$ python predict.py
#optional
#predict_dir:預測圖片路徑,內需要img路徑
#categories:分類數量
#weight_path:模型參數檔案路徑
#input_length:模型輸入圖片長度
#input_width:模型輸入圖片寬度
```

### 顯示預測結果
```bash
#到專案目錄下
$ cd path_to_dir/imagesegmentation

# 模型預測
$ python visualization.py
#optional
#data_dir:圖片路徑,內需要img和mask路徑
#categories:分類數量
```