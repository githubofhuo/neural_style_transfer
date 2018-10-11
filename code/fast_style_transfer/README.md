### 说明

本目录是对`Fast Neural Style, Justin Johnson, 2016`的实现

代码中的实现与原文有以下几处不同：

1. 原文中风格迁移网络由卷积层，Batch Normalization层和激活层组成，此处将Batch Normalization 替换为Instance Normalization。Instance Normalization只对每个样本求均值和方差，而 Batch Normalization 会对一个batch中的所有样本求均值。对`B*C*H*W`的Tensor, IN 计算`B*C`个均值，BN计算`B`个均值。

   （参考Dmitry Ulyanov的文章：Instance Normalization: The Missing Ingredient for Fast Sytlization）

2. 网络结构有以下特点：

   1. 先下采样，然后上采样使计算量变小。
   2. ResBlock 使网络结构加深
   3. 边缘补齐不是补0，而是采用`Reflection Pad`补齐方法：上下左右反射边缘的像素补齐。
   4. 上采样不用`ConvTransposed2d`而是先`Unsample` 再用`Conv2d`，这样可以避免`ChackerBoard Artifacts` 现象。
   5. 全连接层使用卷积层代替，减少参数量、

### 环境

pytorch 0.4.0

`pip install -r requirements.tx`

### 数据

coco2014:http://images.cocodataset.org/zips/train2014.zip

图片保存在`data/coco/`

```Bash
data
 └─ coco
    ├── COCO_train2014_000000000009.jpg
    ├── COCO_train2014_000000000025.jpg
    ├── COCO_train2014_000000000030.jpg
```

### 使用

visdom 可视化`python -m visdom.server`

#### 训练

`python main.py train --use-gpu --data-root=data --batch-size=2`

#### 生成图片

```bash
python main.py stylize  --model-path='transformer.pth' \
                 --content-path='amber.jpg'\  
                 --result-path='output2.png'\  
                 --use-gpu=False
```

#### 完整配置

```bash
	image_size = 256 # 图片大小
    batch_size = 8  
    data_root = 'data/' # 数据集存放路径：data/coco/a.jpg
    num_workers = 4 # 多线程加载数据
    use_gpu = True # 使用GPU
    
    style_path= 'style.jpg' # 风格图片存放路径
    lr = 1e-3 # 学习率

    env = 'neural-style' # visdom env
    plot_every=10 # 每10个batch可视化一次

    epoches = 2 # 训练epoch

    content_weight = 1e5 # content_loss 的权重 
    style_weight = 1e10 # style_loss的权重

    model_path = None # 预训练模型的路径
    debug_file = '/tmp/debugnn' # touch $debug_fie 进入调试模式 

    content_path = 'input.png' # 需要进行风格迁移的图片
    result_path = 'output.png' # 风格迁移结果的保存路径
   
```

#### 效果

![1539277410883](/home/huo/.config/Typora/typora-user-images/1539277410883.png)

![1539277458673](/home/huo/.config/Typora/typora-user-images/1539277458673.png)

### 