### hloc节点

#### 参数设置
仿照`hloc.launch`对数据集的根目录、数据集名称、以及拍摄RGB图像的相机话题进行设置。

目录格式:
```
${root}
├── data
│   └── ${dataset_name}
│       ├── db
│       └── query
├── outputs
│   └── ${dataset_name}
│       ├── sfm_aligned
│       ├── *.h5
│       ├── *.txt
```


#### 使用
`shfiles`下面两个文件

```sh
sh camera_localize.sh
```
通过订阅相机话题获取当前图像, 并使用hloc进行定位

```sh
sh query_localize.sh [path]
```
通过指定图像地址对图像进行定位


#### 其他
`component`下面是经过修改的hloc文件, 所以原来hloc的文件与环境不需要修改