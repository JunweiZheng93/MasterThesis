# Master Thesis

This is the repository of my master thesis in KIT.

## TODOs

- [ ] update README
- [ ] upload dataset
- [ ] add Dockerfile

## Platform

All scripts have been tested under Mac OS and Linux.

## Setup

### Manually

All necessary packages can be checked in `requirements.txt`. It's a good practice to use python virtual environment 
to set up the programing environment.

To set up python virtual environment: <br>
Follow the [installation instruction of MiniConda](https://docs.conda.io/en/latest/miniconda.html#) and then run 
the following snippet using your favourite terminal application:
```bash
conda create -n py36 python=3.6
conda activate py36
```

To install all necessary packages:
```bash
cd path_of_project_root
pip install -r requirements.txt
```

### Automatically (using docker)

TBA...

## Dataset

We use [ShapeNetCore](https://shapenet.org/) as our dataset. As for the segmented label, we use 
[shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip), 
which is provided by [A Scalable Active Framework
for Region Annotation in 3D Shape Collections](https://cs.stanford.edu/~ericyi/project_page/part_annotation/).

### Prepare dataset from scratch

In order to prepare the dataset from scratch, you need to firstly download 
[ShapeNetVox32](https://cvgl.stanford.edu/data2/ShapeNetVox32.tgz) and 
[shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip). 
ShapeVox32 is the voxelization version of ShapeNetCore, using [binvox](https://www.patrickmin.com/binvox/) for the voxelization. You can also 
voxelize ShapeNetCore using binvox by yourself instead of downloading ShapeVox32. After downloading and unzip those two datasets, run the 
following snippet in your favourite terminal application:
```bash
cd path_of_the_project_root
PYTHONPATH=path_of_project_root python utils/data_preprocessing.py path_of_pcd_category path_of_binvox_category output_path
```
For example, `./shapenetcore_partanno_segmentation_benchmark_v0/03001627/` is the pcd path of the category `chair`. 
`./ShapeNetVox32/03001627/` is the binvox path of the category `chair`. Output path is the path where you want to save the 
generated data.

For more usage about `data_preprocessing.py`, please run:
```bash
PYTHONPATH=path_of_project_root python utils/data_preprocessing.py -h
```

### Using our dataset

chair: [03001627.zip(681MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03001627.zip?inline=false) <br>
number of shapes: 3746 <br>
maximal number of parts: 4

table: <br>
number of shapes: 5263 <br>
maximal number of parts: 3

airplane: [02691156.zip(278MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/02691156.zip?inline=false) <br>
number of shape: 2690 <br>
maximal number of parts: 4

lamp: [03636649.zip(200MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03636649.zip?inline=false) <br>
number of shapes: 1546 <br>
maximal number of parts: 4

### Visualize dataset

The generated dataset consists of some PNG images and `.mat` files. To visualize `.mat` files, please run:
```bash
PYTHONPATH=path_of_project_root python utils/visualization.py path_of_mat_file
```

## Training

TBA...
