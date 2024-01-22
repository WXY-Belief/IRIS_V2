# IRIS_V2
## 1.Introduction
This repository is an upgraded version of **IRIS（https://github.com/th00516/ISS_pyIRIS）**, designed with the primary functionality of analyzing fluorescence data generated by ISS to obtain RNA coordinates.
The corresponding article is https://doi.org/10.1101/2020.04.13.038901, primarily designing for the analysis of data from four rounds and four channels.  

**The improvements are as follows**
- **Save the required input parameters using a YAML file to make the code more user-friendly.**
- **Resolved the issue of slow alignment for large images, significantly speeding up the runtime.**
- **Added detection for reuse points, making the results more accurate and clear.**
- **Added many files for debugging purposes.**
## 2.Prerequisites
To install requirements:  
```
pip install -r requirements.txt
```  
- Python 3.8.13  
## 3.Tutorial
### Directory structure of input data
The names of cycles and channels can be configured in the Configuration.yaml.
- data
  - 1 
    - channel 0.tif
    - channel 1.tif
    - channel 2.tif
    - channel 3.tif
    - channel 4.tif
  - 2 
    - channel 0.tif
    - channel 1.tif
    - channel 2.tif
    - channel 3.tif
    - channel 4.tif
  - 3
    - channel 0.tif
    - channel 1.tif
    - channel 2.tif
    - channel 3.tif
    - channel 4.tif
  - 4
    - channel 0.tif
    - channel 1.tif
    - channel 2.tif
    - channel 3.tif
    - channel 4.tif
### Input parameter  

The Configuration.yaml include all parameter in runing. 

- `data_path`: absolute path of data path.

- `output_path`: absolute path of output path.
  
- `iris_path`: absolute path of IRIS path.

- `register_mode`: register mode. **# Option: "ORB", "BRISK"**.

- `mode`: whether to cut images when images is too large. **# Option: com:NO large:YES**.

- `large_img_para`: First cutting
  - `cut_size` ": the size of cut img
  - `overlap` ": the overlap size of cut img

- `small_img_para`: Second cutting
  - `cut_size` ": the size of cut img
  - `overlap` ": the overlap size of cut img

- `core_num`: the number of CPU core

- `cycle`: cycle name. Supporting for single cycles, two cycles and three cycles.
  - 1 : the name of cycle 1
  - 2 : the name of cycle 2
  - 3 : the name of cycle 3
  - 4 : the name of cycle 4

- `channel`: channel. Only Supporting for four channels(one of A,T,C and G is used as channel 0) and five channels.
  - 0 : the name of channel 0. Reference images in registering.
  - A : the name of channel A
  - T : the name of channel T
  - C : the name of channel C
  - G : the name of channel G
  
- `barcode_path`: gene-barcode file, txt file.

- `search_region`: the size of search region  in connect base on different cycle. **# Option: 1, 2, 3, 4, ... **.

- `temp_flag`: whether to generate debug file. # Option: 0:NO 1:YES.

- `blob_params`: Second cutting
  - `minThreshold` ": The threshold of pixel filtered. # Option: "AUTO", 0, 1, ..., 10.
  - `maxArea` ": The maximum area of point. # Option: 65, 121, 145

### Format of input data(barcode)
1) barcode_path: gene-barcode infomation. 
<div align="center">
  
| gene | barcode |
| ------- | ------- |
| gene 1 | ATCG |
| gene 2 | AATG |
| gene 1 | GCAG |

</div>  

### Running
```
python main.py --c <<absolute path of Configuration.yaml>>
```
### Output result
- `valid_basecalling_data.csv` :
  - `ID` : RNA ID is unique.
  - `barcode` : barcodes corresponding to genes.
  - `quanlity,  1, 2, 3, 4 ` : This used to be debug.
  - `row` : The row coordinates of RNA.
  - `col` : The col coordinates of RNA.
  - `mul_flag` : Whether it is a multiplexed RNA, meaning a point in one cycle and one channel is used by multiple RNAs.
  - `gene` : gene name.
<div align="center">
  
| ID | barcode | quanlity | row | col | 1 | 2 | 3 | 4 | mul_flag | gene |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| r00014c00015 | ATCG | !I!! | 2214.0 | 865.0 | N | r00014c00015 | N | N | 0 | Cdhr1 |
| r00074c00380 | ATTG | !D!! | 2387.0 | 1234.0 | N | r00351c00384 | N | N | 1 | Cdhr1 |
| r00247c00031 | TTCG | I'!! | 2210.0 | 868.0 | N | N | N | N | 0 | Cdhr2 |

</div>  


## 4.A solution for a specific BUG.
**1) pyvips**  
- **For Windows**  
    if you encounter an issue with "pyvips" not importing.Please download the "vips"  package from https://github.com/libvips/libvips/releases and add the installation path into **environment variables**.  
- **For Linux**  
    To install "pyvips" using conda and then adding the installation path into **environment variables**.You can find detailed installation instructions at https://anaconda.org/conda-forge/pyvips.  


