# pconv-lmser
Final project for one course, using Nvidia's partial convolution as backbone and check if multiple reflections in the perception phase(one approximate implementation for lmser) will contribute to the result of image inpainting.  

# Install Requirements
Python 3.6+  
pip install -r requirements.txt  

# Usage  
## Preprocess  
1. Download any image set. Create a big folder which should contain "data_large" and "val_large" as subdirectories.  
2. Generate masks randomly by following code:  
```python
python generate_data.py 
```  
You can control mask size, numbers of masks ,directory to save by using args:  
```python
--image_size 'your image size'
--N 'number of masks'
--save_dir 'directory to save'
```  
## Train  
```python
python train.py
```  
You shall specify the directory of dataset ,mask set and interval model by using args as following:  
```python
--root 'your dataset directory'
--mask_root 'your mask set directory'
--save_dir 'directory to save interval model' #the interval model is ended by .pth
```  
Change args to control traning parameters:  
```python 
--lr 'your learning_rate'
--lr_finetune 'learning_rate at finetune state'
--max_iter 'iterations to run at most'
--save_model_interval 'save interval model per save_model_interval iterations'
--vis_interval 'save interval images per vis_interval iterations'
--image_size 'your image size'
```  
## Finetune
```python
python train.py --finetune --resume <interval_model_name>
```  
## Test
```python
python test.py --snapshot <interval_model_name> --mask_root <mask directory> --root <image set directory>
```
## Metrics

You can use psnr and ssim to check the model.(Assure you've saved ground truth and image generated by the net):  
```python 
python eva.py
```
The origin code is used for evaluate all images in test set. If you only want to calculate 2 metrics for one singe image, then just modify the original and contrast directory in the code.

## Perception phase
Modify number of iterations in ```net.py``` to observe whether multiple reflections can affect inpainting results.
