# Face Landmarks 2D Detector
CNN for face landmarks detection with training scripts in focus of robustness against bounding box size and shape
<img src="ezgif-2-7fa0691c7c.gif"/></img>

## Test trained detector on image from 
Make gif like above. You can run crop test on your own image or prepair images from 300W folder.
To do it run 4 and 6 steps from "300W preprocessing" section.
Also, you can find some more validation scripts in Train.ipynb notebook. 
```buildoutcfg
python testtools.py --h5 vanilla-68/checkpoints/model_final.h5 --img 300W/test_crop/helen/000000.png
```

## 300W preprocessing
1. Download datasets (HELEN, LFPW, AFW, IBUG, 300W all parts) from [official site](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
2. Put train sets of HELEN LFPW, AFW, 300W to 300W/train
3. Put test sets of HELEN LFPW, AFW and all IBUG to 300W/test
4. Change working directory:

````
cd 300W
````

5. To examinate dataset crops run:    

````
python extract_300w.py train train_crop --max_size 256 --debug
````

6. To created cropped data run for train and test:

````
python extract_300w.py train train_crop --max_size 256
python extract_300w.py test test_crop --max_size 256
````

## Visualize data

```
python vis_data.py 300W/train_crop
python vis_data.py 300W/test_crop
```

## Training landmarks detector
