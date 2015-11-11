## Running the code
To run the code clone the repository

```
git clone xxxx
```

`cd` to the `datasets` subfolder and run `create_dataset.py`. This will create the [labeled faces in the wildt dataset](http://vis-www.cs.umass.edu/lfw/). This may take a while depending on your internet connection etc.

Then run

```
th train_lfw.lua -g 0
```

where `-g 0` specifies the GPU you want to use. The code will only run on GPU, but you can esily modify to run on CPU by removing the cudnn dependencies.


##### dependencies
 *  Torch
 *  numpy
 *  skimage
 *  h5py
 *  [cudnn for torch](https://github.com/soumith/cudnn.torch)
