# JPEG Algorithm Implementation
In this project I have implemented similar techniques to compress and thereby reduce the size of any PNG image and thereby transforming it into a Jpeg image. The project is implemented in Python and uses the following libraries:
- Numpy
- OpenCV
- Matplotlib
- Scipy

The project is divided into two parts:
1. JPEG Compression
2. JPEG Decompression, so that we can view the compressed image.

## Get Started
To get started with the project, you need to have the following installed on your machine:
- Python 3.11.10 or higher
- The project's dependencies as mentioned above

As you may have noticed I have a Jupyter Notebook for the project, which is the main file for the project. Even though the Notebook is composed of only 1 cell, it allowed me to write and easily test my implementations. In case you do not have Jupyter Notebook installed, you can also copy its contents into a normal .py file and run it using Python.

---

Before you execute the code you have to make sure that the image you would like to compress is in the same directory as the Jupyter Notebook or the .py file and the image should be called "OriginalImage.png". When you execute the code, you must have will be asked for the compression ratio for the DCT. The higher the ratio, the more coefficients of the image will be removed (set to zero). Afterward, you will be asked to the amount of bits you would like to use for the quantization. The lower the number of bits, the lower the amount of colors available and thereby fewer data. After doing so, the compressed image will be saved as "OutputImage.jpg" in the same directory.

Note that for the Image that I have used while implementing, I noticed the missing colors when setting the Quantization Bits input to 3 and below. 
