# Smart-Home-Control-v2.0
## Introduction
Update it later

## Dependence
dlib, Cmake, boost, NumPy, sklearn

## Face Recognition Install
The installation of face_recognition require dlib, Cmake and boost. You can install it by using the package which in face_recognition folder.
### Install Cmake
Install cmake-3.16.4-win64-x64.msi
 
###	Install dlib
Unzip dlib-19.19.zip to any path you want.
1.	Use power shell switch the environment where you want to install like ‘conda activate xxx (your environment name)’. Then use the cd command to enter the path you just unzipped.
2.	```python setup.py install```
###	Install boost
Unzip boost_1_72_0.zip to any path you want
1.	Double click bootstrap.bat
2.	Switch environment and path
3.	``` b2 install```
4.	```b2 -a --with-python address-model=64 toolset=msvc runtime-link=static```
###	Install face_recognition
``` pip install face_recognition ```
## Required documents
Unzip 3rdparty and model to project root path
```https://drive.google.com/file/d/1ma8e1EdU89hKlzfItkMzA88n3_eEFTRe/view?usp=sharing``` ---face_recognition package
```https://drive.google.com/file/d/1vfRpQ3feUDNvEzO1CtgvnvSKcgmyvphy/view?usp=sharing``` ---3rdparty and models
## Reference
https://github.com/CMU-Perceptual-Computing-Lab/openpose

https://github.com/ageitgey/face_recognition
