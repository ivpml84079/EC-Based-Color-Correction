# Error Compensation-Based Fusion Algorithm for Drone-image Color Correction
"Error Compensation-Based Fusion Algorithm for Drone-image Color Correction" by Kuo-Liang Chung and Te-Wei Hou.

## Introduction
![flowchart](https://github.com/user-attachments/assets/24eb70bb-28d9-49fd-8006-f3d13647a7e3)


Color correction for multiple images.

An example of our algorithm's effect is demonstrated below:
![alt text](sample.png)


## Dependencies
+ C++ 17
+ OpenCV 4.7.0
+ OpenMP (optional)

## Environment
+ Intel Core i7-12700F CPU clocked at 4.8 GHz, 32 GB RAM
+ NVIDIA GeForce RTX 3080 GPU (optional)
+ Microsoft Windows 10 64-bit operating system
+ Microsoft Visual Studio 2022

## Configuration & Run Tests
The project can be configured by CMake with the given CMakeLists.txt file.
Four input directories are required for program execution:
+ "aligned_result": The aligned images obtained after applying an existing method. The image file "XX__warped_img.png" represents the aligned image for XX.
+ "overlap" : The overlapping areas of the aligned images in "aligned_result". The image file "XX__YY__overlap.png" represents the overlapping area between images XX and YY. ("YY__XX__overlap.png" is the same as "XX__YY__overlap.png").
+ "correspondence" : The matching feature points of the aligned images in "aligned_result". The CSV file "XX__YY_inliers.csv" contains the x-y coordinates of the matching feature points on images XX and YY.
+ "img_masks" : The masks of the images after alignment. The image file "mask_XX.png" represents the mask for the aligned image "XX__warped_img.png".

Two output directories are required for saving the color corrected results:
+ "Ours": The color corrected images of "aligned_result" are saved in this directory.
+ "result": The color corrected aligned multiple images of "Ours" is saved in this directory.

All 30 testing multiple images and the qualitative quality results are available at https://drive.google.com/drive/u/3/folders/1avBu1zL5PUl8SsLXG1Ot4XWGgcLRKw4t

## Contact
If you have any questions, please email us via

Te-Wei Hou: deweihou123@gmail.com

Kuo-Liang Chung: klchung01@gmail.com
