# Liver_Tumor_Segmentation

## LIVER SEGMENTATION USING ACTIVE CONTOUR 
This repository contains an implementation of a liver segmentation algorithm using active contour models, also known as snakes. The algorithm aims to accurately segment the liver from abdominal CT scans, providing an efficient and automated solution for liver disease diagnosis and treatment planning.

## Data Set Description
The algorithm's performance is evaluated using two publicly available datasets: Liver Tumor Segmentation Challenge (LiTS) and 3D Image Reconstruction for Comparison of Algorithm Database (3D-ircadb-01). The LiTS dataset comprises 130 CT scan images, while the 3D-ircadb-01 dataset includes 20 3D CT scans.

## Pipeline Development
The liver segmentation pipeline follows the steps outlined below:

- Data Import: CT scan images are imported, and a range of slices containing the liver is identified. Every 10th CT slice within this range is selected for analysis.

- Image Preprocessing: Imported images undergo image preprocessing, involving the application of mean, median, and Gaussian 3x3 filters. The goal is to reduce noise while maintaining image clarity. Histogram equalization is identified as the most effective contrast enhancement method.

- Contour Initialization: Contours are initialized using the thresholding method, transforming the grayscale images into appropriate formats for further processing.

- Contour Deformation: A snake algorithm, an active contour model, is applied for contour deformation. The algorithm iteratively adjusts the contour to fit the liver's boundary.

## Results and Discussion
The algorithm's accuracy is assessed through various metrics, including the Dice similarity coefficient (Dice) and mean squared error (MSE). The Dice coefficient indicates the spatial overlap between segmented and ground truth results. The algorithm achieves an average Dice coefficient of 91.8%, with exceptional segmentation accuracy for simpler liver shapes.

The MSE, which measures pixel-wise differences between segmented and ground truth images, is exceptionally low at 0.0375. This low value indicates the algorithm's ability to closely match the ground truth.

## Algorithm Performance and Challenges
The algorithm performs best on CT slices where the liver's shape is simple and well-defined. As liver complexity increases and it interfaces with neighboring structures, segmentation quality decreases. Factors influencing this include the algorithm's internal energy parameters (alpha and beta) and the quality of image gradients. Challenges arise when liver edges are close to or share characteristics with surrounding structures.

## Conclusion and Future Directions
The developed active contour algorithm demonstrates its potential for accurate liver segmentation from CT scans. With an average Dice coefficient of 91.8% and a small error in liver volume estimation (5.8%), the algorithm proves effective. Future work could involve iterative parameter adjustment for the snake algorithm to enhance performance.

In summary, this repository presents an efficient and viable solution for liver segmentation from CT scans using active contour models. The algorithm's integration into computer-aided diagnosis systems can improve accuracy and efficiency, ultimately benefiting liver disease diagnosis and treatment planning, and subsequently, patient outcomes. Further advancements can focus on refining the algorithm's accuracy and extending it to segment other organs simultaneously.

## Results 

<img width="200" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/889a1204-d6da-4277-adc6-bc84befe88c3">
<img width="243" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/0e912c2b-205b-4e07-91f8-98a1c5308b79">
<img width="216" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/9f6f46d0-7b28-47ee-ae91-4308ebb91ec8">




Note: This README provides a concise overview of the project. For detailed methodologies, code implementations, and complete analysis, refer to the repository's code and accompanying documentation.
