# Liver_Tumor_Segmentation

## LIVER SEGMENTATION USING ACTIVE CONTOUR 
This repository contains the implementation and evaluation of a liver segmentation algorithm using deformable models. The algorithm's performance is assessed using publicly available liver CT scan datasets. The primary goal is to provide a comprehensive analysis of the algorithm's effectiveness and limitations.

## Data Set Description
The liver segmentation algorithm is evaluated on two distinct datasets:

- Liver Tumor Segmentation Challenge (LiTS): This dataset comprises 130 CT scan images. It serves as a crucial source of data for evaluating the algorithm's performance.
- 3D Image Reconstruction for Comparison of Algorithm Database (3D-ircadb-01): This dataset consists of 20 3D CT scans, providing additional depth for a comprehensive assessment.
## Pipeline Development
The algorithm's segmentation pipeline involves the following key steps:

1. Data Import: CT scan data is imported, and liver-occupied slices are identified within the dataset.
2. Slice Selection: Every 10th CT slice is selected within the liver-occupied slice range (slices 280 to 430).
3. Image Preprocessing: The imported images undergo image preprocessing, including mean, median, and Gaussian 3x3 filter applications. This step aims to reduce noise without compromising image quality.
4. Contrast Enhancement: Histogram equalization is identified as the most effective contrast enhancement method, enhancing liver boundary definition and overall segmentation quality.
5. Segmentation: The segmentation algorithm, based on active contours, is applied to the preprocessed images.
## Discussion of Results
The evaluation of the algorithm's performance involves several key metrics:

- Dice Similarity Coefficient: A metric quantifying spatial overlap between segmented and ground truth results. The algorithm achieves a high average Dice coefficient of 91.8% for liver segmentation, comparable to existing studies.
- Mean Squared Error (MSE): A measure of the difference between segmented and ground truth images. The algorithm attains a low average MSE of 0.0375, indicating accurate segmentations.
- Segmentation Performance: The algorithm's effectiveness varies based on the complexity of the liver's shape and its proximity to surrounding structures. Performance is observed to decline for more complex liver shapes due to internal energy considerations and edge detection challenges.
- Liver Volume Estimation: The algorithm calculates liver volumes and achieves a slight 5.8% error when compared to ground truth volumes. This error rate is competitive with related studies.
## Conclusion and Future Work
The liver segmentation algorithm, based on active contours, demonstrates strong performance across various metrics. It successfully segments livers with well-defined shapes and performs adequately even for complex shapes. The algorithm's strengths and limitations, particularly its sensitivity to liver complexity, are thoroughly discussed. Future enhancements could focus on refining energy terms and contour initialization to address segmentation challenges for complex liver shapes.

## Usage Instructions
To reproduce the results and insights presented in this repository, follow the steps outlined in the provided code files. Ensure that the necessary datasets are obtained from the references provided (LiTS and 3D-ircadb-01). The code includes implementations for preprocessing, contrast enhancement, active contour segmentation, and performance metric calculations.

## Result comparison 

<img width="200" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/889a1204-d6da-4277-adc6-bc84befe88c3">
<img width="243" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/0e912c2b-205b-4e07-91f8-98a1c5308b79">
<img width="216" alt="image" src="https://github.com/prateek2468/Liver_Tumor_Segmentation/assets/69041894/9f6f46d0-7b28-47ee-ae91-4308ebb91ec8">




Note: This README provides a concise overview of the project. For detailed methodologies, code implementations, and complete analysis, refer to the repository's code and accompanying documentation.
