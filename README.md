# Data-Science-Bowl-2017
My shot at the Data Science Bowl Challenge 2017 using SPIE-AAPM Lung CT Dataset.

# Background
Lung Cancer is a heterogenous and aggressive form of cancer and is the leading cause of cancer death in men and women, accounting for etiology of 1 in every 4 cancer deaths in the United States. There were 224,000 new cases of lung cancer and 158,000 deaths caused by lung cancer in 2016.

The lifetime likelihood that a man will develop lung cancer in his lifetime is 1 in 14, whereas the risk for a woman is 1 in 17 in her lifetime.

The primary method in use by physicians to screen for lung cancer is radiographic imaging of the Chest using Computed Tomography (CT) scans. This imaging modality makes use of many X-ray images to create a 3-dimensional representation of a patient's chest cavity. Unfortunately, Chest CT scans expose patients to a high level of radiation, on the order of 100-500 times the amount of radiation from a single x-ray.

Given the high prevalence of lung cancer screening and the harmful effects of excessive repeat radiation exposure, computational machine learning techniques have the potential to aid radiologists in their ability to spot lung nodules/tumors and minimize radiation exposure to patients. As Chest CT scans are representations of three-dimensional objects, I use a three-dimensional convolutional neural network to classify whether an area of the lung is likely to be healthy or a lung nodule/tumor.

# Overview

The approach taken by me consists of the following steps. For a more detailed explanation of what I did, you can read the [Pipeline](code/Pipeline.ipynb) file in the code folder

1. I utilized the data from the [SPIE-AAPM Lung CT Challenge Dataset](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge). I extracted the metadata corrosponding to each Patient in [extracting_metadata](code/extracting_metadata.py) .

2. I preprocessed and generated the training data i.e two 32 X 32 X 32 cube from each patient, one having the nodule and one not having the nodule in [extracting_data](code/extracting_data.py).

3. I used different data augmentation techniques like rotation, translation, scaling etc to generate more data and implement a varient of [C3D architecture](https://arxiv.org/pdf/1412.0767.pdf) to learn spatiotemporal features for better prediction. The corrosponding code for it is in [neural_network](code/neural_network.py) file.

4. In the [predict_nodule](code/predict_nodule.py) file, I convolved through the whole 3D test image and extracted 32 X 32 X 32 cubes from the test image and predicted the probability of it having a nodule and it's malignancy for each cube and stored it in a csv file for each patient.

5. I used the prediction generated for each patient in [display_results](code/display_results.py) to make a gif file for each patient highlighting the malignant nodule.

# Dependencies

pandas==0.23.4
pydicom==1.2.2
Keras==2.2.4
matplotlib==2.2.2
scipy==1.1.0
moviepy==0.2.3.5
scikit_image==0.14.0
opencv_contrib_python==4.0.0.21
tensorflow_gpu==1.13.1
numpy==1.14.3

# Installation

1. Download and install jupyter notebook. Visit this link for [installation instructions](https://jupyter.readthedocs.io/en/latest/install.html)

2. Install all the dependencies given in the [requirements.txt](code/requirements.txt).
            **pip install -r requirements.txt**

# Results

                                ![](results/ct-training-be001)
