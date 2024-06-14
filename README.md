# Classification-Clustering-Project
Evaluate classification mechanisms using the machine learning library Scikit Learn on the “Sleep Health and Lifestyle Dataset” from Kaggle. 

Evaluation metrics: accuracy, precision, and F1-score to measure the quality of the classification models

Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) were used for this project. The data was preprocessed by encoding categorical columns ('Gender', 'Occupation', 'BMI Category', and 'Sleep Disorder') using LabelEncoder. The 'Person ID' column serves as an identifier so it was removed because it does not contribute to the classification task. The 'Blood Pressure' column was split into 'Systolic' and 'Diastolic' columns to extract numerical values. The numeric columns were then scaled using StandardScaler. For SVM, I used a rbf kernel because rbf can be used for a wide range of classification tasks. For KNN, I chose the number 10 as the n_neighbors
