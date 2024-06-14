import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Load the dataset
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

#Convert categorical columns to numerical values using LabelEncoder
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])
data['Sleep Disorder'] = label_encoder.fit_transform(data['Sleep Disorder'])

#Drop Person ID
data = data.drop(columns=['Person ID'])

#Separate features (X) and target variable (y)
X = data.drop(columns=['Sleep Disorder'])
y = data['Sleep Disorder']

#Split 'Blood Pressure' column into two separate columns 'Systolic' and 'Diastolic' to get BP format
X[['Systolic', 'Diastolic']] = X['Blood Pressure'].str.split('/', expand=True).astype(float)
X.drop(columns=['Blood Pressure'], inplace=True)

#Select only numeric columns for scaling and store in numerical_columns
numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep',
                     'Physical Activity Level', 'Stress Level',
                     'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps']


#Create a new DataFrame with scaled numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_columns])

#Replace the original columns with the scaled columns in the DataFrame
X[numerical_columns] = X_scaled

#Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SVM Classifier using rbf kernel
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions_svm = svm_model.predict(X_test)

#KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

#Calculate accuracy, precision, anf F1 score for SVM and KNN
svm_accuracy = accuracy_score(y_test, svm_predictions_svm)
svm_precision = precision_score(y_test, svm_predictions_svm, average='weighted', zero_division=1)  # Add zero_division to handle zero precision case
svm_f1 = f1_score(y_test, svm_predictions_svm, average='weighted')

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted', zero_division=1)  # Add zero_division to handle zero precision case
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

#Print the results for SVM with 2 decimals
print("SVM Classifier")
print("---------------")
print(f"Accuracy: {svm_accuracy:.2f}")
print(f"Precision: {svm_precision:.2f}")
print(f"F1-score: {svm_f1:.2f}")

#Print the results for KNN with 2 decimals
print("\nKNN Classifier")
print("---------------")
print(f"Accuracy: {knn_accuracy:.2f}")
print(f"Precision: {knn_precision:.2f}")
print(f"F1-score: {knn_f1:.2f}")


