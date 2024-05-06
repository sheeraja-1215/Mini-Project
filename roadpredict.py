import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'cleaned.csv'
data = pd.read_csv(file_path)

# Preprocessing: Encoding categorical variables
data = pd.get_dummies(data, columns=['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 
                                     'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                                     'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                                     'Pedestrian_movement', 'Cause_of_accident'])

# Split the dataset into features and target variable
X = data.drop(columns=['Accident_severity'])
y = data['Accident_severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the severity of accidents for the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

column='Accident_severity'
# Plot histograms for column
plt.figure(figsize=(8, 6))
data[column].hist()
plt.title('Accident Severity')
plt.xlabel('Scale of Severity')
plt.ylabel('Occurence Rate')
plt.show()