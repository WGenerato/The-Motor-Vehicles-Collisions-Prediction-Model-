import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.regularizers import l2

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Load the dataset from CSV file
data = pd.read_csv("C:/Users/Tremble/Downloads/Motor Vehicle Collisions with KSI Data - 4326.csv")

# Convert all columns to uniform string types where necessary
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype(str)
    elif data[col].dtype == 'float64' or data[col].dtype == 'int64':
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill NaN values in X and y with appropriate values
X = data.drop(columns=["INJURY", "DISABILITY", "DIVISION", "geometry"])  # Features that are not being considered
y = data["INJURY"]  # Target variable

# Fill NaN values with a placeholder or appropriate values
X_filled = X.fillna('NaN')
y_filled = y.fillna('NaN')

# Map target variable to numeric values
injury_mapping = {
    'Major': 3,
    'Minor': 2,
    'Minimal': 1,
    'Fatal': 4,
    'NaN': 0  # Ensure 'None/nan' is mapped to a value
}

# Apply the mapping, with a fallback value for unmapped categories
y_mapped = y_filled.map(injury_mapping)
y_mapped = y_mapped.fillna(0).astype(int)  # Replace NaN values with 0 and convert to integer

# Separate numerical and categorical columns
numeric_cols = X_filled.select_dtypes(include=[np.number]).columns
categorical_cols = X_filled.select_dtypes(include=[object]).columns

# Convert all categorical columns to strings to ensure uniform data type
for col in categorical_cols:
    X_filled[col] = X_filled[col].astype(str)

# Define column transformers for numerical and categorical data separately
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Apply transformers to numerical and categorical columns separately
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X_filled)
y_processed = y_mapped

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# One-hot encode the target variables y_train and y_test
num_classes = len(np.unique(y_processed))  # Number of unique classes in y_processed
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')  # Softmax activation for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_one_hot, epochs= 35, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)

print("Test Accuracy:", test_accuracy)
print("Loss:", test_loss)