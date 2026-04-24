"""#BI 1"""

import pandas as pd
import pyodbc
import cx_Oracle

# Excel Source
excel_df = pd.read_excel('data.xlsx')

# SQL Server Source
sql_conn = pyodbc.connect(
    'DRIVER={SQL Server};SERVER=localhost;DATABASE=testdb;UID=sa;PWD=1234'
)
sql_query = "SELECT id, name, salary FROM employees"
sql_df = pd.read_sql(sql_query, sql_conn)

# Oracle Source
oracle_conn = cx_Oracle.connect("user/password@localhost:1521/xe")
oracle_query = "SELECT id, name, salary FROM employees"
oracle_df = pd.read_sql(oracle_query, oracle_conn)

# Combine Data
combined_df = pd.concat([excel_df, sql_df, oracle_df], ignore_index=True)

# Basic Cleaning
combined_df.drop_duplicates(inplace=True)
combined_df.fillna(0, inplace=True)

print("Combined Data:")
print(combined_df.head())


"""#BI 2"""

import pandas as pd
import matplotlib.pyplot as plt

# Extract
df = pd.read_csv('sales_data.csv')

# Transform
df.dropna(inplace=True)
df['sales'] = df['sales'].astype(float)
df['sales'] = df['sales'] * 1.1  # example transformation

# Load (optional save)
df.to_csv('cleaned_sales.csv', index=False)

# Visualization
plt.figure()
plt.plot(df['month'], df['sales'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()

plt.figure()
df.groupby('region')['sales'].sum().plot(kind='bar')
plt.title("Region-wise Sales")

plt.show()


"""#BI 3"""

import pandas as pd
import pyodbc

# Extract
df = pd.read_csv('employee.csv')

# Transform
df.dropna(inplace=True)
df['salary'] = df['salary'].astype(float)
df['salary'] = df['salary'] + 2000

# Load
conn = pyodbc.connect(
    'DRIVER={SQL Server};SERVER=localhost;DATABASE=testdb;UID=sa;PWD=1234'
)
cursor = conn.cursor()

# Create table (if not exists)
cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees')
CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    salary FLOAT
)
""")

# Insert data
for index, row in df.iterrows():
    cursor.execute(
        "INSERT INTO employees (id, name, salary) VALUES (?, ?, ?)",
        int(row['id']), row['name'], float(row['salary'])
    )

conn.commit()
conn.close()

print("Data Loaded Successfully")


"""#BI 4"""

import pandas as pd
import matplotlib.pyplot as plt

# Load Excel
df = pd.read_excel('sales.xlsx')

# Data Analysis (Pivot-like)
pivot = pd.pivot_table(
    df,
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum'
)

print("Pivot Table:")
print(pivot)

# Visualization
pivot.plot(kind='bar')
plt.title("Sales by Region and Product")
plt.xlabel("Region")
plt.ylabel("Sales")

# Export result
pivot.to_excel('analysis_output.xlsx')

plt.show()


"""#BI 5"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('classification.csv')

# Features & Target
X = df[['age', 'salary']]
y = df['purchased']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test with new data
new_data = [[30, 50000]]
prediction = model.predict(new_data)
print("Prediction for new data:", prediction)


"""#DL 1"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load CSV
df = pd.read_csv('USA_Housing.csv')

# Preprocessing
df = df.dropna()

X = df[['Avg. Area Income', 'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population']]
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, pred))

# Test sample
print(model.predict([[60000, 5, 7, 3, 30000]]))


"""#DL 2"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load CSV
df = pd.read_csv('mnist_csv.csv')

# Split features & label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize
X = X / 255.0

# Reshape to 28x28 image
X = X.reshape(-1, 28, 28, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3)

# Evaluate
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

print("Confusion Matrix:\n", confusion_matrix(y_test, pred_classes))


"""#DL 3"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load CSV
df = pd.read_csv('reviews.csv')

# Preprocessing
X = df['review']
y = df['sentiment']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# Model
model = Sequential([
    Embedding(5000, 64, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)


"""#DL 4"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv('image_data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize
X = X / 255.0

# Reshape (example 32x32 RGB)
X = X.reshape(-1, 32, 32, 3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)


"""#DL 5"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load CSV
df = pd.read_csv('sentiment.csv')

X = df['text']
y = df['sentiment']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# Model
model = Sequential([
    Embedding(5000, 64, input_length=100),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)