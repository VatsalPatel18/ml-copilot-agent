# Data Preprocessing

The agent assists in preprocessing your dataset to prepare it for model training.

## Features

- **Missing Value Handling**: Detect and handle missing data.
- **Categorical Encoding**: Convert categorical variables into numerical form.
- **Scaling/Normalization**: Scale numerical features appropriately.

## Usage

```plaintext
> preprocess data
Please enter the dataset path:
> data/dataset.csv
Please enter the target column name:
> target
Any additional preprocessing instructions?
> use standard scaler

The preprocessed data is saved to data/preprocessed_data.csv by default.
