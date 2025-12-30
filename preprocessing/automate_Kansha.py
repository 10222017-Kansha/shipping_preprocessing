from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import fire

def preprocess_data(data, target_column, save_path):
    # Menentukan fitur numerik dan kategoris
    data = data.drop(columns=['ID'])
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Menghapus kolom yang missing values nya lebih dari 75% total dataset
    missing_values = data.isnull().sum()
    over = missing_values[missing_values >= ((3/4)*data.shape[0])].index
    data = data.drop(columns=over)

    # Menghapus baris yang terduplikat
    duplicate = data.duplicated().sum()
    if duplicate.any():
        data = data.drop_duplicates()

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder(handle_unknown='ignore'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test = preprocessor.transform(X_test)

    # Menggabungkan menjadi satu dataset
    X_train_final = pd.DataFrame(X_train, index=y_train.index)
    shipping_train = pd.concat([X_train_final, y_train], axis=1)
    X_test_final = pd.DataFrame(X_test, index=y_test.index)
    shipping_test = pd.concat([X_test_final, y_test], axis=1)

    # Export to csv
    shipping_train.to_csv(f"{save_path}/train.csv")
    shipping_test.to_csv(f"{save_path}/test.csv")

if __name__ == '__main__':
    fire.Fire(preprocess_data)
