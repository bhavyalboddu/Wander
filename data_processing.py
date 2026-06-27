import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

COLUMNS_TO_KEEP = [
    'category', 'families', 'couples', 'solo', 'child-friendly', '21+',
    'cost', 'ada_accessibility', 'interactivity', 'atmosphere',
    'estimated_duration', 'reservation_needed', 'rare_find', 'person_responses'
]

def load_data(csv_path='wander_training_data.csv'):
    df = pd.read_csv(csv_path)
    return df[COLUMNS_TO_KEEP]

def split_data(full_data_set, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(
        full_data_set, test_size=test_size, random_state=random_state
    )

    locations_train = train_set.drop("person_responses", axis=1)
    train_labels = train_set["person_responses"].copy()
    locations_test = test_set.drop("person_responses", axis=1)
    test_labels = test_set["person_responses"].copy()

    return locations_train, train_labels, locations_test, test_labels

def build_pipeline(data_cat):
    cat_pipeline = Pipeline([
        ('imputer2', SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    cat_attribs = list(data_cat)

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline

def prepare_data(csv_path='wander_training_data.csv'):

    full_data_set = load_data(csv_path)
    locations_train, train_labels, locations_test, test_labels = split_data(full_data_set)

    data_cat = full_data_set.select_dtypes(include='object')
    full_pipeline = build_pipeline(data_cat)

    train_prepared = full_pipeline.fit_transform(locations_train)
    test_prepared = full_pipeline.transform(locations_test)

    return train_prepared, test_prepared, train_labels, test_labels, full_pipeline