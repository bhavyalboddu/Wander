import pandas as pd
import data_processing
from tested_models import logistic_regression
import anvil.server
import json

COLUMNS_TO_KEEP = [
    'category', 'families', 'couples', 'solo', 'child-friendly', '21+',
    'cost', 'ada_accessibility', 'interactivity', 'atmosphere',
    'estimated_duration', 'reservation_needed', 'rare_find', 'person_responses'
]

ORIGINAL_COLUMNS = [
    'city_name', 'activity_name', 'activity_description',
    'activity_location', 'latitude', 'longitude', 'google_rating',
    'category', 'families', 'couples', 'solo', 'child-friendly', '21+',
    'cost', 'ada_accessibility', 'interactivity', 'atmosphere',
    'estimated_duration', 'reservation_needed', 'rare_find',
    'person_responses'
]

train_prepared, test_prepared, train_labels, test_labels, full_pipeline = data_processing.prepare_data()
model, test_acc, report = logistic_regression.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)

cville_df = pd.read_csv(csv_path='charlottesville_data.csv')
cville_data_copy_df = cville_df.copy()
full_data_set = cville_df[COLUMNS_TO_KEEP]
full_data_set.drop('person_responses', axis=1, inplace=True)
cville_test = full_pipeline.transform(full_data_set)
predictions = model.predict(cville_test)
column_names = ORIGINAL_COLUMNS
FINAL_df = pd.DataFrame(columns=column_names)
for x in range(len(predictions)):
  if predictions[x] == 1:
    row_df = pd.DataFrame(cville_data_copy_df.iloc[[x]])
    FINAL_df = pd.concat([FINAL_df, row_df], ignore_index=True)

json_records = json.loads(FINAL_df.to_json(orient='records'))

anvil.server.connect("server_QKJVIENGBSKEVZAI6XH5O2Y6-R3CA5QSEVWVHQ654")
@anvil.server.callable
def get_charlottesville_activities2():
  return json_records
anvil.server.wait_forever()