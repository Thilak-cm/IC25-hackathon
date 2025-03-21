import pandas as pd
from datetime import datetime, timedelta, time
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sys, os

def isSliceCyclc(slice):
#   print(slice.loc[0, 'Start Time - Daily'], slice.loc[0, 'End Time - Daily'])
  if slice.loc[0, 'Start Time - Daily'] == slice.loc[0, 'End Time - Daily']:
    flag = True
  else:
    flag = False

  start = slice.loc[0, 'Start Time - Daily']
#   print(start)
  for idx, row in slice.iloc[1:,:].iterrows():
#     print(row['Start Time - Daily'],row['End Time - Daily'])
    if row['End Time - Daily'] == start:
      flag = True
      break

    if row['Start Time - Daily'] != slice.loc[idx-1, 'End Time - Daily']:
      flag = False
      break

  return flag

def check_cyclic_lots(data, lot_names):
  cyclic_lots_weekdays = []
  non_cyclic_lots_weekdays = []
  cyclic_lots_weekends = []
  non_cyclic_lots_weekends = []


  for lot_name in lot_names:
#     print(lot_name)
    slice = data[(data['Lot Name']==lot_name) & (data['Enforcement Days']=='Weekdays')].reset_index()
    if len(slice)!=0:
      if isSliceCyclc(slice):
        cyclic_lots_weekdays.append(lot_name)
      else:
        non_cyclic_lots_weekdays.append(lot_name)
    slice = data[(data['Lot Name']==lot_name) & (data['Enforcement Days']=='Weekends')].reset_index()
    if len(slice)!=0:
      if isSliceCyclc(slice):
        cyclic_lots_weekends.append(lot_name)
      else:
        non_cyclic_lots_weekends.append(lot_name)
    slice = data[(data['Lot Name']==lot_name) & (data['Enforcement Days']=='Always')].reset_index()
    if len(slice)!=0:
      if isSliceCyclc(slice):
        cyclic_lots_weekends.append(lot_name)
      else:
        non_cyclic_lots_weekends.append(lot_name)
  return cyclic_lots_weekdays, non_cyclic_lots_weekdays, cyclic_lots_weekends, non_cyclic_lots_weekends

def parse_parking_data(df):
    """
    Given a DataFrame 'df' with columns:
        - "Lot Type"
        - "Physical Location (Yes/No)"
        - "Parking Lot / Zone Name"
        - "Posted Restrictions"
        - "Enforcement Days"
        - "Start Time - Daily"
        - "End Time - Daily"
        - ... plus many permit columns (e.g., "17FAE", "A", "AA", etc.)
    Return a nested dictionary in the format:

    {
        lot_name: {
            "Type": <string>,
            "Physical Location": <bool>,
            "Permissions": {
                day_type (e.g. "Weekdays"): {
                    (start_time, end_time): {
                        permit_name: bool,
                        ...
                    }
                }
            }
        },
        ...
    }
    """

    # Identify which columns are permits by excluding known metadata columns
    known_columns = {
        "Lot Type ",
        "Physical Location (Yes/No)",
        "Lot Name",
        "Posted Restrictions",
        "Enforcement Days",
        "Start Time - Daily",
        "End Time - Daily",
        "Count Valid Permissions in Lot by Date/Time",
    }

    # All other columns are presumably permit columns
    permit_columns = [col for col in df.columns if col not in known_columns]

    # Our final nested dictionary
    lots_dict = {}

    for _, row in df.iterrows():
        lot_name = str(row["Lot Name"])
        lot_type = str(row["Lot Type "])
        physical_location_val = str(row["Physical Location (Yes/No)"]).strip().upper()
        # Convert "YES"/"NO" to boolean
        physical_location_bool = (physical_location_val == "YES")

        enforcement_day = str(row["Enforcement Days"]).strip()
        start_time_raw = str(row["Start Time - Daily"]).strip()
        end_time_raw = str(row["End Time - Daily"]).strip()

        # If your dataset uses "0:00:00" to mean midnight, you might
        # want to convert to "00:00" or "24:00" for clarity. For example:
        # start_time = "00:00" if start_time_raw == "0:00:00" else start_time_raw
        # But here, we just keep them as-is or do minimal cleanup:
        start_time = start_time_raw
        end_time = end_time_raw

        # Initialize lot entry if not present
        if lot_name not in lots_dict:
            lots_dict[lot_name] = {
                "Type": lot_type,
                "Physical Location": physical_location_bool,
                "Permissions": {}
            }

        # Prepare to store the permit booleans
        permit_dict = {}
        for pcol in permit_columns:
            val = row[pcol]
            # Convert 1 -> True, 0 -> False (or strings "1"/"0" similarly)
            permit_dict[pcol] = bool(val)

        # Insert into the nested structure
        if enforcement_day not in lots_dict[lot_name]["Permissions"]:
            lots_dict[lot_name]["Permissions"][enforcement_day] = {}

        # Use (start_time, end_time) as a tuple key
        time_tuple = (start_time, end_time)
        lots_dict[lot_name]["Permissions"][enforcement_day][time_tuple] = permit_dict

    return lots_dict

def subtract_one_second(t):
    if t == datetime.min.time():  # Check if it's midnight
        return time(23, 59, 59)  # Use time class to create time object
    else:
        return (datetime.combine(datetime.min, t) - timedelta(seconds=1)).time()
    
def find_non_unique_prefixes(df, column_name='Prefix'):
    """
    Finds and returns a list of non-unique values in the specified column of a DataFrame.
    """
    value_counts = df[column_name].value_counts()
    non_unique_values = value_counts[value_counts > 1].index.tolist()
    return non_unique_values

# Assuming data_hella_new is your DataFrame
# non_unique_prefixes = find_non_unique_prefixes(data, column_name='Prefix')

def is_time_in_tuple(time_str, time_tuple):
    """
    Checks if a time string is within a given time tuple.

    Args:
        time_str: The time string to check (e.g., "08:30:00").
        time_tuple: A tuple containing two time strings representing the start and end times (e.g., ("07:00:00", "16:00:00")).

    Returns:
        True if the time string falls within the time tuple (inclusive), False otherwise.
    """
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
        start_time_obj = datetime.strptime(time_tuple[0], "%H:%M:%S").time()
        end_time_obj = datetime.strptime(time_tuple[1], "%H:%M:%S").time()

        return start_time_obj <= time_obj <= end_time_obj
    except ValueError:
        return False # Handle invalid time string format
    
def extract_prefix(permit_no: str) -> str: #Assumption: 5 numbers after prefix
  return permit_no[:-5]

day_mapper = {
    'monday': 'Weekdays',
    'tuesday': 'Weekdays',
    'wednesday': 'Weekdays',
    'thursday': 'Weekdays',
    'friday': 'Weekdays',
    'saturday': 'Weekends',
    'sunday': 'Weekends'
}

def number_letter_extract(number_letter):
  s = ""
  for idx in range(len(number_letter)-1, -1, -1):
    try:
      s = number_letter[:idx]
      return int(s)

    except:
      continue



def extract_number_from_lot(lot: str)->str:
  if 'Lot' in lot:
    the_rest = lot[4:]

    return number_letter_extract(the_rest)

def find_lot_name(input_name: str, lot_names: set) -> dict:
    """
    Takes a string `input_name` and a set of known lot names `lot_names`.
    If `input_name` is exactly in `lot_names`, returns [input_name].
    Otherwise, it returns a list of the 5 closest lot names.
    """
    # 1) Quick exact check
    if input_name in lot_names:
        return {1: input_name}

    # 2) Build a list from the set so we can index it
    lot_names_list = list(lot_names)

    # 3) Load a sentence embedding model (or any other embedding method you prefer)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4) Compute embeddings for all known lot names
    lot_embeddings = model.encode(lot_names_list, convert_to_numpy=True)

    # 5) Build a FAISS index
    embed_dim = lot_embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embed_dim)
    index.add(lot_embeddings)

    # 6) Compute embedding for input_name
    input_embedding = model.encode([input_name], convert_to_numpy=True)

    # 7) Search for the top 5 matches
    k = 5
    distances, indices = index.search(input_embedding, k)

    # 8) Return the 5 closest names
    closest_matches = {i+1: lot_names_list[idx] for i, idx in enumerate(indices[0])}
    return closest_matches

import pandas as pd

def build_prefix_mapper(csv_file_path):
    # Read the CSV
    df = pd.read_csv(csv_file_path)

    # Create a dictionary to accumulate data
    prefix_mapper = {}

    for _, row in df.iterrows():
        prefix = row["Prefix"]
        permission = row["Permissions"]
        time_val = row["Time"]
        permit_full_name = row["Permit Full Name"]

        # If this prefix hasn't been seen before, initialize
        if prefix not in prefix_mapper:
            prefix_mapper[prefix] = {
                "Lots": set(),
                "Time": set(),
                "PermitFullName": set()
            }

        # Add the permission to "Lots"
        prefix_mapper[prefix]["Lots"].add(permission)

        # Add the time
        prefix_mapper[prefix]["Time"].add(time_val)

        # Add the permit full name
        prefix_mapper[prefix]["PermitFullName"].add(permit_full_name)

    # Convert sets to lists for final output
    for prefix in prefix_mapper:
        prefix_mapper[prefix]["Lots"] = list(prefix_mapper[prefix]["Lots"])
        prefix_mapper[prefix]["Time"] = list(prefix_mapper[prefix]["Time"])
        prefix_mapper[prefix]["PermitFullName"] = list(prefix_mapper[prefix]["PermitFullName"])

    return prefix_mapper

import pickle

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

print("Trained model loaded successfully.")

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def check_parking_eligibility():
  global non_unique_prefixes
  global map_prefix_to_permission
  global parsed_data
  global lot_names
  global day_mapper

  permit = str(input("Enter permit number: "))
  lot = str(input("Enter lot name: "))
  day = str(input("Enter day: "))
  time = str(input("Enter time: "))


  if lot[0] in "1234567890":
    lot = f'Lot {lot}'

  lot_results = find_lot_name(lot, lot_names)

  if len(lot_results) == 1:
    lot = lot_results[1]
  else:
    print(f"Lot not found. Closest Matches:\n{lot_results}")
    lot_key = input('Enter Key: ')
    lot = lot_results[int(lot_key)]

  if lot == 'Lot 5' or lot == 'II':
    print(f'{lot} is Permanently Closed')
    return

  prefix = extract_prefix(permit)

  if len(map_prefix_to_permission[prefix]['Lots'])>1:
    lot_perm = input(f"Enter Lot you have this {prefix} permit for.\nAvailable Slots under this permit:\n{map_prefix_to_permission[prefix]['Lots']}:\n ")
  else:
    lot_perm = map_prefix_to_permission[prefix]['Lots'][0]


  if day.lower() in day_mapper:
    day = day_mapper[day.lower()]
  else:
    print('Invalid Day Input')
    return


  available_days = list(parsed_data[lot]['Permissions'].keys())
  if day not in available_days:
    day = 'Always'

  available_times = list(parsed_data[lot]['Permissions'][day].keys())
  data = pd.read_csv('data/Lots_Permissions_CH5_fakedata (1).csv')
  start_col = data.columns.get_loc('17FAE')
  output_columns = data.columns[start_col:]


  if not parsed_data[lot]['Physical Location']:
    print(f'{lot} is permanently closed')
    return

  for time_tuple in available_times:
    if is_time_in_tuple(time, time_tuple):
      decision = parsed_data[lot]['Permissions'][day][time_tuple][lot_perm]
      if not decision:
        true_lots = []
        for perm in parsed_data[lot]['Permissions'][day][time_tuple]:
          if parsed_data[lot]['Permissions'][day][time_tuple][perm]:
            true_lots.append(perm)
      model_input = pd.DataFrame({
        'Lot Name': [lot],
        'Enforcement Days': [day],
        'Start Time - Daily': [time_to_seconds(time_tuple[0])],
        'End Time - Daily': [time_to_seconds(time_tuple[1])]
      })
      output_list = model.predict(model_input)
      allowed_permits = [output_columns[i] for i, val in enumerate(output_list[0]) if val == 1]
      if 'Campus Meter' in true_lots and lot_perm not in true_lots:
        hasCampusMeter = str(input('Do you have a Campus Meter Permit? (Y/N): ')).upper()
        if hasCampusMeter == 'Y':
          print(f'Campus Metered Parking Permission for {lot} with permit {lot_perm}.')
          return
        elif hasCampusMeter == 'N':
          print(f'Parking Permission for {lot} with permit {lot_perm} is not available')
          print(f'Only {allowed_permits} can park here.')
          return

      print(f'Lookup:\nParking Permission for {lot} with permit {lot_perm} is {decision}')
      print('Model Prediction:')
      if lot_perm in allowed_permits:
        print(f'Parking Permission for {lot} with permit {lot_perm} is {lot_perm in allowed_permits}')
      else:
        print(f'Parking Permission for {lot} with permit {lot_perm} is not available')
        print(f'Only {allowed_permits} can park here.')
        # print(lot_perm)
        permit_name = map_prefix_to_permission[prefix]['PermitFullName'][0]
        if 'Commuter' in permit_name and time_tuple == ('3:00:00', '04:59:59') and lot_perm in ['Lot 1','Lot 3','Lot 4','Lot 6','Lot 9','Lot 11']:
          print(' ⁠⁠⁠Commuter Passes can’t park between 3-5 am in Lots 1, 3, 4, 6, 9, 11')
      return
    
fakedata = pd.read_csv('data/Lots_Permissions_CH5_fakedata (1).csv')
lot_names = set(fakedata['Lot Name'])

fakedata['Campus Meter'] = 0

cyclic_lots_weekdays, non_cyclic_lots_weekdays, cyclic_lots_weekends, non_cyclic_lots_weekends = check_cyclic_lots(fakedata, lot_names)
print(non_cyclic_lots_weekdays, non_cyclic_lots_weekends)

fakedata['End Time - Daily'] = pd.to_datetime(fakedata['End Time - Daily'], format='%H:%M:%S').dt.time

fakedata['End Time - Daily'] = fakedata['End Time - Daily'].apply(subtract_one_second)

# Convert 'End Time - Daily' back to string format
fakedata['End Time - Daily'] = fakedata['End Time - Daily'].astype(str)

parsed_data = parse_parking_data(fakedata)

file_path = "data/Permits & Permissions.csv"
map_prefix_to_permission = build_prefix_mapper(file_path)

check_parking_eligibility()