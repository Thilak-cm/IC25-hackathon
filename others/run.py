import csv
import random
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from app import create_app
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

app = create_app()

# Global data structures to be populated from CSV
VALID_LOT_NAMES = set()
VALID_PERMIT_TYPES = set()
SPECIAL_EVENTS = []  # each item is a dict like {"date": "YYYY-MM-DD", "lot_name": "Lot A"}

# Global variable declaration
map_prefix_to_permission = {}

def load_lots_permissions(csv_path):
    """
    Loads lot names and permit types from the CSV.
    Expected CSV columns:
       - "Parking Lot / Zone Name" for lot names
       - "Permits Type (Category)" for permit types
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use the correct column name for lot names
            lot_name = row.get("Parking Lot / Zone Name", "").strip()
            if lot_name:
                VALID_LOT_NAMES.add(lot_name)
            # Use the correct column name for permit types
            permit_type = row.get("Permits Type (Category)", "").strip()
            if permit_type:
                VALID_PERMIT_TYPES.add(permit_type)

def load_special_events(csv_path):
    """
    Loads special event data from 'DOTS - Special Events _ Construction.csv'.
    Expected CSV columns:
       - date (YYYY-MM-DD)
       - lot_name
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_date_str = row.get("Start Date", "").strip()
            end_date_str = row.get("End Date", "").strip()
            lot_name = row.get("Affected Lot/Populations", "").strip()
            
            if start_date_str and end_date_str and lot_name:
                try:
                    start_date = datetime.strptime(start_date_str, "%m/%d/%Y")
                    end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
                    # Generate events for every day in the date range (inclusive)
                    delta = (end_date - start_date).days
                    for i in range(delta + 1):
                        event_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                        SPECIAL_EVENTS.append({
                            "date": event_date,
                            "lot_name": lot_name
                        })
                except ValueError as e:
                    print(f"Error parsing dates: {start_date_str}, {end_date_str} with error: {e}")

def build_prefix_mapper(csv_file_path):
    # Read the CSV
    df = pd.read_csv(csv_file_path)

    # Create a dictionary to accumulate data
    prefix_mapper = {}

    for _, row in df.iterrows():
        prefix = row["Prefix"]
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

# Initialize data at startup
with app.app_context():
    load_lots_permissions("data/Lots_Permissions_CH5_fakedata (1).csv")
    load_special_events("data/Special Events & Construction.csv")
    global lot_names
    fakedata = pd.read_csv('data/Lots_Permissions_CH5_fakedata (1).csv')
    lot_names = set(fakedata['Lot Name'])

    # Initialize map_prefix_to_permission
    try:
        map_prefix_to_permission = build_prefix_mapper("data/Permits & Permissions.csv")
        print("Prefix to permission mapping loaded successfully.")
    except Exception as e:
        print(f"Error loading prefix to permission mapping: {e}")

@app.route('/check_parking', methods=['POST'])
def check_parking():
    data = request.json

    # Extract fields from request JSON
    license_plate_or_permit_type = data.get('license_plate_or_permit_type')
    lot_name = data.get('lot_name')
    date_time = data.get('date_time')  # e.g. "YYYY-MM-DD HH:MM"
    user_type = data.get('user_type')  # faculty, staff, student
    disability_placard = data.get('disability_placard', False)

    # 1) Check that the lot name is valid
    if lot_name not in VALID_LOT_NAMES:
        return jsonify({
            "status": "error",
            "message": f"Invalid lot name '{lot_name}'",
            "alternatives": list(VALID_LOT_NAMES)[:3]
        })

    # 2) Check that the permit type is valid
    if license_plate_or_permit_type not in VALID_PERMIT_TYPES:
        return jsonify({
            "status": "error",
            "message": f"Invalid permit type '{license_plate_or_permit_type}'",
            "alternatives": list(VALID_PERMIT_TYPES)[:3]
        })

    # 3) Special event check:
    if date_time:
        # Extract just the YYYY-MM-DD portion (assuming date_time = "YYYY-MM-DD HH:MM")
        date_part = date_time.split(" ")[0] if " " in date_time else date_time
        for event in SPECIAL_EVENTS:
            if event["date"] == date_part and event["lot_name"] == lot_name:
                return jsonify({
                    "status": "denied",
                    "message": "Special event in progress",
                    "alternatives": list(VALID_LOT_NAMES)[:3]
                })

    # 4) If none of the above is triggered, return random yes/no decision
    decision = random.choice([True, False])
    if decision:
        return jsonify({
            "status": "allowed",
            "message": f"You can park in {lot_name}"
        })
    else:
        return jsonify({
            "status": "denied",
            "message": "Parking not allowed",
            "alternatives": list(VALID_LOT_NAMES)[:3]
        })

# New GET endpoint to retrieve valid lot names
@app.route('/lots', methods=['GET'])
def get_lot_names():
    return jsonify(sorted(list(VALID_LOT_NAMES)))

# New GET endpoint to retrieve valid permit types
@app.route('/permits', methods=['GET'])
def get_permit_types():
    return jsonify(sorted(list(VALID_PERMIT_TYPES)))

# Load parking data
def load_parking_data():
    lots = {}
    try:
        with open('data/Lots_Permissions_CH5_fakedata (1).csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                lot_name = row.get("Lot Name", "").strip()
                if lot_name:
                    lots[lot_name] = row
        return lots
    except FileNotFoundError:
        print("Error: Could not find the Lots & Permissions CSV file")
        return {}

def isSliceCyclc(slice):
#   print(slice.loc[0, 'Start Time - Daily'], slice.loc[0, 'End Time - Daily'])
  if slice.loc[0, 'Start Time - Daily'] == slice.loc[0, 'End Time - Daily']:
    flag = True
  else:
    flag = False

  start = slice.loc[0, 'Start Time - Daily']
  for idx, row in slice.iloc[1:,:].iterrows():
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

parsed_data = parse_parking_data(fakedata)

def subtract_one_second(t):
    if t == datetime.min.time():  # Check if it's midnight
        return datetime.strptime("23:59:59", "%H:%M:%S").time()
    else:
        return (datetime.combine(datetime.min, t) - timedelta(seconds=1)).time()
    
def find_non_unique_prefixes(df, column_name='Prefix'):
    """
    Finds and returns a list of non-unique values in the specified column of a DataFrame.
    """
    value_counts = df[column_name].value_counts()
    non_unique_values = value_counts[value_counts > 1].index.tolist()
    return non_unique_values

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

    lot_names_list = list(lot_names)

    if not lot_names_list:
        return {"status": "error", "message": "No lot names available"}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    lot_embeddings = model.encode(lot_names_list, convert_to_numpy=True)

    if lot_embeddings.size == 0:
        return {"status": "error", "message": "No lot names available"}

    embed_dim = lot_embeddings.shape[1]
    index = faiss.IndexFlatL2(embed_dim)
    index.add(lot_embeddings)

    input_embedding = model.encode([input_name], convert_to_numpy=True)
    k = 5
    distances, indices = index.search(input_embedding, k)

    if len(indices[0]) == 0:
        return {"status": "error", "message": "No close matches found"}

    closest_matches = {i+1: lot_names_list[idx] for i, idx in enumerate(indices[0]) if idx < len(lot_names_list)}
    print(closest_matches)
    return closest_matches

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/check_parking_eligibility', methods=['POST'])
def check_parking_eligibility_route():
    data = request.json
    permit = data.get('permit')
    lot = data.get('lot')
    day = data.get('day')
    time = data.get('time')

    result = check_parking_eligibility(permit, lot, day, time)
    print(result)

    return jsonify(result)

def check_parking_eligibility(permit, lot, day, time):
    if lot[0] in "1234567890":
        lot = f'Lot {lot}'
    print(f"Initial lot: {lot}")

    if lot.strip() == 'Lot 5' or lot.strip() == 'II':
        print('Entering the Lot 5 or II case')
        message = f"{lot} is Permanently Closed"
        return {
            "status": "denied",
            "message": message,
            "continue": False
        }

    lot_results = find_lot_name(lot, lot_names)
    print(f"Lot results: {lot_results}")

    if len(lot_results) == 1:
        lot = lot_results[1]
        print(f"Selected lot: {lot}")
    else:
        return {
            "status": "error",
            "message": "Lot not found. Please select from the closest matches.",
            "closest_matches": lot_results,
            "continue": True
        }

    prefix = extract_prefix(permit)

    if len(map_prefix_to_permission[prefix]['Lots']) > 1:
        return {
            "status": "error",
            "message": f"Multiple lots available for this permit. Available Slots: {map_prefix_to_permission[prefix]['Lots']}",
            "continue": False  # Flag indicating the process can end
        }
    else:
        lot_perm = map_prefix_to_permission[prefix]['Lots'][0]

    if day.lower() in day_mapper:
        day = day_mapper[day.lower()]
    else:
        return {
            "status": "error",
            "message": "Invalid Day Input",
            "continue": False  # Flag indicating the process can end
        }

    available_days = list(parsed_data[lot]['Permissions'].keys())
    if day not in available_days:
        day = 'Always'

    available_times = list(parsed_data[lot]['Permissions'][day].keys())
    data = pd.read_csv('data/Lots_Permissions_CH5_fakedata (1).csv')
    start_col = data.columns.get_loc('17FAE')
    output_columns = data.columns[start_col:]

    if not parsed_data[lot]['Physical Location']:
        return {
            "status": "error",
            "message": f"{lot} is permanently closed",
            "continue": False  # Flag indicating the process can end
        }

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
                return {
                    "status": "denied",
                    "message": f"Parking Permission for {lot_perm} with permit {lot} is not available. Only {allowed_permits} can park here.",
                    "allowed_permits": allowed_permits,
                    "continue": False  # Flag indicating the process can end
                }

            return {
                "status": "success",
                "message": f"Parking Permission for {lot_perm} with permit {lot} is {decision}",
                "allowed_permits": [],
                "continue": False  # Flag indicating the process can end
            }

    return {
        "status": "error",
        "message": "No valid time slot found for the given inputs.",
        "continue": False  # Flag indicating the process can end
    }

if __name__ == '__main__':
    try:
        app.parking_lots = load_parking_data()  # Make data available to app
        print(f"Successfully loaded {len(app.parking_lots)} parking lots")
        app.run(host='0.0.0.0', port=2000, debug=True)
    except Exception as e:
        print(f"Error starting application: {e}")

