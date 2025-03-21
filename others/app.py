import os
import re
import json
import requests
import pickle
import faiss
import numpy as np
import pandas as pd
import random
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, time as time_cls
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


#########################could 
#   CONFIG / PATHS
#########################

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")
FILE_PATH = "data/Lots_Permissions_CH5_fakedata (1).csv"
MODEL_PATH = "trained_model.pkl"
PERM_PATH = "data/Permits & Permissions.csv"

#########################
#   GLOBAL RESOURCES
#########################

# Pre-load closures data for lot status
closures_df = pd.read_csv('data/Special Events & Construction.csv')

# Pre-load parking restrictions data
parking_restrictions_df = pd.read_csv("data/Parking Restrictions.csv")

# Load FAISS index and metadata for parking restrictions search
faiss_index = faiss.read_index("parking_restrictions.index")
faiss_metadata = np.load("metadata.npy", allow_pickle=True)
search_model = SentenceTransformer("all-MiniLM-L6-v2")

#########################
#   FLASK APP
#########################

app = Flask(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

#########################
#   PREPROCESSING FUNCTIONS
#########################

def build_prefix_mapper(csv_file_path):
    df = pd.read_csv(csv_file_path)
    prefix_mapper = {}
    for _, row in df.iterrows():
        prefix = row["Prefix"]
        permission = row["Permissions"]
        time_val = row["Time"]
        permit_full_name = row["Permit Full Name"]

        if prefix not in prefix_mapper:
            prefix_mapper[prefix] = {
                "Lots": set(),
                "Time": set(),
                "PermitFullName": set()
            }
        prefix_mapper[prefix]["Lots"].add(permission)
        prefix_mapper[prefix]["Time"].add(time_val)
        prefix_mapper[prefix]["PermitFullName"].add(permit_full_name)

    for pref in prefix_mapper:
        prefix_mapper[pref]["Lots"] = list(prefix_mapper[pref]["Lots"])
        prefix_mapper[pref]["Time"] = list(prefix_mapper[pref]["Time"])
        prefix_mapper[pref]["PermitFullName"] = list(prefix_mapper[pref]["PermitFullName"])
    return prefix_mapper

def parse_parking_data(df):
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
    permit_columns = [col for col in df.columns if col not in known_columns]
    lots_dict = {}

    for _, row in df.iterrows():
        lot_name = str(row["Lot Name"])
        lot_type = str(row["Lot Type "])
        physical_location_val = str(row["Physical Location (Yes/No)"]).strip().upper()
        physical_location_bool = (physical_location_val == "YES")

        enforcement_day = str(row["Enforcement Days"]).strip()
        start_time_raw = str(row["Start Time - Daily"]).strip()
        end_time_raw = str(row["End Time - Daily"]).strip()

        if lot_name not in lots_dict:
            lots_dict[lot_name] = {
                "Type": lot_type,
                "Physical Location": physical_location_bool,
                "Permissions": {}
            }
        permit_dict = {}
        for pcol in permit_columns:
            val = row[pcol]
            permit_dict[pcol] = bool(val)

        if enforcement_day not in lots_dict[lot_name]["Permissions"]:
            lots_dict[lot_name]["Permissions"][enforcement_day] = {}
        time_tuple = (start_time_raw, end_time_raw)
        lots_dict[lot_name]["Permissions"][enforcement_day][time_tuple] = permit_dict

    return lots_dict

def subtract_one_second(t):
    if t == datetime.min.time():
        return time_cls(23, 59, 59)
    return (datetime.combine(datetime.min, t) - timedelta(seconds=1)).time()

def preprocessing_data(file_path, model_path, perm_path):
    map_prefix_to_permission = build_prefix_mapper(perm_path)

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    data = pd.read_csv(file_path)
    data['Campus Meter'] = 0

    data['End Time - Daily'] = pd.to_datetime(data['End Time - Daily'], format='%H:%M:%S').dt.time
    data['End Time - Daily'] = data['End Time - Daily'].apply(subtract_one_second)
    data['End Time - Daily'] = data['End Time - Daily'].astype(str)

    parsed_data = parse_parking_data(data)
    start_col = data.columns.get_loc('17FAE')  # adjust if needed
    output_columns = data.columns[start_col:]
    lot_names = list(parsed_data.keys())

    return parsed_data, model, output_columns, lot_names, map_prefix_to_permission

#########################
#   HELPER FUNCTIONS
#########################

def generate_lot_coordinates(lot_names):
    return {lot: (random.randint(0, 100), random.randint(0, 100)) for lot in lot_names}

def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def suggest_nearest_lots(closed_lot, lot_coordinates):
    closed_coord = lot_coordinates[closed_lot]
    distances = {
        lot: calculate_distance(closed_coord, coord)
        for lot, coord in lot_coordinates.items()
        if lot != closed_lot
    }
    return sorted(distances, key=distances.get)[:20]

def extract_prefix(permit_no: str) -> str:
    return permit_no[:-5]

def validate_permit(permit: str):
    prefix = extract_prefix(permit)
    if not prefix:
        return None, "Sorry, that doesn't look like a valid permit format. Please double-check the number."
    if prefix not in map_prefix_to_permission:
        return None, f"Hmm, I don't recognize the prefix '{prefix}'. Try another permit."
    return prefix, None

def process_user_response_gpt(user_text: str, options: list):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"Extract the user's choice from this text. Options: {options}. Respond ONLY with the matching option. \
                            Remember to even match the cases of the letters of your response."
            }, {
                "role": "user",
                "content": user_text
            }]
        )
        choice = response.choices[0].message.content.strip()
        return choice if choice in options else None
    except Exception as e:
        print(e)
        return None

def no_prefix_gpt(text: str) -> bool:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": f"Respond in True or False and nothing else. True if you DON'T see a permit ID like 17FAE12345 or 12TMP34566. Else False.\
                        A permit has a prefix like in {list(map_prefix_to_permission.keys())} followed by 5 random digits"
        }, {
            "role": "user",
            "content": text
        }]
    )
    reply = response.choices[0].message.content.strip()
    print(reply)
    return (reply == "True")

system_message = f"""Paraphrase the input while keeping its meaning and all key details intact. 
            Do not change any specific numbers, important terms, or **anything related to parking lots**—if 'parking lot' or 'lot' is mentioned, it **must stay exactly as is**. 
            Keep things **clear, natural, and smooth**, occasionally making the wording a bit **terse or conversational** to avoid sounding repetitive.  
            If the input includes a **list of lots**, do not alter it—keep the exact order and wording.  
            Your goal is to **reword it subtly**, making it sound natural, fluid, and slightly varied each time."""

def paraphrase_prompt(input_text, sys_msg = system_message, api_key=OPENAI_API_KEY):
    """
    Uses OpenAI's GPT API to paraphrase a given prompt while preserving its meaning.
    
    Args:
        input_text (str): The input string to be paraphrased.
        api_key (str): OpenAI API key.
        
    Returns:
        str: The paraphrased version of the input string.
    """
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"**Input:** {input_text}  **Paraphrased:**"}
            ]
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error: {e}")
        return input_text  # Fallback to the original input in case of an error

#########################
#   ORIGINAL ENDPOINTS
#########################

parsed_data, model, output_columns, lot_names, map_prefix_to_permission = preprocessing_data(
    FILE_PATH, MODEL_PATH, PERM_PATH
)

@app.route('/permit_lookup', methods=['POST'])
def permit_lookup():
    data = request.get_json(force=True)
    user_text = data.get("permit") or data.get("text")
    print(user_text)
    if not user_text:
        return jsonify(error="Missing 'permit' or 'text' in request."), 400

    if no_prefix_gpt(user_text):
        return jsonify(error="No permit prefix detected in your input."), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "From the input given, return what seems like a permit ID like 12TMP12345 or FS34567. The format is some prefix and 5 numbers. Return literally the permit number from the query and nothing else."
        }, {
            "role": "user",
            "content": user_text
        }]
    )
    permit_gpt = response.choices[0].message.content.strip()
    if not permit_gpt:
        return jsonify(error="Could not parse any permit from your input."), 400

    prefix, err = validate_permit(permit_gpt)
    if err:
        return jsonify(error=err), 400

    lots = map_prefix_to_permission[prefix]["Lots"]
    if len(lots) == 1:
        return jsonify(
            permit=permit_gpt,
            prefix=prefix,
            lots=[lots[0]],
            message=paraphrase_prompt(f"Great! {permit_gpt} is automatically assigned to {lots[0]}.")
        ), 200
    else:
        return jsonify(
            permit=permit_gpt,
            prefix=prefix,
            lots=lots,
            message=paraphrase_prompt(f"I see {permit_gpt} can use these lots: {', '.join(lots)}. Which one do you want?")
        ), 200

@app.route('/lot_selection', methods=['POST'])
def lot_selection():
    data = request.get_json(force=True)
    query = data.get("query")
    if not query:
        return jsonify(error="Missing 'query' in request."), 400

    class ValidatorState:
        def __init__(self, lot_names):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.lot_names = list(set(lot_names))
            self._build_index()

        def _build_index(self):
            embeddings = self.model.encode(self.lot_names)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype(np.float32))

        def find_matches(self, query_text):
            lower_query = query_text.lower()
            exact_matches = [name for name in self.lot_names if name.lower() == lower_query]
            if exact_matches:
                return {"exact": True, "lot": exact_matches[0]}
            query_embed = self.model.encode([query_text])
            distances, indices = self.index.search(query_embed.astype(np.float32), 5)
            suggestions = [self.lot_names[idx] for idx in indices[0] if 0 <= idx < len(self.lot_names)]
            return {"exact": False, "suggestions": suggestions}

    validator = ValidatorState(lot_names)
    result = validator.find_matches(query)

    if result.get("exact"):
        return jsonify(
            lot=result["lot"],
            message=paraphrase_prompt(f"Lot '{result['lot']}' selected.")
        ), 200
    else:
        suggestions = result.get("suggestions", [])
        if len(suggestions) == 1:
            return jsonify(
                lot=suggestions[0],
                message=paraphrase_prompt(f"Lot '{suggestions[0]}' selected (only suggestion).")
            ), 200
        else:
            return jsonify(
                suggestions=suggestions,
                message=paraphrase_prompt("Multiple matches found. Please choose.")
            ), 200

@app.route('/validate_datetime', methods=['POST'])
def validate_datetime():
    data = request.get_json(force=True)
    print(data)
    lot = data.get("lot")
    day_input = data.get("day")
    time_input = data.get("time")
    
    if not lot or not day_input or not time_input:
        return jsonify(error="lot, day, and time are required."), 400

    if lot not in parsed_data:
        return jsonify(error=f"Unknown lot: {lot}"), 400

    day_map = {
        'mon': 'Weekdays', 'tue': 'Weekdays', 'wed': 'Weekdays',
        'thu': 'Weekdays', 'fri': 'Weekdays', 'sat': 'Weekends',
        'sun': 'Weekends', 'weekend': 'Weekends', 'weekday': 'Weekdays'
    }
    lot_days = list(parsed_data[lot]['Permissions'].keys())

    found_day = None
    clean_day = re.sub(r"\b(next|this|on|for|the|parking|day|days)\b", "", day_input.lower()).strip()
    
    for term, mapped_day in day_map.items():
        if term in clean_day:
            if mapped_day in lot_days:
                found_day = mapped_day
            elif "Always" in lot_days:
                found_day = "Always"
            break

    if not found_day:
        # check direct match
        if day_input in lot_days:
            found_day = day_input
        else:
            return jsonify(error="Could not interpret day. Please specify 'Weekdays' or 'Weekends'.", day_options=lot_days), 400
        
    if found_day not in parsed_data[lot]['Permissions']:
        return jsonify(error=f"No permission data available for '{found_day}' in lot {lot}. Available options: {lot_days}"), 400

    available_intervals = list(parsed_data[lot]['Permissions'][found_day].keys())

    system_prompt = """You are a time converting assistant. Follow these rules:
    1. Extract time from user input in any format
    2. Convert it to 24hr HH:MM:SS format ONLY and nothing else

    Examples:
    User: "Quarter Past 4 in the evening" -> 16:15:00
    User: "13:05:11" -> 13:05:11
    User: "13:05" -> 13:05:00
    User: "around 2pm" -> 14:00:00
    User: "noon" -> 12:00:00
    User: "19:30" -> 19:30:00
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": time_input}
            ],
            temperature=0.1
        )
        raw_time = resp.choices[0].message.content.strip()
        parsed_time = datetime.strptime(raw_time, "%H:%M:%S").time()
    except Exception:
        return jsonify(error="Unable to parse time from GPT. Try a different format."), 400

    matched_interval = None
    print(available_intervals)
    for interval in available_intervals:
        start_str, end_str = interval
        try:
            start_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            end_obj = datetime.strptime(end_str, "%H:%M:%S").time()
        except:
            try:
                start_obj = datetime.strptime(start_str, "%H:%M").time()
            except:
                start_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            try:
                end_obj = datetime.strptime(end_str, "%H:%M").time()
            except:
                end_obj = datetime.strptime(end_str, "%H:%M:%S").time()

        print(start_obj, parsed_time, end_obj)

        if start_obj <= end_obj:
            in_range = (start_obj <= parsed_time <= end_obj)
        else:
            in_range = (parsed_time >= start_obj) or (parsed_time <= end_obj)

        if in_range:
            matched_interval = interval
            break
        print(matched_interval)
    if not matched_interval:
        url = "http://10.4.6.248:1000/log_alert"
        payload = {
            "alert_message": f"A time gap was discovered!.",
            "details": f"Lot: {lot}, Day: {found_day}, Time: {raw_time}"
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("Failed to log alert")
        return jsonify(error=f"No availability at {raw_time} in {found_day} for lot {lot}."), 400

    return jsonify(
        day=found_day,
        time_interval=list(matched_interval),
        message=paraphrase_prompt(f"Valid day/time found: {found_day} {matched_interval[0]}-{matched_interval[1]}")
    ), 200



@app.route('/check_eligibility', methods=['POST'])
def check_eligibility():
    data = request.get_json(force=True)
    permit = data.get("permit")
    lot = data.get("lot")
    day = data.get("day")
    ada = data.get("ada")
    time_interval = data.get("time_interval")
    campus_meter_response = data.get("campus_meter")

    if not permit or not lot or not day or not time_interval:
        print(permit, lot, day, time_interval)
        return jsonify(error="permit, lot, day, and time_interval are required."), 400
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "From the input given, return what seems like a permit ID like 12TMP12345 or FS34567. The format is some prefix and 5 numbers. Return literally the permit number from the query and nothing else."
        }, {
            "role": "user",
            "content": permit
        }]
    )
    permit_gpt = response.choices[0].message.content.strip()
    if not permit_gpt:
        return jsonify(error="Could not parse any permit from your input."), 400

    prefix, err = validate_permit(permit_gpt)
    if err:
        return jsonify(error=err), 400

    lot_perm = data.get("lot_perm")
    if not lot_perm:
        lots_for_prefix = map_prefix_to_permission[prefix]["Lots"]
        if len(lots_for_prefix) == 1:
            lot_perm = lots_for_prefix[0]
        else:
            return jsonify(error="lot_perm is required if multiple lots are possible for this prefix."), 400

    if lot not in parsed_data:
        return jsonify(error=f"Unknown lot {lot}"), 400
    if day not in parsed_data[lot]["Permissions"]:
        return jsonify(error=f"Day {day} not in this lot's data."), 400
    interval_tuple = (time_interval[0], time_interval[1])
    if interval_tuple not in parsed_data[lot]["Permissions"][day]:
        return jsonify(error=f"No data for {lot} on {day} at {interval_tuple}."), 400

    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    if not parsed_data[lot]['Physical Location']:
        return jsonify(allowed=False, message=paraphrase_prompt(f"{lot} is permanently closed.")), 200
    if lot in ["Lot 5", "II"]:
        return jsonify(allowed=False, message=paraphrase_prompt(f"{lot} is Permanently Closed.")), 200

    decision = parsed_data[lot]['Permissions'][day][interval_tuple].get(lot_perm, False)

    model_input = pd.DataFrame([{
        'Lot Name': lot,
        'Enforcement Days': day,
        'Start Time - Daily': time_to_seconds(interval_tuple[0]),
        'End Time - Daily': time_to_seconds(interval_tuple[1])
    }])
    output_list = model.predict(model_input)
    allowed_permits = []
    if len(output_list) > 0:
        row = output_list[0]
        for i, val in enumerate(row):
            if val == 1:
                allowed_permits.append(output_columns[i])

    all_true = parsed_data[lot]['Permissions'][day][interval_tuple]
    true_lots = [p for p, v in all_true.items() if v]

    if "Campus Meter" in true_lots and lot_perm not in true_lots:
        if campus_meter_response is None:
            return jsonify(
                campus_meter_required=True,
                message=paraphrase_prompt("This lot/time requires a Campus Meter permit. Do you have one? (Y/N)")
            ), 200
        else:
            cm_resp = campus_meter_response.strip().lower()
            if cm_resp in ["y","yes","true","1"]:
                return jsonify(allowed=True, message=paraphrase_prompt(f"Campus Metered Parking is permitted for {lot_perm} in {lot}.")), 200
            else:
                return jsonify(allowed=False, message=paraphrase_prompt(f"Parking not allowed for {lot_perm} here. Only {allowed_permits} can park.")), 200

    from_prefix_fullname = map_prefix_to_permission[prefix]['PermitFullName'][0]
    if lot_perm in allowed_permits:
        if 'Commuter' in from_prefix_fullname and interval_tuple == ("3:00:00", "04:59:59"):
            if any(lot.startswith(x) for x in ["Lot 1","Lot 3","Lot 4","Lot 6","Lot 9","Lot 11"]):
                return jsonify(
                    allowed=False,
                    message=paraphrase_prompt("Commuter passes can't park between 3-5 AM in lots 1,3,4,6,9,11.")
                ), 200
        return jsonify(allowed=True, message=paraphrase_prompt(f"Parking is allowed for {lot_perm} in {lot} on {day}.")), 200
    else:
        msg = paraphrase_prompt(f"Parking is NOT allowed for {lot_perm} in {lot} on {day}. Only {allowed_permits} can park.")
        if 'Commuter' in from_prefix_fullname and interval_tuple == ("3:00:00","04:59:59"):
            if any(lot.startswith(x) for x in ["Lot 1","Lot 3","Lot 4","Lot 6","Lot 9","Lot 11"]):
                msg += " Commuter passes can't park between 3-5 AM in lots 1,3,4,6,9,11."
        return jsonify(allowed=False, message=msg), 200

#########################
#   NEW HELPER FUNCTIONS (IPYNB)
#########################

def check_for_closures(lot, date_input):
    BOT_NAME = "Parking Assistant"
    try:
        if date_input.lower() == 'today':
            input_date = datetime.now()
        elif date_input.lower() == 'tomorrow':
            input_date = datetime.now() + timedelta(days=1)
        else:
            input_date = datetime.strptime(date_input, '%m-%d-%Y')
    except ValueError:
        return False, f"{BOT_NAME}: The date format is incorrect. Please try again."

    lot_closures = closures_df[closures_df['Affected Lot/Populations'] == lot]
    for _, row in lot_closures.iterrows():
        start_date = datetime.strptime(row['Start Date'], '%m/%d/%Y')
        end_date = datetime.strptime(row['End Date'], '%m/%d/%Y')
        if start_date <= input_date <= end_date:
            return True, row['Closure Type']
    return False, None

def get_parking_details(lot_name):
    lot_info = parking_restrictions_df[parking_restrictions_df["Parking Lot / Zone Name"] == lot_name]
    if lot_info.empty:
        return "No additional parking details found for this lot."
    relevant_columns = ["Restrictions", "Required", "Parking Restrictions", "Overflow Lot"]
    selected_data = lot_info.iloc[0][relevant_columns].dropna()
    details_text = "\n".join([f"**{col}**: {val}" for col, val in selected_data.items()])
    return details_text

def search_parking_restrictions(query_json, top_k=3):
    query_text = json.dumps(query_json, indent=2)
    query_embedding = search_model.encode([query_text], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(faiss_metadata):
            # Convert metadata to list if needed
            meta = faiss_metadata[idx]
            if isinstance(meta, (np.ndarray, list)):
                meta = meta.tolist()
            results.append({
                "rank": i + 1,
                "score": float(distances[0][i]),
                "restrictions": meta
            })
    return results

def get_ada_policy(sys_msg):
    with open('ada_policies.json', 'r') as file:
        ada_policies = json.load(file)
    ada_policies = paraphrase_prompt(ada_policies, sys_msg)
    return ada_policies

#########################
#   NEW API ENDPOINTS
#########################

@app.route('/paraphrase', methods=['POST'])
def paraphrase_endpoint():
    data = request.get_json(force=True)
    input_text = data.get("text")
    if not input_text:
        return jsonify(error="Missing 'text'."), 400
    paraphrased = paraphrase_prompt(input_text)
    return jsonify(original=input_text, paraphrased=paraphrased), 200

@app.route('/check_closures', methods=['POST'])
def check_closures_endpoint():
    data = request.get_json(force=True)
    lot = data.get("lot")
    date_input = data.get("date")
    if not lot or not date_input:
        return jsonify(error="Missing 'lot' or 'date' in request."), 400
    closed, closure_type = check_for_closures(lot, date_input)
    return jsonify(lot=lot, date=date_input, closed=closed, closure_type=closure_type), 200

@app.route('/search_restrictions', methods=['POST'])
def search_restrictions_endpoint():
    data = request.get_json(force=True)
    query_json = data.get("query")
    if not query_json:
        return jsonify(error="Missing 'query'."), 400
    results = search_parking_restrictions(query_json)
    return jsonify(results=results), 200

@app.route('/get_parking_details', methods=['GET'])
def get_parking_details_endpoint():
    permit = request.args.get("lot_perm")
    lot_name = request.args.get("lot")
    day = request.args.get("day")
    time_interval = request.args.get("interval_tuple")
    time_interval = time_interval.split(",")
    time_interval = (time_interval[0], time_interval[1])
    print(permit, lot_name, day, time_interval)
    
    if not permit or not lot_name or not day or not time_interval:
        return jsonify(error="Missing 'lot_perm', 'lot', 'day', or 'interval_tuple' in request."), 400

    lot_coordinates = generate_lot_coordinates(lot_names)
    closed, closure_type = check_for_closures(lot_name, day)
    if closed:
        return jsonify(error="The lot is closed."), 400
    else:
        nearest_lots = suggest_nearest_lots(lot_name, lot_coordinates)
        lot_info = parsed_data.get(lot_name)
        true_lots = [perm for perm, value in lot_info['Permissions'][day][time_interval].items() if value]
        valid_lots = [lot for lot in nearest_lots if lot in true_lots]
        message = f"With these permissions, you can also park in {valid_lots}."
        return jsonify(message=message), 200

@app.route('/get_ada_policy', methods=['GET'])
def get_ada_policy_endpoint():
    ada_policies = get_ada_policy(
        sys_msg="You are a professional parking assistant. Provide clear and concise ADA policy details with insights. Avoid markdown and headers. Maintain a formal and informative tone."
    )
    return jsonify(ada_policies=ada_policies), 200

#########################
#   RUN APP
#########################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
