import os
import re
import json
import pickle
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, time as time_cls
from dotenv.main import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# The same OpenAI usage you have in your notebook
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("Please install openai: pip install openai")

# The same SentenceTransformer usage
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise RuntimeError("Please install sentence-transformers: pip install sentence-transformers")

#########################
#   CONFIG / PATHS
#########################

FILE_PATH = "data/Lots_Permissions_CH5_fakedata (1) (1).csv"
MODEL_PATH = "data/trained_model.pkl"
PERM_PATH = "data/Permits & Permissions.csv"

#########################
#   FLASK APP
#########################

app = Flask(__name__)

#########################
#   PREPROCESSING
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

# Called once at startup
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
#   LOAD GLOBAL DATA
#########################

parsed_data, model, output_columns, lot_names, map_prefix_to_permission = preprocessing_data(
    FILE_PATH, MODEL_PATH, PERM_PATH
)

client = OpenAI(api_key=OPENAI_API_KEY)

#########################
#   HELPER FUNCS
#########################

def extract_prefix(permit_no: str) -> str:
    # Find the alphabetical prefix before numbers
    return permit_no[:-5]

def validate_permit(permit: str):
    prefix = extract_prefix(permit)
    if not prefix:
        return None, "Sorry, that doesn't look like a valid permit format. Please double-check the number."
    if prefix not in map_prefix_to_permission:
        return None, f"Hmm, I don't recognize the prefix '{prefix}'. Try another permit."
    return prefix, None

def process_user_response_gpt(user_text: str, options: list):
    """
    Exactly the GPT usage from your 'process_user_response' in the notebook.
    We'll do a single call here. If GPT picks an option that exactly matches,
    we return it; otherwise, None.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"Extract the user's choice from this text. Options: {options}. Respond ONLY with the matching option. \
                            Remember to even match the cases of the letters of your response. If you respond with 'SDSTAR' when {options} has \
                            'SDStar', that would completely ruin my life. Stay Warned."
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
    """
    The 'no_prefix' function from your code:
    GPT returns strictly True or False
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content":f"Respond in True or False and nothing else. True if you DON'T see a permit ID like 17FAE12345 or 12TMP34566. Else False.\
                        A permit has a prefix like in {map_prefix_to_permission.keys()} followed by 5 random digits"
        }, {
            "role": "user",
            "content": text
        }]
    )
    reply = response.choices[0].message.content.strip()
    print(reply)
    return (reply == "True")

#########################
#   ENDPOINTS
#########################

@app.route('/permit_lookup', methods=['POST'])
def permit_lookup():
    """
    Single-request version of your 'extract_permission' logic:
    1) If user_text is recognized as a permit, validate prefix
    2) If multiple lots, return them
    3) If single lot, auto-assign
    4) If not recognized, return error
    """
    data = request.get_json(force=True)
    user_text = data.get("permit") or data.get("text")
    print(user_text)
    if not user_text:
        return jsonify(error="Missing 'permit' or 'text' in request."), 400

    # 1) Use GPT to see if there's a permit or not
    # If the user text might not have a prefix, check with no_prefix_gpt
    if no_prefix_gpt(user_text):
        return jsonify(error="No permit prefix detected in your input."), 400

    # 2) Extract the actual permit number via GPT
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

    # 3) Validate that permit prefix
    prefix, err = validate_permit(permit_gpt)
    if err:
        return jsonify(error=err), 400

    # 4) Check how many possible lots
    lots = map_prefix_to_permission[prefix]["Lots"]
    # Instead of 'assignedLot' or 'possibleLots', do:
    if len(lots) == 1:
        return jsonify(
            permit=permit_gpt,
            prefix=prefix,
            lots=[lots[0]],
            message=f"Great! {permit_gpt} is automatically assigned to {lots[0]}."
        ), 200
    else:
        return jsonify(
            permit=permit_gpt,
            prefix=prefix,
            lots=lots,
            message=f"I see {permit_gpt} can use these lots: {', '.join(lots)}. Which one do you want?"
        ), 200



@app.route('/lot_selection', methods=['POST'])
def lot_selection():
    """
    Single-request version of 'get_valid_lot' logic.
    1) Takes a user query for a physical lot
    2) Finds best matches using FAISS
    3) If exact or single, return it; else return suggestions
    """
    data = request.get_json(force=True)
    query = data.get("query")
    if not query:
        return jsonify(error="Missing 'query' in request."), 400

    # Build FAISS if not already
    # We do this at startup. We'll just do it once:
    # We only do a single global index for all lot_names
    # But let's do the search logic here
    # We'll replicate your 'ValidatorState' approach inline

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
            # Case-insensitive exact match check
            lower_query = query_text.lower()
            exact_matches = [name for name in self.lot_names if name.lower() == lower_query]
            if exact_matches:
                return {"exact": True, "lot": exact_matches[0]}

            # semantic search
            query_embed = self.model.encode([query_text])
            distances, indices = self.index.search(query_embed.astype(np.float32), 5)
            suggestions = [self.lot_names[idx] for idx in indices[0] if 0 <= idx < len(self.lot_names)]
            return {"exact": False, "suggestions": suggestions}

    validator = ValidatorState(lot_names)
    result = validator.find_matches(query)

    if result.get("exact"):
        return jsonify(
            lot=result["lot"],
            message=f"Lot '{result['lot']}' selected."
        ), 200
    else:
        suggestions = result.get("suggestions", [])
        if len(suggestions) == 1:
            # single suggestion
            return jsonify(
                lot=suggestions[0],
                message=f"Lot '{suggestions[0]}' selected (only suggestion)."
            ), 200
        else:
            return jsonify(
                suggestions=suggestions,
                message="Multiple matches found. Please choose."
            ), 200


@app.route('/validate_datetime', methods=['POST'])
def validate_datetime():
    """
    Combines 'get_valid_day' + 'rag_time_validation' in a single request:
    1) Check if day is valid for the chosen lot
    2) Use GPT to parse time & see if it fits an interval
    Returns { day: ..., time_interval: [...] } or error
    """
    data = request.get_json(force=True)
    lot = data.get("lot")
    day_input = data.get("day")
    time_input = data.get("time")
    if not lot or not day_input or not time_input:
        return jsonify(error="lot, day, and time are required."), 400

    if lot not in parsed_data:
        return jsonify(error=f"Unknown lot: {lot}"), 400

    # 1) get_valid_day logic
    # The code: we map user day to "Weekdays" or "Weekends" or "Always"
    # from your get_valid_day function:
    day_map = {
        'mon': 'Weekdays', 'tue': 'Weekdays', 'wed': 'Weekdays',
        'thu': 'Weekdays', 'fri': 'Weekdays', 'sat': 'Weekends',
        'sun': 'Weekends', 'weekend': 'Weekends', 'weekday': 'Weekdays'
    }
    lot_days = list(parsed_data[lot]['Permissions'].keys())

    # naive approach: check if day_input has any of day_map's keys
    # if found, map it, else error
    found_day = None
    clean_day = re.sub(r"\b(next|this|on|for|the|parking|day|days)\b", "", day_input.lower()).strip()
    for term, mapped_day in day_map.items():
        if term in clean_day:
            # check if mapped_day is in lot_days, else 'Always'
            if mapped_day not in lot_days:
                found_day = "Always"
            else:
                found_day = mapped_day
            break

    if not found_day:
        # check direct match
        if day_input in lot_days:
            found_day = day_input
        else:
            return jsonify(error="Could not interpret day. Please specify 'Weekdays' or 'Weekends'.", day_options=lot_days), 400

    # 2) rag_time_validation logic for time
    # We'll replicate your GPT approach for a single request:
    available_intervals = list(parsed_data[lot]['Permissions'][found_day].keys())

    # GPT call
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

    # see if parsed_time fits any interval
    matched_interval = None
    print(available_intervals)
    for interval in available_intervals:
        start_str, end_str = interval
        # parse them as HH:MM:SS or fallback
        try:
            start_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            end_obj = datetime.strptime(end_str, "%H:%M:%S").time()
        except:
            # fallback
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
        # Normal interval (e.g. 08:00–17:30)
            in_range = (start_obj <= parsed_time <= end_obj)
        else:
            # Wrap-around interval (e.g. 16:00–06:59)
            # Means it goes from start_obj to midnight, and midnight to end_obj
            in_range = (parsed_time >= start_obj) or (parsed_time <= end_obj)

        if in_range:
            # We found a matching interval
            matched_interval = interval
            break
        print(matched_interval)
    if not matched_interval:
        return jsonify(error=f"No availability at {raw_time} in {found_day} for lot {lot}."), 400

    # success
    return jsonify(
        day=found_day,
        time_interval=list(matched_interval),
        message=f"Valid day/time found: {found_day} {matched_interval[0]}-{matched_interval[1]}"
    ), 200


@app.route('/check_eligibility', methods=['POST'])
def check_eligibility():
    """
    Single-request version of 'check_parking_eligibility'.
    We'll pass:
      - permit
      - prefix (optional if we need it, or we can recalc)
      - lot_perm (the user-chosen permit-lot)
      - day
      - time_interval or time (we prefer the time_interval from /validate_datetime)
    Then we do the same logic as the notebook code.
    """
    data = request.get_json(force=True)
    permit = data.get("permit")
    lot = data.get("lot")
    day = data.get("day")
    time_interval = data.get("time_interval")  # expected [start_str, end_str]
    campus_meter_response = data.get("campus_meter")  # optional (Y/N)

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

    # Re-derive prefix
    prefix, err = validate_permit(permit_gpt)
    if err:
        return jsonify(error=err), 400

    # We also need which 'lot_perm' the user picked if multiple. We'll store it in data['lot_perm'] if you want.
    # But your code calls it "lot_perm". We'll see if user provided it. If not, we default to the first in map_prefix_to_permission
    lot_perm = data.get("lot_perm")
    if not lot_perm:
        # if user didn't supply, we fallback to the first lot in the prefix
        lots_for_prefix = map_prefix_to_permission[prefix]["Lots"]
        if len(lots_for_prefix) == 1:
            lot_perm = lots_for_prefix[0]
        else:
            return jsonify(error="lot_perm is required if multiple lots are possible for this prefix."), 400

    # Convert day/time into the actual keys
    if lot not in parsed_data:
        return jsonify(error=f"Unknown lot {lot}"), 400
    if day not in parsed_data[lot]["Permissions"]:
        return jsonify(error=f"Day {day} not in this lot's data."), 400
    interval_tuple = (time_interval[0], time_interval[1])
    if interval_tuple not in parsed_data[lot]["Permissions"][day]:
        return jsonify(error=f"No data for {lot} on {day} at {interval_tuple}."), 400

    # Now replicate check_parking_eligibility
    # We'll do the "campus meter" prompt logic by param
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    # is the lot physically available?
    if not parsed_data[lot]['Physical Location']:
        return jsonify(allowed=False, message=f"{lot} is permanently closed."), 200
    if lot in ["Lot 5", "II"]:
        return jsonify(allowed=False, message=f"{lot} is Permanently Closed."), 200

    # check the direct data
    decision = parsed_data[lot]['Permissions'][day][interval_tuple].get(lot_perm, False)

    # build a model_input for scikit
    model_input = pd.DataFrame([{
        'Lot Name': lot,
        'Enforcement Days': day,
        'Start Time - Daily': time_to_seconds(interval_tuple[0]),
        'End Time - Daily': time_to_seconds(interval_tuple[1])
    }])
    output_list = model.predict(model_input)
    allowed_permits = []
    if len(output_list) > 0:
        # output_list[0] is e.g. [0,1,0,...] shape
        row = output_list[0]
        for i, val in enumerate(row):
            if val == 1:
                allowed_permits.append(output_columns[i])

    # check if campus meter is needed
    all_true = parsed_data[lot]['Permissions'][day][interval_tuple]
    true_lots = [p for p, v in all_true.items() if v]

    if "Campus Meter" in true_lots and lot_perm not in true_lots:
        # user must confirm campus meter
        if campus_meter_response is None:
            # ask for it
            return jsonify(
                campus_meter_required=True,
                message="This lot/time requires a Campus Meter permit. Do you have one? (Y/N)"
            ), 200
        else:
            # interpret
            cm_resp = campus_meter_response.strip().lower()
            if cm_resp in ["y","yes","true","1"]:
                return jsonify(allowed=True, message=f"Campus Metered Parking is permitted for {lot_perm} in {lot}."), 200
            else:
                return jsonify(allowed=False, message=f"Parking not allowed for {lot_perm} here. Only {allowed_permits} can park."), 200

    # if no campus meter scenario
    # check if lot_perm is in allowed_permits
    from_prefix_fullname = map_prefix_to_permission[prefix]['PermitFullName'][0]
    if lot_perm in allowed_permits or decision:
        # special commuter check
        if 'Commuter' in from_prefix_fullname and interval_tuple == ("3:00:00", "04:59:59"):
            if any(lot.startswith(x) for x in ["Lot 1","Lot 3","Lot 4","Lot 6","Lot 9","Lot 11"]):
                return jsonify(
                    allowed=False,
                    message="Commuter passes can’t park between 3-5 AM in lots 1,3,4,6,9,11."
                ), 200
        return jsonify(allowed=True, message=f"Parking is allowed for {lot_perm} in {lot} on {day}."), 200
    else:
        # not allowed
        msg = f"Parking is NOT allowed for {lot_perm} in {lot} on {day}. Only {allowed_permits} can park."
        # commuter check
        if 'Commuter' in from_prefix_fullname and interval_tuple == ("3:00:00","04:59:59"):
            if any(lot.startswith(x) for x in ["Lot 1","Lot 3","Lot 4","Lot 6","Lot 9","Lot 11"]):
                msg += " Commuter passes can’t park between 3-5 AM in lots 1,3,4,6,9,11."
        return jsonify(allowed=False, message=msg), 200


#########################
#   RUN
#########################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
