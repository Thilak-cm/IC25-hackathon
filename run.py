import csv
import random
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# Global data structures to be populated from CSV
VALID_LOT_NAMES = set()
VALID_PERMIT_TYPES = set()
SPECIAL_EVENTS = []  # each item is a dict like {"date": "YYYY-MM-DD", "lot_name": "Lot A"}

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
    print("Loaded lots:", VALID_LOT_NAMES)
    print("Loaded permits:", VALID_PERMIT_TYPES)

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
            # print("Special event row:", row)  # Debug: print each row
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

@app.before_first_request
def initialize_data():
    """
    Called once before the first request. Loads data from CSV files into global variables.
    Replace the file paths below with the actual paths to your CSV files.
    """
    load_lots_permissions("data/InfoChallenge Dataset_ DOTS - Lots & Permissions.csv")
    load_special_events("data/InfoChallenge Dataset_ DOTS - Special Events _ Construction.csv")

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
            "permission": "No",
            "reason": f"Invalid lot name '{lot_name}'",
            "fine": "$69"
        }), 400

    # 2) Check that the permit type is valid
    if license_plate_or_permit_type not in VALID_PERMIT_TYPES:
        return jsonify({
            "permission": "No",
            "reason": f"Invalid permit type '{license_plate_or_permit_type}'",
            "fine": "$69"
        }), 400

    # 3) Special event check:
    if date_time:
        # Extract just the YYYY-MM-DD portion (assuming date_time = "YYYY-MM-DD HH:MM")
        date_part = date_time.split(" ")[0] if " " in date_time else date_time
        for event in SPECIAL_EVENTS:
            if event["date"] == date_part and event["lot_name"] == lot_name:
                return jsonify({
                    "permission": "No",
                    "reason": "special event",
                    "fine": "$10"
                })

    # 4) If none of the above is triggered, return random yes/no decision
    decision = random.choice([True, False])
    if decision:
        return jsonify({"permission": "Yes"})
    else:
        return jsonify({
            "permission": "No",
            "reason": "random",
            "fine": "$69"
        })

# New GET endpoint to retrieve valid lot names
@app.route('/lots', methods=['GET'])
def get_lot_names():
    return jsonify(sorted(list(VALID_LOT_NAMES)))

# New GET endpoint to retrieve valid permit types
@app.route('/permits', methods=['GET'])
def get_permit_types():
    return jsonify(sorted(list(VALID_PERMIT_TYPES)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
