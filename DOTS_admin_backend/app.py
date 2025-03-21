from flask import Flask, jsonify, request
import threading, time, datetime
import sqlite3, os
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)


################################################################
# Load CSV data and create global LOTS and PERMS variables
################################################################
try:
    # Adjust the file path as necessary.
    df = pd.read_csv("data/Lots_Permissions_CH5_fakedata (1) (1).csv")
    LOTS = list(set(df['Lot Name']))
    # Assuming that the permit names are stored as the column headers from index 8 onward.
    PERMS = list(set(df.columns[8:]))
except Exception as e:
    print("Error loading CSV:", e)
    LOTS = []
    PERMS = []

################################################################
# Global data (schema as specified)
################################################################
special_rules = {
    'Allowed': {
       'Lot 1': {
         ('07:00:00','23:59:59'): [],
         'End Day': '2025-04-12'
       }
    },
    'Not Allowed': {
       # same schema as Allowed
    },
    'Closed': {
       'Lot 1': {
         'End Day': '2025-04-12',
         'End Time': '23:59:59'
       }
    },
    'New Permits': {
       'Name1': {
         'Perms': [],
         'End Day': '2025-04-12',
         'End Time': '23:59:59'
       }
    }
}

# Global list to hold pending updates (for future in_effect_from times)
pending_updates = []  # each: {category, lot, time_slot, perms, in_effect_from, end_day, end_time}

################################################################
# Helper functions for time conversions and interval handling
################################################################
def time_to_seconds(t: str) -> int:
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(s: int) -> str:
    s %= 86400
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def add_events_for_interval(start_str: str, end_str: str, perms: dict, source: str, events: list):
    start = time_to_seconds(start_str)
    end = time_to_seconds(end_str)
    if start <= end:
        events.append((start, 'start', source, perms))
        events.append((end + 1, 'end', source, perms))
    else:
        # crosses midnight: split into two segments
        events.append((start, 'start', source, perms))
        events.append((86400, 'end', source, perms))
        events.append((0, 'start', source, perms))
        events.append((end + 1, 'end', source, perms))

def combine_permission_dict(new_dict, old_dict):
    """
    Original line‐sweep algorithm: merges two dicts with tuple keys.
    Used as fallback if we cannot perform a pairwise conflict resolution.
    """
    events = []
    for (start, end), perms in new_dict.items():
        add_events_for_interval(start, end, perms, 'new', events)
    for (start, end), perms in old_dict.items():
        add_events_for_interval(start, end, perms, 'old', events)
    if not events:
        return {}
    events.sort(key=lambda x: (x[0], 0 if x[1]=='start' else 1))
    active_new = 0
    active_old = 0
    cur_new = None
    cur_old = None
    result = []
    prev_time = events[0][0]
    for time_pt, etype, source, perms in events:
        if time_pt > prev_time:
            effective = cur_new if active_new > 0 else (cur_old if active_old > 0 else None)
            if effective is not None:
                result.append((prev_time, time_pt, effective))
            prev_time = time_pt
        if etype == 'start':
            if source == 'new':
                active_new += 1
                cur_new = perms
            else:
                active_old += 1
                cur_old = perms
        else:
            if source == 'new':
                active_new -= 1
                if active_new == 0:
                    cur_new = None
            else:
                active_old -= 1
                if active_old == 0:
                    cur_old = None
    # Merge contiguous segments
    merged = []
    for seg in result:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if last[1] == seg[0] and last[2] == seg[2]:
                merged[-1] = (last[0], seg[1], last[2])
            else:
                merged.append(seg)
    # Circular merge if applicable
    if merged and merged[0][0] == 0 and merged[-1][1] == 86400 and merged[0][2] == merged[-1][2]:
        first = merged.pop(0)
        last = merged.pop(-1)
        merged.insert(0, (last[0], first[1], first[2]))
    out = {}
    for seg in merged:
        sstart, send, perms = seg
        disp_start = seconds_to_time(sstart)
        disp_end = seconds_to_time(send - 1)
        if send - sstart <= 0:
            continue
        out[(disp_start, disp_end)] = perms
    return out

def circular_interval_to_segments(interval):
    """
    Converts a circular interval (tuple of time strings) into a list of linear segments (in seconds).
    E.g., ('10:00:00','06:59:59') -> [(36000,86400), (0, 25199)]
    """
    start, end = interval
    start_sec = time_to_seconds(start)
    end_sec = time_to_seconds(end)
    if start_sec <= end_sec:
        return [(start_sec, end_sec)]
    else:
        return [(start_sec, 86400), (0, end_sec)]

def subtract_intervals(segment, interval_to_subtract):
    """
    Given a linear segment (s, e) in seconds and an interval (as a circular interval tuple),
    subtract the interval from the segment and return a list of remaining segments.
    """
    s, e = segment
    sub_segs = circular_interval_to_segments(interval_to_subtract)
    result = [(s, e)]
    for ns, ne in sub_segs:
        new_result = []
        for rs, re in result:
            if ne <= rs or ns >= re:
                # no overlap
                new_result.append((rs, re))
            else:
                if ns > rs:
                    new_result.append((rs, ns))
                if ne < re:
                    new_result.append((ne, re))
        result = new_result
    return result

def resolve_intervals(new_interval, new_perms, old_interval, old_perms):
    """
    Given:
      new_interval: tuple (start, end) as strings (circular) for new rule.
      old_interval: tuple (start, end) as strings for old rule.
    Returns a dict mapping effective interval(s) to perms.
    New rule takes precedence.
    
    Examples:
      2. new: ('10:00:00','06:59:59'), old: ('12:00:00','08:59:59')
         => {('10:00:00','06:59:59'): new_perms, ('07:00:00','08:59:59'): old_perms}
      3. new: ('12:00:00','08:59:59'), old: ('10:00:00','06:59:59')
         => {('10:00:00','11:59:59'): old_perms, ('12:00:00','08:59:59'): new_perms}
    """
    # New rule is output exactly as submitted.
    out = {new_interval: new_perms}
    # For the old rule, subtract the new interval from the old interval.
    old_segs = circular_interval_to_segments(old_interval)
    remaining = []
    for seg in old_segs:
        diff = subtract_intervals(seg, new_interval)
        remaining.extend(diff)
    # Try to merge remaining segments if they are contiguous in circular sense.
    remaining = sorted(remaining)
    merged = []
    for seg in remaining:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], seg[1]))
            else:
                merged.append(seg)
    # Convert merged segments back to time-string tuples
    for s, e in merged:
        interval_str = (seconds_to_time(s), seconds_to_time(e))
        out[interval_str] = old_perms
    return out

def serialize_special_rules(rules):
    """
    For 'Allowed'/'Not Allowed', convert tuple keys to "start|end".
    For 'Closed' and 'New Permits', pass them through.
    """
    serialized = {}
    for cat, content in rules.items():
        if cat in ['Allowed', 'Not Allowed']:
            cat_dict = {}
            for lot, rd in content.items():
                new_rd = {}
                for k, v in rd.items():
                    if isinstance(k, tuple):
                        new_k = f"{k[0]}|{k[1]}"
                        new_rd[new_k] = v
                    else:
                        new_rd[k] = v
                cat_dict[lot] = new_rd
            serialized[cat] = cat_dict
        else:
            serialized[cat] = content
    return serialized

def normalize_time(t: str) -> str:
    if len(t) == 5:
        return t + ":00"
    return t

################################################################
# update_lot_rule for Allowed/Not Allowed using conflict resolution if possible
################################################################
def update_lot_rule(category, lot_name, time_slot, perms, end_day, end_time):
    """
    For categories "Allowed" and "Not Allowed", merge new rule (time_slot, perms)
    with existing rules for the given lot.
    If there is exactly one existing interval, attempt to resolve conflict to produce
    a circular interval for the new rule.
    Otherwise, fall back to the standard combine_permission_dict.
    """
    if category not in special_rules:
        special_rules[category] = {}
    if lot_name not in special_rules[category]:
        # No existing rule: simply store the new one.
        special_rules[category][lot_name] = {(time_slot[0], time_slot[1]): perms, "End Day": end_day, "End Time": end_time}
        return
    current = special_rules[category][lot_name]
    # Separate keys: those that are time intervals vs. others.
    existing = {k: v for k, v in current.items() if isinstance(k, tuple)}
    others = {k: v for k, v in current.items() if not isinstance(k, tuple)}

    # If there's exactly one existing interval, try to resolve conflict:
    if len(existing) == 1:
        (old_interval, old_perms) = next(iter(existing.items()))
        resolved = resolve_intervals(time_slot, perms, old_interval, old_perms)
        # Merge with non-tuple keys:
        resolved.update(others)
        special_rules[category][lot_name] = resolved
    else:
        # Fallback: use the original combine method.
        new_dict = { (time_slot[0], time_slot[1]) : perms }
        combined = combine_permission_dict(new_dict, existing)
        combined.update(others)
        combined["End Day"] = end_day
        combined["End Time"] = end_time
        special_rules[category][lot_name] = combined

################################################################
# Expiration logic for all categories
################################################################
def update_expired_rules():
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    curr_time_str = now.strftime("%H:%M:%S")
    for cat, cat_rules in list(special_rules.items()):
        for key, rule in list(cat_rules.items()):
            end_day = rule.get("End Day")
            if not end_day:
                continue
            e_time = normalize_time(rule.get("End Time", "23:59:59"))
            if today_str > end_day or (today_str == end_day and curr_time_str > e_time):
                alert_msg = f"Rule for '{key}' in category '{cat}' removed due to expiration."
                details = f"Expired at {today_str} {curr_time_str}. Rule details: {rule}"
                log_alert_to_db(alert_msg, details)
                del cat_rules[key]

################################################################
# Scheduler thread
################################################################
def scheduler_thread():
    while True:
        update_expired_rules()
        now = datetime.datetime.now()
        to_remove = []
        for up in pending_updates:
            if now >= up["in_effect_from"]:
                cat = up["category"]
                slot = up["time_slot"]
                if isinstance(slot, list):
                    slot = tuple(slot)
                if cat in ["Allowed", "Not Allowed"]:
                    update_lot_rule(
                        cat,
                        up["lot"],
                        slot,
                        up["perms"],
                        up["end_day"],
                        up["end_time"]
                    )
                elif cat == "Closed":
                    if "Closed" not in special_rules:
                        special_rules["Closed"] = {}
                    special_rules["Closed"][up["lot"]] = {
                        "End Day": up["end_day"],
                        "End Time": up["end_time"]
                    }
                elif cat == "New Permits":
                    if "New Permits" not in special_rules:
                        special_rules["New Permits"] = {}
                    special_rules["New Permits"][up["lot"]] = {
                        "Perms": up["perms"],
                        "End Day": up["end_day"],
                        "End Time": up["end_time"]
                    }
                alert_msg = f"Pending update applied for '{up['lot']}' in category '{cat}'."
                details = f"Applied at {now.strftime('%Y-%m-%d %H:%M:%S')} => {up}"
                log_alert_to_db(alert_msg, details)
                to_remove.append(up)
        for item in to_remove:
            pending_updates.remove(item)
        time.sleep(1)

threading.Thread(target=scheduler_thread, daemon=True).start()

################################################################
# Database: Alerts
################################################################
def init_db():
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_message TEXT,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_alert_to_db(msg, details):
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    c.execute("INSERT INTO alerts (alert_message, details) VALUES (?, ?)", (msg, details))
    conn.commit()
    conn.close()

init_db()

################################################################
# New endpoint: Return LOTS and PERMS from CSV
################################################################
@app.route('/get_lot_perms', methods=['GET'])
def get_lot_perms():
    return jsonify({
        "LOT_NAMES": LOTS,
        "PERM_NAMES": PERMS
    })


################################################################
# Flask endpoints
################################################################
@app.route('/get_restrictions', methods=['GET'])
def get_restrictions():
    return jsonify(serialize_special_rules(special_rules))

@app.route('/log_alert', methods=['POST'])
def log_alert():
    data = request.get_json()
    alert_msg = data.get("alert_message", "No msg")
    det = data.get("details", "")
    log_alert_to_db(alert_msg, det)
    return jsonify({"status": "logged"}), 200

@app.route('/update_rule', methods=['POST'])
def update_rule():
    """
    Expects fields depending on category:
    For 'Allowed'/'Not Allowed':
      {
        "category": "Allowed" or "Not Allowed",
        "lots": ["Lot A", "Lot B", ...],
        "perms": ["Permit1", "Permit2", ...],
        "time_slot": "HH:MM:SS|HH:MM:SS",
        "in_effect_from": "YYYY-MM-DD HH:MM:SS",
        "end_day": "YYYY-MM-DD",
        "end_time": "HH:MM:SS"
      }
    For 'Closed':
      {
        "category": "Closed",
        "lots": ["LotA", "LotB", ...],
        "in_effect_from": "YYYY-MM-DD HH:MM:SS",
        "end_day": "YYYY-MM-DD",
        "end_time": "HH:MM:SS"
      }
    For 'New Permits':
      {
        "category": "New Permits",
        "new_permit_name": "SomeName",
        "perms": ["Permit1", "Permit2", ...],
        "in_effect_from": "YYYY-MM-DD HH:MM:SS",
        "end_day": "YYYY-MM-DD",
        "end_time": "HH:MM:SS"
      }
    """
    data = request.get_json()
    cat = data.get("category")
    in_effect_from_str = data.get("in_effect_from")
    end_day = data.get("end_day")
    end_time = data.get("end_time")
    if not (cat and in_effect_from_str and end_day and end_time):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        in_effect_from = datetime.datetime.strptime(in_effect_from_str, '%Y-%m-%d %H:%M:%S')
    except:
        return jsonify({"error": "Invalid in_effect_from format. Use YYYY-MM-DD HH:MM:SS"}), 400

    now = datetime.datetime.now()

    if cat in ["Allowed", "Not Allowed"]:
        lots = data.get("lots", [])
        if isinstance(lots, str):
            lots = [lots]
        perms = data.get("perms", [])
        if not isinstance(perms, list):
            perms = []
        time_slot_str = data.get("time_slot")
        if not time_slot_str:
            return jsonify({"error": "Missing time_slot for Allowed/Not Allowed"}), 400
        try:
            start_str, end_str = time_slot_str.split("|")
        except:
            return jsonify({"error": "time_slot must be 'start|end'"}), 400

        if in_effect_from <= now:
            for ln in lots:
                update_lot_rule(cat, ln, (start_str, end_str), perms, end_day, end_time)
            return jsonify({"status": "updated immediately"}), 200
        else:
            for ln in lots:
                pending_updates.append({
                    "category": cat,
                    "lot": ln,
                    "time_slot": (start_str, end_str),
                    "perms": perms,
                    "in_effect_from": in_effect_from,
                    "end_day": end_day,
                    "end_time": end_time
                })
            return jsonify({"status": "pending update scheduled"}), 200

    elif cat == "Closed":
        lots = data.get("lots", [])
        if isinstance(lots, str):
            lots = [lots]
        if in_effect_from <= now:
            for ln in lots:
                if "Closed" not in special_rules:
                    special_rules["Closed"] = {}
                special_rules["Closed"][ln] = {
                    "End Day": end_day,
                    "End Time": end_time
                }
            return jsonify({"status": "updated immediately"}), 200
        else:
            for ln in lots:
                pending_updates.append({
                    "category": "Closed",
                    "lot": ln,
                    "time_slot": ("00:00:00", "00:00:00"),  # dummy
                    "perms": [],
                    "in_effect_from": in_effect_from,
                    "end_day": end_day,
                    "end_time": end_time
                })
            return jsonify({"status": "pending update scheduled"}), 200

    elif cat == "New Permits":
        name = data.get("new_permit_name")
        perms = data.get("perms", [])
        if not name:
            return jsonify({"error": "Missing 'new_permit_name' for New Permits"}), 400
        if in_effect_from <= now:
            if "New Permits" not in special_rules:
                special_rules["New Permits"] = {}
            special_rules["New Permits"][name] = {
                "Perms": perms,
                "End Day": end_day,
                "End Time": end_time
            }
            return jsonify({"status": "updated immediately"}), 200
        else:
            pending_updates.append({
                "category": "New Permits",
                "lot": name,  # reusing "lot" field for the name
                "time_slot": ("00:00:00", "00:00:00"),  # dummy
                "perms": perms,
                "in_effect_from": in_effect_from,
                "end_day": end_day,
                "end_time": end_time
            })
            return jsonify({"status": "pending update scheduled"}), 200
    else:
        return jsonify({"error": f"Unknown category {cat}"}), 400

@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    c.execute("SELECT id, alert_message, details, timestamp FROM alerts ORDER BY timestamp DESC")
    rows = c.fetchall()
    alerts = []
    for r in rows:
        alerts.append({
            "id": r[0],
            "alert_message": r[1],
            "details": r[2],
            "timestamp": r[3]
        })
    conn.close()
    return jsonify(alerts)

@app.route('/pending_updates', methods=['GET'])
def get_pending_updates():
    out = []
    for up in pending_updates:
        copyup = up.copy()
        copyup["in_effect_from"] = copyup["in_effect_from"].strftime("%Y-%m-%d %H:%M:%S")
        out.append(copyup)
    return jsonify(out)

@app.route('/delete_alert/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    try:
        conn = sqlite3.connect("alerts.db")
        c = conn.cursor()
        c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_alerts', methods=['DELETE'])
def clear_alerts():
    try:
        conn = sqlite3.connect("alerts.db")
        c = conn.cursor()
        c.execute("DELETE FROM alerts")
        conn.commit()
        conn.close()
        return jsonify({"status": "cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete_rule', methods=['POST'])
def delete_rule():
    data = request.get_json()
    cat = data.get("category")
    if cat in ["Allowed", "Not Allowed"]:
        lot = data.get("lot")
        time_frame = data.get("time_frame")
        if not (lot and time_frame):
            return jsonify({"error": "Missing lot or time_frame"}), 400
        try:
            start, end = time_frame.split("|")
        except Exception:
            return jsonify({"error": "Invalid time_frame format"}), 400
        key = (start, end)
        if cat in special_rules and lot in special_rules[cat] and key in special_rules[cat][lot]:
            del special_rules[cat][lot][key]
            return jsonify({"status": "deleted"}), 200
        else:
            return jsonify({"error": "Rule not found"}), 404
    elif cat == "Closed":
        lot = data.get("lot")
        if cat in special_rules and lot in special_rules[cat]:
            del special_rules[cat][lot]
            return jsonify({"status": "deleted"}), 200
        else:
            return jsonify({"error": "Rule not found"}), 404
    elif cat == "New Permits":
        new_permit_name = data.get("new_permit_name")
        if cat in special_rules and new_permit_name in special_rules[cat]:
            del special_rules[cat][new_permit_name]
            return jsonify({"status": "deleted"}), 200
        else:
            return jsonify({"error": "Rule not found"}), 404
    else:
        return jsonify({"error": "Unknown category"}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port, debug=True)
