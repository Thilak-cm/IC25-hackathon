import requests
from pynput import keyboard

def post_alert():
    url = "http://localhost:1000/log_alert"
    payload = {
        "alert_message": "Test Alert Triggered",
        "details": "Alert triggered by key press 'a'"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Alert posted")
        else:
            print("Failed to post alert:", response.text)
    except Exception as e:
        print("Error posting alert:", e)

def fetch_rules():
    url = "http://localhost:1000/get_restrictions"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            rules = response.json()
            print("Fetched Rules:")
            print(rules)
        else:
            print("Failed to fetch rules:", response.text)
    except Exception as e:
        print("Error fetching rules:", e)

def on_press(key):
    try:
        if key.char == 'a':
            post_alert()
        elif key.char == 'r':
            fetch_rules()
        elif key.char == 'q':
            print("Quitting...")
            return False
    except AttributeError:
        # Handle special keys without 'char' attribute.
        pass

def main():
    print("Press 'a' to send an alert, 'r' to fetch rules, 'q' to quit.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    main()
