# AI-Powered Parking Guidance Chatbot

Click [here](https://drive.google.com/drive/folders/1zUFSnA89EQCtH64fJHWZMUL1JggyswtD?usp=sharing) to see python scripts for the admin console and AI chatbot

Declaration: We used AI for ideation, code generation and front end implementation ✨ 

## Overview
This project introduces an **AI-powered Parking Guidance Chatbot** that leverages **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to provide a seamless and interactive parking experience for users. The system is designed to address the complexities of parking permissions at the University of Maryland (UMD) by enabling natural conversations with users and providing real-time, accurate parking guidance.

---

## Key Features

### 1. User-Facing Chatbot
- Users can naturally converse with the chatbot to determine:
  - Whether they are allowed to park in a specific lot at a given time with their permit.
  - Alternative parking lots if they are not allowed to park in their desired lot.
  - If restrictions change after a certain time, allowing them to park later.
  - The type of violation and fine they would incur for wrongful parking.

### 2. Admin System for Rule Updates
- The admin side of the system uses AI to address the issue of frequent updates to parking permissions caused by:
  - Events
  - Game matches
  - Construction
  - Other disruptions requiring access to specific parking lots
- Admins can update parking rules dynamically, and alerts are logged and automatically expire when their effective duration ends.

---

## Algorithm Workflow
1. **User Query**:
   - The user sends a query to the chatbot, including:
     - Permit type
     - Lot number
     - Day and time
2. **Rule Database Check**:
   - The chatbot scans the permissions rule database and passes the query through an ML model.
3. **Response**:
   - The chatbot replies with "Yes" or "No" depending on whether the user is allowed to park.
   - If the query does not exist in the rules database:
     - The system alerts the admin to add new rules to the server.
4. **Training Data**:
   - The databases on the server are used to train the ML model.
   - Since every rule in the dataset is unique and deterministic, the training dataset is also used as the testing dataset, resulting in an accuracy of **100%**.

---

## Interactive UI
The system includes an intuitive UI with an interactive map of UMD's campus featuring all parking lots. Key features include:
- **Color-Coded Map**:
  - Parking lots are color-coded based on their permission types.
- **User Input**:
  - Users can input date, time, and duration of parking to check:
    - Whether they can park in a specific lot.
    - Violation details and fines if they park wrongfully.
    - Alternative parking lots based on proximity and permission availability.
- **Navigation Integration**:
  - Users can navigate directly to parking lots via Google Maps or Apple Maps.

---

## Future Enhancements
1. **Smart Permit Validation & Fraud Detection**  
   Use AI to automatically validate permit details and flag inconsistencies or potential fraud in real-time.

2. **Integration with External Data Sources**  
   Enhance functionality by integrating with external APIs such as city traffic data, weather forecasts, and event schedules for contextualized parking insights.

3. **Anomaly Detection & Predictive Maintenance**  
   Apply anomaly detection models to monitor unusual parking patterns or rule violations, enabling proactive enforcement and maintenance scheduling.

4. **Scenario Simulation & Forecasting**  
   Build simulation models for "what-if" analyses to predict how changes in parking policies or campus events might impact parking availability and compliance.

---

## Environment Setup
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to `.env`
3. Never commit `.env` file

