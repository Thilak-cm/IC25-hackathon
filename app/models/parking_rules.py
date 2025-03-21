from flask import current_app
from datetime import datetime
import random

def check_parking_eligibility(license_plate, lot_name, user_type, current_time):
    lots = getattr(current_app, 'parking_lots', {})
    lot_info = lots.get(lot_name, {})
    
    if not lot_info:
        return {
            'status': 'error',
            'message': f'Lot {lot_name} not found',
            'alternatives': ['Lot 1', 'Lot 6', 'Lot 11']
        }
    
    # Random response for demo
    allowed = random.choice([True, False])
    
    if allowed:
        return {
            'status': 'allowed',
            'message': f'You can park in {lot_name}',
            'alternatives': []
        }
    else:
        return {
            'status': 'denied',
            'message': f'Parking not allowed in {lot_name}',
            'alternatives': ['Lot 1', 'Lot 6', 'Lot 11']
        } 