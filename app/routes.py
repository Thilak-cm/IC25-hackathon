from flask import Blueprint, render_template, request, jsonify
from app.models.parking_rules import check_parking_eligibility
from app.utils.validators import validate_license_plate

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/check_parking', methods=['POST'])
def check_parking():
    data = request.get_json()
    
    # Extract data from request
    license_plate = data.get('license_plate')
    lot_name = data.get('lot_name')
    user_type = data.get('user_type')
    current_time = data.get('time')
    
    # Validate inputs
    if not validate_license_plate(license_plate):
        return jsonify({
            'status': 'error',
            'message': 'Invalid license plate format'
        })
    
    # Check parking eligibility
    result = check_parking_eligibility(
        license_plate=license_plate,
        lot_name=lot_name,
        user_type=user_type,
        current_time=current_time
    )
    
    return jsonify(result)
