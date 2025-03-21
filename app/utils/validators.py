def validate_license_plate(plate):
    # Basic validation - just check if not empty for now
    return bool(plate and plate.strip()) 