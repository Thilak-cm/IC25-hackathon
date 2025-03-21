function validateInputs() {
    const permit = document.getElementById('permit').value;
    const lot = document.getElementById('lot').value;
    const day = document.getElementById('day').value;
    const time = document.getElementById('time').value;
    const errors = [];

    // Validate permit number
    if (!/^[A-Z0-9]+$/.test(permit)) {
        errors.push("Permit number should only contain uppercase letters and numbers.");
    }

    // Validate lot name
    if (!lot) {
        errors.push("Lot name cannot be empty.");
    }

    // Validate day
    const validDays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
    if (!validDays.includes(day.toLowerCase())) {
        errors.push("Day must be a valid weekday (e.g., Monday, Tuesday).");
    }

    // Validate time format and range
    const timeParts = time.split(':');
    if (timeParts.length !== 3 || !timeParts.every(part => /^\d+$/.test(part))) {
        errors.push("Time must be in HH:MM:SS format.");
    } else {
        const [hours, minutes, seconds] = timeParts.map(Number);
        if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59 || seconds < 0 || seconds > 59) {
            errors.push("Time values must be within valid ranges: 00:00:00 to 23:59:59.");
        }
    }

    if (errors.length > 0) {
        alert("Please correct the following errors:\n" + errors.join("\n"));
        return false;
    }
    return true;
}

async function checkEligibility() {
    if (!validateInputs()) {
        return;
    }

    const permit = document.getElementById('permit').value;
    const lot = document.getElementById('lot').value;
    const day = document.getElementById('day').value;
    const time = document.getElementById('time').value;
    
    try {
        const response = await fetch('/check_parking_eligibility', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                permit: permit,
                lot: lot,
                day: day,
                time: time
            })
        });

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error("Error checking eligibility:", error);
        alert("An error occurred while checking eligibility. Please try again.");
    }
}

function displayResult(result) {
    const resultContainer = document.getElementById('result');
    const lookup = document.getElementById('lookup');
    const modelPrediction = document.getElementById('modelPrediction');
    const allowedPermits = document.getElementById('allowedPermits');

    if (result.status === 'error' && result.closest_matches) {
        lookup.textContent = result.message;
        modelPrediction.innerHTML = 'Closest Matches:<br>' + Object.entries(result.closest_matches)
            .map(([key, value]) => `<button onclick="selectLot('${key}', '${value}')">${key}: ${value}</button>`)
            .join('<br>');
        allowedPermits.textContent = '';
    } else {
        lookup.textContent = `Parking Permission for ${result.permit || ''} with permit ${result.lot || ''} is ${result.status === 'success' ? 'allowed' : 'not allowed'}`;
        modelPrediction.textContent = result.message;
        allowedPermits.textContent = result.allowed_permits && result.allowed_permits.length > 0 
            ? `Only the following permits can park here: ${result.allowed_permits.join(', ')}`
            : '';
    }

    resultContainer.classList.remove('hidden');

    if (!result.continue) {
        console.log("Process completed.");
    }
}

function selectLot(key, lot) {
    document.getElementById('lot').value = lot;
    document.getElementById('result').classList.add('hidden');
    checkEligibility();
}
