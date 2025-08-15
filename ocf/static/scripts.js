
        
function updatePlaceholders() {
    const unitSystem = document.querySelector('input[name="unit_system"]:checked').value;

    document.getElementById("channelSlope").placeholder = `Slope in ${unitSystem === 'm' ? 'm/m' : 'ft/ft'}`;
    document.getElementById("waterDepth").placeholder = `Depth in ${unitSystem}`;
    document.getElementById("bottomWidth").placeholder = `Width in ${unitSystem}`;
    document.getElementById("flow").placeholder = `Discharge in ${unitSystem === 'm' ? 'm³/s' : 'ft³/s'}`;
    document.getElementById("velocity").placeholder = `Velocity in ${unitSystem === 'm' ? 'm/s' : 'ft/s'}`;
}

function toggleZInputs() {
    const channelType = document.getElementById('channelType').value;
    const zInputsRow = document.getElementById('zInputsRow');
    const aInputsRow=document.getElementById('aInputsRow');
    const z1Input = document.getElementById('z1');
    const z2Input = document.getElementById('z2');
    const a_dField = document.getElementById('a_d');
    if(channelType === 'Circle') {
        aInputsRow.classList.remove('hidden');
        zInputsRow.classList.add('hidden');
        z1Input.removeAttribute('required');
        z2Input.removeAttribute('required');
        a_dField.setAttribute('required', '');

    }


    if (channelType === 'Circle'|| channelType === 'Rectangle') {
        zInputsRow.classList.add('hidden');
        z1Input.removeAttribute('required');
        z2Input.removeAttribute('required');
       
    } else {
        zInputsRow.classList.remove('hidden');
        z1Input.setAttribute('required', '');
        z2Input.setAttribute('required', '');
        
    }
    
}
//  function circle_field(){
//     const c_t =document.getElementById('channelType').value;
//     const circleInputsRow = document.getElementById('aInputsRow');
//     const a_dField = document.getElementById('a_d');
    
//     if (c_t === 'Circle') {

//     circleInputsRow.classList.remove('hidden');
        
//         a_dField.setAttribute('required', '');
//     } 
//     else {
//         circleInputsRow.classList.add('hidden');
        
//         a_dField.removeAttribute('required');
//     }
   
//  }

function toggleVelocityFlowFields() {
    const calculationType = document.getElementById('calculationType').value;
    const velocityField = document.getElementById('velocity');
    const flowField = document.getElementById('flow');
    const channelSlopeField = document.getElementById('channelSlope');

    // Reset fields
    velocityField.disabled = false;
    flowField.disabled = false;
    channelSlopeField.disabled = false;

    switch(calculationType) {
        case 'velocity_discharge':
            velocityField.disabled = true;
            flowField.disabled = true;
            break;
        case 'channel_slope_v':
            channelSlopeField.disabled = true;
            flowField.disabled = true;
            velocityField.disabled = false;
            break;
        case 'channel_slope_q':
            channelSlopeField.disabled = true;
            velocityField.disabled = true;
            break;
        case 'manning_v':
            flowField.disabled = true;
            break;
        case 'manning_q':
            velocityField.disabled = true;
            break;
        case 'depth_q':
            velocityField.disabled = true;
            break;
    }
}
function convertUnits(unit) {
const lengthConversionFactor = unit === 'm' ? 0.3048 : 1 / 0.3048;
const flowConversionFactor = unit === 'm' ? 0.0283168 : 1 / 0.0283168;
const areaConversionFactor = lengthConversionFactor ** 2;

const lengthFields = ['waterDepth', 'bottomWidth', 't-w','w-p','h-r'];
const velocityFields = ['velocity'];
const flowFields = ['flow'];
const areaFields = ['flow-area'];

lengthFields.forEach(fieldId => {
    const field = document.getElementById(fieldId);
    if (field && field.value) {
        field.value = (parseFloat(field.value) * lengthConversionFactor).toFixed(4);
    }
});

velocityFields.forEach(fieldId => {
    const field = document.getElementById(fieldId);
    if (field && field.value) {
        field.value = (parseFloat(field.value) * lengthConversionFactor).toFixed(4);
    }
});

flowFields.forEach(fieldId => {
    const field = document.getElementById(fieldId);
    if (field && field.value) {
        field.value = (parseFloat(field.value) * flowConversionFactor).toFixed(4);
    }
});

areaFields.forEach(fieldId => {
    const field = document.getElementById(fieldId);
    if (field && field.value) {
        field.value = (parseFloat(field.value) * areaConversionFactor).toFixed(4);
    }
});
}



document.getElementById('flowForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const form = event.target;
    const formData = new FormData(form);

    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': formData.get('csrfmiddlewaretoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response data
        console.log(data);
        // Update the form fields with the response data if needed
        // For example:
        document.getElementById('flow-area').value = data.flow_area;
    document.getElementById('w-p').value = data.wetted_perimeter;
    document.getElementById('h-r').value = data.hydraulic_radius;
    document.getElementById('t-w').value = data.top_width;
    document.getElementById('flow').value = data.flow;
    document.getElementById('velocity').value = data.velocity;
    document.getElementById('channelSlope').value = data.channel_slope;
    document.getElementById('t-w').value = data.topwidth;
    //document.getElementById('m').value = data.n;
    
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function handleReset() {
    const form = document.getElementById('flowForm');
    form.reset();

    // Enable the input fields for velocity and discharge
    document.getElementById('velocity').disabled = false;
    document.getElementById('flow').disabled = false;
    document.getElementById('channelSlope').disabled = false;
}


function showLastFiveInputs() {
    const lastFiveInputs = document.querySelectorAll('.form-control:not(.hidden)');
    lastFiveInputs.forEach(input => input.classList.remove('hidden'));
}

// Initialize event listeners
document.getElementById('channelType').addEventListener('change', toggleZInputs);
document.getElementById('calculationType').addEventListener('change', toggleVelocityFlowFields);
document.getElementById('unitSystem').addEventListener('change', function() {
    const unit = this.value;
    convertUnits(unit);
});
document.getElementById('channelType').addEventListener('change', circle_field);

// Initial setup
showLastFiveInputs();
circle_field();
updatePlaceholders();
selectRoughnessCoefficient();
toggleZInputs();
toggleVelocityFlowFields();
