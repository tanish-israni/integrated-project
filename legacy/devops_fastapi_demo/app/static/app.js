const API_URL = window.location.origin;

// Initialize on page load
window.addEventListener('load', () => {
  checkRoomAvailability('standard', 'standardAvail');
  checkRoomAvailability('deluxe', 'deluxeAvail');
  checkAvailability();
});

// Auto-refresh every 30 seconds
setInterval(() => {
  checkRoomAvailability('standard', 'standardAvail');
  checkRoomAvailability('deluxe', 'deluxeAvail');
}, 30000);

async function checkRoomAvailability(roomType, elementId) {
  try {
    const response = await fetch(`${API_URL}/availability?room_type=${roomType}`);
    const data = await response.json();
    
    const element = document.getElementById(elementId);
    if (!element) return;

    if (data.available) {
      const roomsText = data.rooms_left === 1 ? 'room' : 'rooms';
      element.innerHTML = `
        <span class="badge available">
          <i class="fas fa-check"></i> ${data.rooms_left} ${roomsText} available
        </span>
      `;
    } else {
      element.innerHTML = `
        <span class="badge unavailable">
          <i class="fas fa-times"></i> Fully booked
        </span>
      `;
    }
  } catch (error) {
    console.error('Error:', error);
    const element = document.getElementById(elementId);
    if (element) {
      element.innerHTML = `<span class="badge unavailable"><i class="fas fa-exclamation"></i> Error</span>`;
    }
  }
}

async function checkAvailability() {
  const roomType = document.getElementById('roomTypeCheck').value;
  const resultDiv = document.getElementById('availabilityResult');
  resultDiv.innerHTML = '<div class="result loading"><i class="fas fa-spinner fa-spin"></i> <span>Checking...</span></div>';
  
  try {
    const response = await fetch(`${API_URL}/availability?room_type=${roomType}`);
    const data = await response.json();
    
    updateStatus('Ready', 'ready');
    
    if (data.available) {
      const roomsText = data.rooms_left === 1 ? 'room' : 'rooms';
      resultDiv.innerHTML = `
        <div class="result available">
          <i class="fas fa-check"></i>
          <div><strong>${data.rooms_left} ${roomsText} available</strong><br><span style="font-size: 12px; opacity: 0.8;">You can book now</span></div>
        </div>
      `;
    } else {
      resultDiv.innerHTML = `
        <div class="result unavailable">
          <i class="fas fa-times"></i>
          <div><strong>No rooms available</strong><br><span style="font-size: 12px; opacity: 0.8;">Try another room type</span></div>
        </div>
      `;
    }
  } catch (error) {
    resultDiv.innerHTML = `
      <div class="result unavailable">
        <i class="fas fa-exclamation"></i>
        <div><strong>Error checking availability</strong></div>
      </div>
    `;
    updateStatus('Error', 'error');
  }
}

async function bookRoom() {
  const roomType = document.getElementById('roomTypeBook').value;
  const nights = parseInt(document.getElementById('nights').value);
  const resultDiv = document.getElementById('bookingResult');
  
  // Validation
  if (!nights || nights < 1 || nights > 30) {
    resultDiv.innerHTML = `
      <div class="result unavailable">
        <i class="fas fa-exclamation"></i>
        <div><strong>Invalid input</strong><br><span style="font-size: 12px; opacity: 0.8;">Enter 1-30 nights</span></div>
      </div>
    `;
    return;
  }
  
  resultDiv.innerHTML = '<div class="result loading"><i class="fas fa-spinner fa-spin"></i> <span>Booking...</span></div>';
  updateStatus('Booking...', 'booking');
  
  try {
    const response = await fetch(`${API_URL}/book`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room_type: roomType, nights: nights })
    });
    
    const data = await response.json();
    
    if (data.status === 'confirmed') {
      const totalNights = nights === 1 ? '1 night' : `${nights} nights`;
      const roomName = roomType.charAt(0).toUpperCase() + roomType.slice(1);
      resultDiv.innerHTML = `
        <div class="result available">
          <i class="fas fa-check"></i>
          <div>
            <strong>Booking Confirmed!</strong><br>
            <span style="font-size: 12px; opacity: 0.9; line-height: 1.6;">
              Duration: ${totalNights}<br>
              ID: <strong>${data.booking_id}</strong><br>
              Room: ${roomName}
            </span>
          </div>
        </div>
      `;
      
      // Refresh availability
      checkRoomAvailability('standard', 'standardAvail');
      checkRoomAvailability('deluxe', 'deluxeAvail');
      checkAvailability();
    } else {
      resultDiv.innerHTML = `
        <div class="result unavailable">
          <i class="fas fa-times"></i>
          <div><strong>Booking failed</strong><br><span style="font-size: 12px; opacity: 0.8;">${data.message || 'Unable to complete booking'}</span></div>
        </div>
      `;
    }
    
    updateStatus('Ready', 'ready');
  } catch (error) {
    resultDiv.innerHTML = `
      <div class="result unavailable">
        <i class="fas fa-exclamation"></i>
        <div><strong>Booking error</strong></div>
      </div>
    `;
    updateStatus('Error', 'error');
  }
}

function updateStatus(text, type) {
  const statusElement = document.getElementById('statusText');
  const dotElement = document.querySelector('.status-dot');
  
  if (!statusElement || !dotElement) return;

  statusElement.textContent = text;
  
  const colorMap = {
    ready: '#27ae60',
    booking: '#f39c12',
    error: '#e74c3c'
  };
  
  dotElement.style.background = colorMap[type] || '#27ae60';
}
