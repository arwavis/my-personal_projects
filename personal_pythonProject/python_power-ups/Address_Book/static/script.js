function submitContact() {
    const formData = {
        firstName: document.getElementById('firstName').value,
        lastName: document.getElementById('lastName').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        phoneType: document.getElementById('phoneType').value,
        address: document.getElementById('address').value,
        addressType: document.getElementById('addressType').value
    };

    fetch('/contact', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        document.getElementById('contactForm').reset(); // Reset form after submission
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
