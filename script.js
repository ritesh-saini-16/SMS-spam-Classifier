
// Predict function using our API
async function predict() {
    const inputText = document.getElementById('inputText').value.trim();
    const resultDiv = document.getElementById('result');
    
    if (!inputText) {
        resultDiv.textContent = 'Please enter a message to classify.';
        resultDiv.className = '';
        return;
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: inputText })
        });
        
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Error during prediction');
        }
        
        const prediction = data.prediction;
        resultDiv.className = prediction === 1 ? 'spam' : 'not-spam';
        resultDiv.textContent = prediction === 1 ? 'Spam' : 'Not Spam';
    } catch (error) {
        console.error('Prediction error:', error);
        resultDiv.textContent = 'Error during prediction: ' + error.message;
        resultDiv.className = '';
    }
}

// Event listeners
document.getElementById('predictBtn').addEventListener('click', predict);
document.getElementById('inputText').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.shiftKey === false) {
        e.preventDefault();
        predict();
    }
});
