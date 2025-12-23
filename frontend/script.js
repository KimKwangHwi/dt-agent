document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send-button');
    const questionInput = document.getElementById('question-input');
    const resultContainer = document.getElementById('result-container');

    sendButton.addEventListener('click', handleSend);
    questionInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            handleSend();
        }
    });

    async function handleSend() {
        const question = questionInput.value.trim();
        if (!question) return;

        // Display user's question
        appendMessage('You', question);
        questionInput.value = '';
        
        // Show loading indicator
        appendMessage('Agent', 'Thinking...');

        try {
            // This is where we call our backend API
            const response = await fetch('http://127.0.0.1:8001/invoke', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            
            // Remove "Thinking..." and display agent's answer
            resultContainer.lastChild.remove(); // Removes the loading message
            appendMessage('Agent', data.answer);

        } catch (error) {
            console.error('Error:', error);
            resultContainer.lastChild.remove(); // Removes the loading message
            appendMessage('Agent', 'Sorry, something went wrong.');
        }
    }

    function appendMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message-block');

        // Only parse markdown for messages from the Agent
        const content = (sender === 'Agent' && typeof marked !== 'undefined')
            ? marked.parse(text)
            : text;

        messageElement.innerHTML = `<strong>${sender}:</strong><div class="message-content">${content}</div>`;
        resultContainer.appendChild(messageElement);
        resultContainer.scrollTop = resultContainer.scrollHeight; // Scroll to the bottom
    }
});
