const apiUrl = "http://localhost:80/v1/chat/messages/";

function displayMessage(message, role) {
    const chatMessages = document.getElementById("chat-messages");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", role);
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const inputElement = document.getElementById("chat-input");
    const userMessage = inputElement.value;

    if (userMessage.trim() === "") return;

    displayMessage(userMessage, "user");

    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question: userMessage,
                user_id: "adamadaniel321@gmail.com"
            }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        const botMessage = result.data;
        displayMessage(botMessage, "model");

        inputElement.value = "";
    } catch (error) {
        console.error('Error:', error);
    }
}
