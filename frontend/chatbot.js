document.addEventListener("DOMContentLoaded", () => {

    // =========================================================
    // GET CHAT ELEMENTS
    // =========================================================

    const chatButton = document.getElementById("chatButton");
    const chatWindow = document.getElementById("chatWindow");
    const sendButton = document.getElementById("sendChatBtn");
    const input = document.getElementById("chatInput");
    const messages = document.getElementById("chatMessages");


    const newChatButton = document.getElementById("newChatBtn");


    // =========================================================
    // CHECK REQUIRED ELEMENTS
    // =========================================================

    if (
        !chatButton ||
        !chatWindow ||
        !sendButton ||
        !input ||
        !messages
    ) {
        console.error(
            "Chatbot initialization failed: Required HTML elements not found."
        );
        return;
    }


    // =========================================================
    // CHAT STATE
    // =========================================================

    let currentController = null;
    let stopTyping = false;
    let isProcessing = false;

    // Create unique session ID for n8n conversation memory
    let sessionId = generateSessionId();


    // =========================================================
    // GENERATE UNIQUE SESSION ID
    // =========================================================

    function generateSessionId() {

        if (
            typeof crypto !== "undefined" &&
            typeof crypto.randomUUID === "function"
        ) {
            return crypto.randomUUID();
        }

        return (
            "xrayai-" +
            Date.now() +
            "-" +
            Math.random().toString(36).substring(2, 10)
        );

    }


    // =========================================================
    // OPEN / CLOSE CHAT WINDOW
    // =========================================================

    chatButton.addEventListener("click", () => {

        chatWindow.classList.toggle("hidden");

        if (!chatWindow.classList.contains("hidden")) {
            input.focus();
        }

    });


    // =========================================================
// CLOSE CHAT WINDOW WHEN CLICKING OUTSIDE
// =========================================================

document.addEventListener("click", (event) => {

    // Do nothing if chat window is already closed
    if (chatWindow.classList.contains("hidden")) {
        return;
    }

    // Check if user clicked inside the chat window
    const clickedInsideChat =
        chatWindow.contains(event.target);

    // Check if user clicked the floating chat button
    const clickedChatButton =
        chatButton.contains(event.target);

    // Close only when clicking outside both
    if (!clickedInsideChat && !clickedChatButton) {

        chatWindow.classList.add("hidden");

    }

});

    // =========================================================
    // SEND MESSAGE TO N8N CHATBOT
    // =========================================================

    async function sendMessageToBot(message, signal) {

        const response = await fetch(
            "http://localhost:5678/webhook/0a8d732a-5482-478a-b4e4-d6a4e9580fb8",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    message: message,
                    sessionId: sessionId
                }),

                signal: signal
            }
        );


        // Check HTTP response
        if (!response.ok) {

            throw new Error(
                `Chatbot server error: ${response.status}`
            );

        }


        // Read JSON response
        const data = await response.json();

        console.log(
            "n8n chatbot response:",
            data
        );


        // =====================================================
        // HANDLE DIFFERENT N8N RESPONSE FORMATS
        // =====================================================

        if (Array.isArray(data)) {

            return (
                data[0]?.reply ||
                data[0]?.output ||
                data[0]?.text ||
                "No response received."
            );

        }


        return (
            data.reply ||
            data.output ||
            data.text ||
            "No response received."
        );

    }


    // =========================================================
    // TYPEWRITER + MARKDOWN EFFECT
    // =========================================================

    async function typeWriter(element, text) {

        element.classList.add("ai-message");

        stopTyping = false;


        // =====================================================
        // CHECK MARKED.JS
        // =====================================================

        if (typeof marked === "undefined") {

            console.error(
                "Marked.js is not loaded. Showing plain text instead."
            );

            element.textContent = text;

            messages.scrollTop =
                messages.scrollHeight;

            return;

        }


        const fullText = String(text);

        const words =
            fullText.split(/\s+/);

        let displayedText = "";


        // =====================================================
        // DISPLAY RESPONSE WORD BY WORD
        // =====================================================

        for (
            let i = 0;
            i < words.length;
            i++
        ) {

            // Stop immediately if Stop button is clicked
            if (stopTyping) {

                element.innerHTML =
                    marked.parse(displayedText);

                return;

            }


            displayedText +=
                (i === 0 ? "" : " ") +
                words[i];


            // Render Markdown
            element.innerHTML =
                marked.parse(displayedText);


            // Auto scroll
            messages.scrollTop =
                messages.scrollHeight;


            // Typing speed
            await new Promise(
                resolve =>
                    setTimeout(resolve, 20)
            );

        }


        // =====================================================
        // FINAL CLEAN MARKDOWN RENDER
        // =====================================================

        if (!stopTyping) {

            element.innerHTML =
                marked.parse(fullText);

        }


        messages.scrollTop =
            messages.scrollHeight;

    }


    // =========================================================
    // STOP CURRENT RESPONSE
    // =========================================================

    // =========================================================
// CHANGE SEND BUTTON TO STOP BUTTON
// =========================================================

function showStopButton() {

    sendButton.disabled = false;

    sendButton.classList.add("stop-mode");

    sendButton.innerHTML = `
        <span class="stop-icon"></span>
    `;

    sendButton.title = "Stop response";
}

// =========================================================
// CHANGE STOP BUTTON BACK TO SEND BUTTON
// =========================================================

function showSendButton() {

    sendButton.classList.remove("stop-mode");

    sendButton.textContent = "Send";

    sendButton.disabled = false;

}


// =========================================================
// STOP CURRENT RESPONSE
// =========================================================

function stopCurrentResponse() {

    // Stop typewriter animation
    stopTyping = true;


    // Cancel active n8n request
    if (currentController) {

        currentController.abort();

        currentController = null;

    }


    // Remove typing indicator if visible
    const typingIndicator =
        messages.querySelector(
            ".typing-indicator"
        );


    if (typingIndicator) {

        typingIndicator.remove();

    }


    // Reset processing state
    isProcessing = false;


    // Change Stop button back to Send
    showSendButton();


    // Allow user to type again
    input.focus();

}


    // =========================================================
    // STOP BUTTON
    // =========================================================

   


    // =========================================================
    // NEW CHAT BUTTON
    // =========================================================

    if (newChatButton) {

        newChatButton.addEventListener(
            "click",
            () => {

                // Stop current response/request
                stopCurrentResponse();


                // Clear all messages
                messages.innerHTML = "";


                // Generate completely new conversation ID
                sessionId =
                    generateSessionId();


                console.log(
                    "New XrayAI chat session:",
                    sessionId
                );


                // Clear input
                input.value = "";


                // Optional welcome message
                const welcomeMessage =
                    document.createElement("div");

                welcomeMessage.classList.add(
                    "ai-message"
                );

                welcomeMessage.innerHTML = `
                    <strong>🫁 XrayAI Assistant</strong>
                    <p>New chat started. How can I help you with a medical or health-related question?</p>
                `;

                messages.appendChild(
                    welcomeMessage
                );


                messages.scrollTop =
                    messages.scrollHeight;


                input.focus();

            }
        );

    }


    // =========================================================
    // SEND MESSAGE FUNCTION
    // =========================================================

    async function handleSendMessage() {

        const message =
            input.value.trim();


        // Ignore empty messages
        if (!message) {
            return;
        }


        // Prevent multiple simultaneous requests
        if (isProcessing) {
            return;
        }


        isProcessing = true;

        stopTyping = false;


        // Create AbortController for this request
        currentController =
            new AbortController();


        // =====================================================
        // SHOW USER MESSAGE
        // =====================================================

        const userMessage =
            document.createElement("div");


        const userLabel =
            document.createElement("b");


        userLabel.textContent =
            "You: ";


        userMessage.appendChild(
            userLabel
        );


        userMessage.appendChild(
            document.createTextNode(
                message
            )
        );


        messages.appendChild(
            userMessage
        );


        messages.scrollTop =
            messages.scrollHeight;


        // Clear input
input.value = "";


// Convert Send button into animated Stop button
showStopButton();


        // =====================================================
        // SHOW AI TYPING INDICATOR
        // =====================================================

        const typingIndicator =
            document.createElement("div");


        typingIndicator.classList.add(
            "typing-indicator"
        );


        typingIndicator.innerHTML = `
            <span class="typing-label">
                AI is typing
            </span>

            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;


        messages.appendChild(
            typingIndicator
        );


        messages.scrollTop =
            messages.scrollHeight;


        try {

            // =================================================
            // GET AI RESPONSE
            // =================================================

            const reply =
                await sendMessageToBot(
                    message,
                    currentController.signal
                );


            // Request finished
            currentController = null;


            // Remove typing indicator
            if (
                typingIndicator &&
                typingIndicator.isConnected
            ) {

                typingIndicator.remove();

            }


            // If user pressed Stop while request was finishing
            if (stopTyping) {
                return;
            }


            // =================================================
            // CREATE AI MESSAGE
            // =================================================

            const aiMessage =
                document.createElement("div");


            messages.appendChild(
                aiMessage
            );


            // =================================================
            // DISPLAY RESPONSE WITH TYPEWRITER
            // =================================================

            await typeWriter(
                aiMessage,
                reply
            );


            messages.scrollTop =
                messages.scrollHeight;


        } catch (error) {


            // Remove typing indicator
            if (
                typingIndicator &&
                typingIndicator.isConnected
            ) {

                typingIndicator.remove();

            }


            // =================================================
            // HANDLE USER CANCELLATION
            // =================================================

            if (error.name === "AbortError") {

                console.log(
                    "Chatbot request stopped by user."
                );

                return;

            }


            // =================================================
            // HANDLE REAL ERROR
            // =================================================

            console.error(
                "Chatbot error:",
                error
            );


            const errorMessage =
                document.createElement("div");


            errorMessage.classList.add(
                "ai-message"
            );


            const errorLabel =
                document.createElement("b");


            errorLabel.textContent =
                "AI: ";


            errorMessage.appendChild(
                errorLabel
            );


            errorMessage.appendChild(
                document.createTextNode(
                    "Sorry, I couldn't connect to the assistant."
                )
            );


            messages.appendChild(
                errorMessage
            );


            messages.scrollTop =
                messages.scrollHeight;


        } finally {

            currentController = null;

isProcessing = false;

// Automatically return Stop button to Send
showSendButton();

input.focus();

        }

    }


    // =========================================================
    // SEND BUTTON CLICK
    // =========================================================

    // =========================================================
// SEND / STOP BUTTON CLICK
// =========================================================

sendButton.addEventListener(
    "click",
    () => {

        // AI is currently working:
        // Button behaves as STOP
        if (isProcessing) {

            stopCurrentResponse();

            return;

        }


        // AI is idle:
        // Button behaves as SEND
        handleSendMessage();

    }
);

    // =========================================================
    // PRESS ENTER TO SEND
    // =========================================================

    input.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                handleSendMessage();

            }

        }
    );

});