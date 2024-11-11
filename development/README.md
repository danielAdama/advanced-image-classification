## Project Documentation for AI Chatbot with Redis Integration

### Chat Interface
<img width="1680" alt="Screenshot 2024-09-30 at 14 15 51" src="https://github.com/user-attachments/assets/9528198b-48dc-40c5-a3e0-858df2438345">


### Project Title: **MasteryHive AI Chatbot**

### Overview
The **MasteryHive AI Chatbot** is an AI-powered application built using **FastAPI** and **Google Generative AI** for conversational interactions. It leverages Redis for caching chat messages and session management. The application consists of a backend server handling chat interactions and a frontend interface for users to communicate with the AI chatbot in real-time.

### Table of Contents
1. [Features](#features)
2. [Technology Stack](#technology-stack)
3. [Installation and Setup](#installation-and-setup)
4. [Environment Variables](#environment-variables)
5. [Project Structure](#project-structure)
6. [API Endpoints](#api-endpoints)
7. [Frontend Application](#frontend-application)
8. [Running the Application](#running-the-application)
9. [Contributing](#contributing)
10. [License](#license)

---

### Features
- **Interactive AI Chatbot**: Uses Google Generative AI to engage users in educational conversations.
- **Session Management**: Stores and retrieves conversation history for individual users using Redis.
- **RESTful API**: Offers an API to send and retrieve chat messages.
- **Frontend**: A simple web-based interface for users to interact with the AI.

---

### Technology Stack
- **Backend**:
  - FastAPI: Backend framework for building APIs.
  - Redis: In-memory data structure store for caching messages.
  - Google Generative AI: AI model for conversational responses.
  - Docker: Containerization for easy deployment.

- **Frontend**:
  - HTML, CSS, JavaScript: Simple web-based frontend for chat interaction.

---

### Installation and Setup

#### 1. Clone the Repository:
```bash
git clone https://github.com/your-username/masteryhive-ai-chatbot.git
cd masteryhive-ai-chatbot
```

#### 2. Setup Environment Variables:
Create a `.env` file in the root directory and configure the following variables:

```bash
REDIS_HOST=your_redis_host
REDIS_PORT=your_redis_port
REDIS_PASSWORD=your_redis_password
GEMINI_API_KEY=your_google_generative_ai_api_key
```

#### 3. Install Dependencies:

Backend dependencies:
```bash
pip install -r requirements.txt
```

Frontend dependencies:
No dependencies are required for the frontend as it uses HTML/CSS/JavaScript.

#### 4. Run the Application:
```bash
uvicorn src.main:app --reload
```

The application will be available at `http://localhost:80`.

---

### Environment Variables

- **REDIS_HOST**: The host address of the Redis instance.
- **REDIS_PORT**: The port Redis is running on.
- **REDIS_PASSWORD**: The password for Redis authentication.
- **GEMINI_API_KEY**: The API key for Google Generative AI.

---

### Project Structure

```plaintext
├── gen_ai/
│   ├── __init__.py          # Core AI chatbot logic
│   ├── prompt/
│   │   └── templates/
│   │       └── system_template.txt  # System prompt for the chatbot
├── src/
│   ├── controllers/
│   │   └── chat.py          # Chat API controller
│   ├── schemas/
│   │   └── chat_schema.py   # Pydantic models for chat input/output
│   ├── services/
│   │   └── chat_service.py  # Chat service interacting with the AI and Redis
│   ├── utils/
│   │   ├── app_utils.py     # Utility functions
│   │   ├── request_response.py  # API response helper
│   │   └── connection.py    # Redis connection management
│   ├── main.py              # FastAPI application entry point
├── frontend/
│   ├── index.html           # Frontend interface for the chatbot
│   ├── script.js            # JavaScript for frontend chat logic
│   └── styles.css           # CSS for frontend styling
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration for the project
```

---

### API Endpoints

#### `POST /v1/chat/messages/`
Send a message to the chatbot.

- **Request Body**:
  ```json
  {
    "question": "What is photosynthesis?",
    "user_id": "test-user"
  }
  ```

- **Response**:
  ```json
  {
    "message": "Chat added successfully",
    "data": {
      "question": "What is photosynthesis?",
      "response": "Photosynthesis is a process used by plants to convert light energy into chemical energy..."
    }
  }
  ```

#### `GET /`
Returns the status of the API.

---

### Makefile Commands
The project includes a Makefile with the following commands:

* compose_up: Runs Docker Compose to start the services.
* compose_down: Shuts down the Docker Compose services.
* start_dev: Runs the application in development mode.
* start_prod: Runs the application in production mode.

### Frontend Application

The frontend is a simple HTML-based chat interface.

- **index.html**: Main chat UI with an input field and a "Send" button.
- **script.js**: Handles sending messages to the backend and displaying responses.
- **styles.css**: Contains styles for the chat interface.

---

### Running the Application
1. Run the Application
  Use the following command to run the application in development mode:
  ```sh
  make start_dev
  ```
  For production mode, start the FastAPI backend using:
  ```sh
  make start_prod
  ```

2. Open the `frontend/index.html` file in a browser to interact with the chatbot.

3. Use the input field to send a message, and the chatbot will respond based on the AI model.

---

### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
