# AI Chatbot

A sophisticated hybrid chatbot application that combines **spaCy** for rule-based natural language processing and **Hugging Face Transformers** (BlenderBot) for generative conversational AI. The project includes both a web-based interface using Streamlit and a command-line interface.

## Features

- **Hybrid AI Architecture**: Combines rule-based NLP (spaCy) with generative AI (Hugging Face BlenderBot)
- **Dual Interfaces**: 
  - Web-based Streamlit application (`app.py`)
  - Command-line interface (`chat_bot.py`)
- **Smart Intent Recognition**: Uses spaCy for detecting specific intents and entities
- **Conversational Memory**: Maintains conversation context with automatic memory management
- **Time Awareness**: Can provide current local time
- **Entity Recognition**: Recognizes persons, locations (GPE), and other named entities

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** (Python 3.12 recommended)
- **pip** (Python package manager)
- **Git** (optional, for cloning repositories)
- **Internet connection** (required for downloading models)

### Verify Python Installation

Open your terminal/command prompt and check:

```bash
python --version
```

or

```bash
python3 --version
```

You should see Python 3.8 or higher.

## Installation Steps

### Step 1: Clone or Navigate to Project Directory

If you have the project in a repository:

```bash
git clone <repository-url>
cd Task3
```

### Step 2: Create a Virtual Environment

Creating a virtual environment is **highly recommended** to avoid dependency conflicts:

**For Windows (PowerShell):**
```powershell
python -m venv venv
```

**For Windows (Command Prompt):**
```cmd
python -m venv venv
```

**For macOS/Linux:**
```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

**For Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

If you encounter execution policy errors, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**For Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**For macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt when activated.

### Step 4: Install Required Packages

Install the core dependencies:

```bash
pip install streamlit spacy transformers torch
```

**Note**: This may take several minutes as it downloads large packages.

### Step 5: Download spaCy Language Model

The application requires the English spaCy language model. Install it using:

```bash
python -m spacy download en_core_web_sm
```

**Note**: This downloads approximately 15-20 MB.

### Step 6: Verify Installation

Verify that all packages are installed correctly:

```bash
pip list
```

You should see:
- streamlit
- spacy
- transformers
- torch
- en_core_web_sm (as a separate package)

## Running the Application

The project includes two ways to run the chatbot:

### Option 1: Streamlit Web Application (Recommended)

The Streamlit app provides a beautiful web interface with chat history, sidebar information, and real-time conversation.

#### Step 1: Activate Virtual Environment (if not already activated)

```powershell
.\venv\Scripts\Activate.ps1
```

#### Step 2: Run the Streamlit Application

```bash
streamlit run app.py
```

#### Step 3: Access the Application

- The application will automatically open in your default web browser
- If it doesn't, look for a URL in the terminal (usually `http://localhost:8501`)
- Copy and paste this URL into your browser

#### Step 4: Interact with the Chatbot

- Type your message in the chat input box at the bottom
- Press Enter or click send
- The chatbot will respond based on the hybrid AI system
- Conversation history is maintained in the chat window

#### Step 5: Stop the Application

- Press `Ctrl + C` in the terminal to stop the Streamlit server

### Option 2: Command-Line Interface

For a simpler terminal-based interaction:

#### Step 1: Activate Virtual Environment (if not already activated)

```powershell
.\venv\Scripts\Activate.ps1
```

#### Step 2: Run the CLI Chatbot

```bash
python chat_bot.py
```

#### Step 3: Start Chatting

- The chatbot will display a welcome message and limitations
- Type your message and press Enter
- To exit, type `quit`, `exit`, or `bye`

## Project Structure

```
Task3/
│
├── app.py                 # Streamlit web application
├── chat_bot.py            # Command-line chatbot interface
├── README.md              # This file
│
└── venv/                  # Virtual environment (not included in version control)
    ├── Scripts/           # Windows executable scripts
    ├── Lib/               # Installed packages
    └── pyvenv.cfg         # Virtual environment configuration
```

## How It Works

### Architecture

1. **Rule-Based Routing (spaCy)**: 
   - Analyzes user input for specific intents (time queries, "about" questions, etc.)
   - Recognizes named entities (persons, locations)
   - Blocks certain types of queries (factual queries, movie reviews, etc.)

2. **Generative Fallback (Hugging Face BlenderBot)**:
   - If no rule matches, the input is passed to the BlenderBot model
   - Generates contextual, conversational responses
   - Maintains conversation history

### Conversation Flow

```
User Input
    ↓
spaCy Analysis (Intent & Entity Detection)
    ↓
Rule Matching?
    ├─ Yes → Return Rule-Based Response
    └─ No → Pass to BlenderBot → Return Generated Response
```

### Memory Management

- The Streamlit app (`app.py`) automatically resets conversation history after 3 exchanges to prevent memory overflow
- The CLI version (`chat_bot.py`) maintains conversation history throughout the session

## Technologies Used

- **Streamlit**: Web application framework for creating the user interface
- **spaCy**: Advanced natural language processing library for intent detection and entity recognition
- **Hugging Face Transformers**: Pre-trained conversational AI model (BlenderBot-400M-distill)
- **PyTorch**: Deep learning framework (backend for transformers)
- **Python**: Programming language

## Known Limitations

The chatbot has the following limitations (as programmed):

1. **No Factual Knowledge**: It's a conversational bot, not a search engine. It cannot provide facts, news, movie reviews, or real-time information.

2. **Frozen Knowledge**: The model's knowledge is based on its training data (around 2020), so it may not know about recent events.

3. **Time Limitations**: Can only provide local system time, not timezone-specific queries.

4. **Person Information**: Cannot provide specific information about people or celebrities.

5. **Conversation Memory**: Limited short-term memory; long conversations may lose context.

6. **Single Query Processing**: Works best with one question at a time.

## Example Queries

Here are some example queries you can try:

### Working Queries:
- "Hello, how are you?"
- "What time is it?"
- "Who are you?"
- "Tell me about yourself"
- "What's the weather like?" (conversational, not factual)
- "How do you work?"

### Blocked/Redirected Queries:
- "What's the plot of Inception?" → Redirected response
- "Who is Barack Obama?" → Blocked (person entity detected)
- "What time is it in London?" → Cannot provide timezone-specific time
- "Give me a movie review" → Blocked



