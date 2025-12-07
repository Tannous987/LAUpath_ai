## Installation and Setup Instructions

This section provides comprehensive step-by-step instructions for installing and setting up the LAUpath AI project environment. Following these instructions will ensure that all dependencies are correctly installed, the system is properly configured, and the application is ready for use.

### 1 Prerequisites

Before beginning the installation process, ensure that the following prerequisites are met:

1. **Python Installation**: Python 3.8 or higher must be installed on the system. To verify the Python version, run:
   ```bash
   python --version
   ```
   or
   ```bash
   python3 --version
   ```

2. **Google Gemini API Key**: A valid API key for Google's Gemini API is required. The API key can be obtained free of charge from [Google AI Studio](https://makersuite.google.com/app/apikey). The free tier provides sufficient quota for development and testing purposes.

3. **System Requirements**: 
   - Minimum 4GB RAM (8GB recommended)
   - At least 500MB free disk space for dependencies and vector database
   - Internet connection for downloading dependencies and API access

### 2 Step-by-Step Installation Process

#### Step 1: Obtain Project Files

Navigate to the project directory. If the project was downloaded as a ZIP file, extract it to a desired location. If using version control, clone the repository:

```bash
cd LAUpath_ai
```

#### Step 2: Create Virtual Environment (Highly Recommended)

Creating a virtual environment isolates project dependencies from the system Python installation, preventing conflicts with other projects. This is considered a best practice in Python development.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, the terminal prompt should display `(venv)`, indicating that the virtual environment is active.

#### Step 3: Install Python Dependencies

Install all required Python packages using pip and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This command will install the following key dependencies:
- **LangChain**: Framework for building LLM applications
- **Streamlit**: Web framework for the user interface
- **ChromaDB**: Vector database for RAG functionality
- **Google Generative AI**: SDK for Gemini API access
- **PyPDF**: PDF processing library
- **python-dotenv**: Environment variable management

The installation process may take several minutes depending on internet speed. Ensure that all packages install without errors.

#### Step 4: Configure Environment Variables

The application requires a Google Gemini API key to function. This key must be stored securely in an environment file.

1. **Create `.env` file**: In the project root directory, create a file named `.env`. This can be done by copying the example file:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```

2. **Add API Key**: Open the `.env` file in a text editor and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
   Replace `your_actual_api_key_here` with the actual API key obtained from Google AI Studio.

3. **Verify File Location**: Ensure the `.env` file is in the project root directory (same level as `app.py` and `requirements.txt`).

**Security Note**: The `.env` file is automatically excluded from version control via `.gitignore` to prevent accidental exposure of API keys. Never commit API keys to version control systems.

#### Step 5: Initialize the RAG System

The Retrieval Augmented Generation system requires a vector database to be created from the PDF documents. This is a one-time setup process that must be completed before running the application.

Run the setup script:
```bash
python setup_rag.py
```

**Verification**: After completion, verify that the `vector_db/` directory was created and contains database files.

#### Step 6: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will:
1. Initialize the LAUpath AI agent
2. Load the vector database
3. Start a local web server
4. Automatically open the application in the default web browser
