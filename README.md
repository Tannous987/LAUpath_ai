# 🎓 LAUpath AI

A specialized LLM agent designed to help undergraduate students navigate Lebanese American University (LAU). This project uses LangChain, Gemini Pro, and Streamlit to provide personalized guidance on admissions, programs, majors, and academic planning.

## 📋 Project Overview

LAUpath AI is an intelligent assistant that helps newly graduated high school students:
- Understand LAU's programs, requirements, and opportunities
- Analyze their academic profiles and determine eligibility
- Get personalized major recommendations based on interests and strengths
- Access comprehensive information about admissions, financial aid, and scholarships
- Make informed decisions about their academic journey

## ✨ Features

### Core Functionality
1. **RAG-Based Information Retrieval**: Vectorized search through LAU documents (admissions, programs, fees, etc.)
2. **Student Profile Analyzer**: Analyzes academic records (GPA, SAT, Lebanese exams) and determines:
   - Eligibility status
   - Remedial English requirements
   - Admission probability
   - Academic strengths and recommendations
3. **Major Recommendation Engine**: Provides personalized major suggestions based on:
   - Student interests and passions
   - Academic strengths
   - Career goals
   - Work environment preferences

### User Interface
- Clean, intuitive Streamlit interface
- Student profile management sidebar
- Real-time chat with the AI agent
- Tool usage visualization
- Quick action buttons

## 🛠️ Technology Stack

- **LLM**: Google Gemini Pro (gemini-2.5-flash)
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Google Generative AI Embeddings
- **Web Interface**: Streamlit
- **PDF Processing**: PyPDF

## 📁 Project Structure

```
LAUpath_ai/
├── app.py                 # Main Streamlit application
├── tools.py               # Custom tools (profile analyzer, major recommender)
├── setup_rag.py           # RAG system setup script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── data/
│   └── pdfs/             # LAU documents (PDFs)
│       ├── English Proficiency Scores.pdf
│       ├── Financial Aid.pdf
│       ├── Minors.pdf
│       ├── Scholarships.pdf
│       ├── Undergraduate Freshman Applicants.pdf
│       ├── Undergraduate Programs and Tuition Fees.pdf
│       └── Undergraduate Sophomore Applicants.pdf
└── vector_db/            # Vector database (created after setup)
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone or Download the Project
```bash
cd LAUpath_ai
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   ```

2. Open `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### Step 5: Set Up RAG System
Run the setup script to process PDFs and create the vector database:
```bash
python setup_rag.py
```

This will:
- Load all PDF files from `data/pdfs/`
- Split them into chunks
- Create embeddings
- Store them in `vector_db/` directory

**Note**: This step may take a few minutes depending on the number and size of PDF files.

### Step 6: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📝 File Path Instructions

### Vector Database Path
The vector database is stored in `./vector_db/` by default. To change this:
1. Edit `setup_rag.py`: Change `VECTOR_DB_DIRECTORY` variable (line 15)
2. Edit `app.py`: Change `VECTOR_DB_DIRECTORY` variable (line 30)

### PDF Documents Path
PDF documents are loaded from `./data/pdfs/` by default. To change this:
1. Edit `setup_rag.py`: Change `PDF_DIRECTORY` variable (line 14)
2. Ensure all LAU-related PDFs are in the specified directory

### Example Path Modifications
If you want to use absolute paths:
```python
# In setup_rag.py
PDF_DIRECTORY = "C:/Users/YourName/Documents/LAU_docs"
VECTOR_DB_DIRECTORY = "C:/Users/YourName/Documents/LAU_vector_db"
```

## 🎯 Usage Guide

### 1. Setting Up Your Profile
- Click on the sidebar "Enter Your Academic Information"
- Fill in your:
  - High School GPA
  - Lebanese Baccalaureate Score (if available)
  - SAT Score (if available)
  - English Proficiency Test (TOEFL, IELTS, or Duolingo) and score
- Click "Save Profile"

### 2. Analyzing Your Profile
- Click "Analyze My Profile" in the sidebar, or
- Type in chat: "Analyze my student profile" or "What's my eligibility?"

### 3. Getting Major Recommendations
- Click "Get Major Recommendations" in the sidebar, or
- Type in chat: "Help me choose a major" or "What major should I study?"

### 4. Asking Questions About LAU
Simply chat with the AI about:
- "What are the admission requirements?"
- "Tell me about Computer Science program"
- "What scholarships are available?"
- "What are the tuition fees?"
- Any other LAU-related questions!

## 🔧 Custom Tools Documentation

### Tool 1: Student Profile Analyzer (`analyze_student_profile`)
**Purpose**: Analyzes student academic records and provides comprehensive feedback.

**Parameters**:
- `school_gpa` (float): High school GPA
- `lebanese_exam_score` (Optional[float]): Lebanese Baccalaureate score
- `sat_score` (Optional[int]): SAT total score
- `english_proficiency_score` (Optional[float]): English test score
- `english_proficiency_type` (Optional[str]): Type of test (TOEFL, IELTS, Duolingo)

**Returns**: Detailed analysis including eligibility, remedial requirements, strengths, and recommendations.

### Tool 2: Major Recommendation Engine (`recommend_major`)
**Purpose**: Provides personalized major recommendations based on student profile.

**Parameters**:
- `interests` (str): Student's interests and hobbies
- `academic_strengths` (str): Subjects where student excels
- `career_goals` (str): Desired career path
- `preferred_work_environment` (str): Preferred work setting
- `student_profile_data` (Optional[str]): JSON string of profile data

**Returns**: Top 3 major recommendations with fit scores and explanations.

### Tool 3: Vector Database Search (`search_vector_db`)
**Purpose**: Searches LAU documents for relevant information.

**Parameters**:
- `query` (str): Search query string

**Returns**: Top 5 most relevant document chunks with source information.

## 🎨 Prompting Strategy

### Style
- **Structured XML-style prompts**: Using `<identity>`, `<purpose>`, and `<guidelines>` tags for clear organization
- **Role-based prompting**: Defining the agent as "LAUpath AI" with specific expertise
- **Tool-aware prompting**: Explicitly instructing when to use each tool

### Intended Outcomes
1. **Consistency**: The agent maintains its role and purpose throughout conversations
2. **Tool Usage**: The agent knows when to use tools vs. when to answer directly
3. **Helpfulness**: The agent is supportive and encouraging to students
4. **Accuracy**: The agent uses RAG to provide accurate, document-based information

### Development Process
The prompting strategy evolved through trial and error:
- **Initial**: Simple system message → Agent didn't use tools effectively
- **Iteration 1**: Added explicit tool usage instructions → Better but still inconsistent
- **Iteration 2**: Structured XML format with clear sections → Improved consistency
- **Final**: Combined structured prompts with tool descriptions → Optimal tool usage and helpful responses

## 🏗️ System Design

```
┌─────────────────┐
│   Streamlit UI  │
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAUpathAgent   │
│   (LangChain)   │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┬──────────────┐
    │         │              │              │
    ▼         ▼              ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  RAG   │ │ Profile  │ │  Major   │ │  Gemini  │
│ Search │ │ Analyzer │ │Recommender│ │   LLM    │
└────────┘ └──────────┘ └──────────┘ └──────────┘
    │
    ▼
┌─────────┐
│ ChromaDB│
│ Vector  │
│Database │
└─────────┘
    │
    ▼
┌─────────┐
│PDF Docs │
│(data/)  │
└─────────┘
```

## 🔐 API Information

### Paid APIs
- **Google Gemini API**: Used for LLM and embeddings
  - **Purpose**: Language model inference and text embeddings
  - **Cost**: Free tier available with usage limits. Check [Google AI Studio](https://makersuite.google.com/app/apikey) for current pricing
  - **Note**: The free tier should be sufficient for development and testing

### API Key Security
- **Never commit API keys to version control**
- Use `.env` file (already in `.gitignore`)
- Use `[insert API key here]` as placeholder in code/documentation

## 🧪 Testing

To test the system:
1. Run `setup_rag.py` to ensure vector database is created
2. Start the app with `streamlit run app.py`
3. Test each tool:
   - Enter profile information and ask for analysis
   - Ask about LAU programs and requirements
   - Request major recommendations
4. Verify tool calls are displayed correctly
5. Check that responses are helpful and accurate

## 📊 Additional Features

### Error Handling
- Graceful error handling for API failures
- User-friendly error messages
- Validation of user inputs

### Type Hints and Docstrings
- All functions include type hints
- Comprehensive docstrings following Google style
- Clear parameter and return type documentation

### Conversation History
- Maintains conversation context across interactions
- Stores messages in Streamlit session state
- Preserves tool call history

## 🤝 Team Contribution

[To be filled by team members]

## 📄 License

This project is created for educational purposes as part of COE548 Final Project.

## 🙏 Acknowledgments

- Lebanese American University for providing the documents
- LangChain team for the excellent framework
- Google for Gemini API

## 📞 Support

For issues or questions:
1. Check the documentation above
2. Verify your API key is set correctly
3. Ensure the vector database is set up (`python setup_rag.py`)
4. Check that all dependencies are installed

---

**Note**: This project follows best practices for AI development including ethical considerations, proper error handling, and comprehensive documentation.

