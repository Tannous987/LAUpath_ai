# System Design Documentation

## Architecture Overview

LAUpath AI follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│                    (Streamlit - app.py)                      │
│  - Chat interface                                            │
│  - Student profile management                                │
│  - Tool visualization                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                               │
│              (LAUpathAgent - LangChain)                     │
│  - Conversation management                                   │
│  - Tool orchestration                                        │
│  - Context handling                                          │
└───────┬─────────────────────────────────────────────────────┘
        │
        ├──────────────────┬──────────────────┬──────────────┐
        │                  │                  │              │
        ▼                  ▼                  ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RAG Tool    │  │ Profile      │  │ Major        │  │  Gemini LLM  │
│              │  │ Analyzer     │  │ Recommender  │  │              │
│ search_      │  │ analyze_     │  │ recommend_   │  │  (Core       │
│ vector_db    │  │ student_     │  │ major        │  │   Inference) │
│              │  │ profile      │  │              │  │              │
└──────┬───────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Vector Database Layer                          │
│                    (ChromaDB)                               │
│  - Document embeddings                                       │
│  - Similarity search                                         │
│  - Persistent storage                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Layer                                      │
│              (PDF Documents)                                │
│  - LAU admission documents                                   │
│  - Program information                                       │
│  - Financial aid information                                 │
└─────────────────────────────────────────────────────────────┘
```

## Component Interactions

### 1. User Request Flow

```
User Input
    │
    ▼
Streamlit Interface (app.py)
    │
    ▼
LAUpathAgent.send_message()
    │
    ▼
LangChain Agent (with tools)
    │
    ├─→ Tool Selection (if needed)
    │   │
    │   ├─→ search_vector_db → ChromaDB → PDF Documents
    │   ├─→ analyze_student_profile → Profile Analysis Logic
    │   └─→ recommend_major → Recommendation Algorithm
    │
    ▼
Gemini LLM (generates response)
    │
    ▼
Response to User
```

### 2. RAG System Flow

```
PDF Documents (data/pdfs/)
    │
    ▼
setup_rag.py
    │
    ├─→ PyPDFLoader (load PDFs)
    ├─→ TextSplitter (chunk documents)
    ├─→ GoogleGenerativeAIEmbeddings (create embeddings)
    └─→ ChromaDB (store vectors)
    │
    ▼
Vector Database (vector_db/)
    │
    ▼
search_vector_db tool
    │
    ├─→ Query embedding
    ├─→ Similarity search (cosine)
    └─→ Top 5 relevant chunks
    │
    ▼
Context for LLM
```

### 3. Student Profile Analysis Flow

```
User Input (Profile Data)
    │
    ▼
Streamlit Sidebar Form
    │
    ▼
Session State (st.session_state.student_profile)
    │
    ▼
User Request: "Analyze my profile"
    │
    ▼
Agent invokes analyze_student_profile tool
    │
    ├─→ GPA Analysis
    ├─→ Lebanese Exam Analysis
    ├─→ SAT Score Analysis
    ├─→ English Proficiency Check
    └─→ Eligibility Determination
    │
    ▼
Structured Analysis Result
    │
    ▼
LLM formats response
    │
    ▼
User receives analysis
```

### 4. Major Recommendation Flow

```
User Request: "Help me choose a major"
    │
    ▼
Agent asks for information (interactive)
    │
    ├─→ Interests
    ├─→ Academic Strengths
    ├─→ Career Goals
    └─→ Work Environment Preference
    │
    ▼
Agent invokes recommend_major tool
    │
    ├─→ Match interests with majors
    ├─→ Match strengths with majors
    ├─→ Match goals with majors
    ├─→ Score each major
    └─→ Rank top recommendations
    │
    ▼
Top 3 Recommendations with Explanations
    │
    ▼
LLM formats personalized response
    │
    ▼
User receives recommendations
```

## Data Flow

### Input Data
1. **User Messages**: Text input through Streamlit chat
2. **Student Profile**: Form data from sidebar
3. **PDF Documents**: Static files in `data/pdfs/`

### Processing
1. **Message Processing**: LangChain agent processes user input
2. **Tool Selection**: Agent decides which tool(s) to use
3. **Tool Execution**: Tools process data and return results
4. **LLM Inference**: Gemini generates response based on context

### Output Data
1. **Chat Messages**: Displayed in Streamlit interface
2. **Tool Results**: Shown with status indicators
3. **Analysis Reports**: Formatted text responses

## State Management

### Session State (Streamlit)
- `st.session_state.agent`: LAUpathAgent instance
- `st.session_state.messages`: Conversation history
- `st.session_state.student_profile`: Student academic data

### Agent State (LangChain)
- `self.messages`: Full conversation history with system message
- Tool call history embedded in messages

## Error Handling

1. **API Errors**: Caught in `send_message()` method
2. **Vector DB Errors**: Handled in `search_vector_db` tool
3. **Profile Analysis Errors**: Try-except in `analyze_student_profile`
4. **UI Errors**: Streamlit error messages for user feedback

## Security Considerations

1. **API Keys**: Stored in `.env` file (not committed)
2. **User Data**: Stored only in session state (not persisted)
3. **Input Validation**: Type checking in tools
4. **Error Messages**: Sanitized to avoid exposing internals

## Scalability Considerations

1. **Vector Database**: Can handle thousands of documents
2. **Chunking Strategy**: Optimized for retrieval (1000 chars, 200 overlap)
3. **Tool Execution**: Synchronous but can be made async if needed
4. **Session Management**: Each user has isolated session state

## Future Enhancements

1. **Database Integration**: Persist student profiles
2. **Multi-language Support**: Arabic language support
3. **Advanced Analytics**: Track recommendation accuracy
4. **Export Functionality**: Generate PDF reports
5. **Admin Dashboard**: Monitor usage and performance

