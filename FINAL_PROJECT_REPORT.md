# COE548 Final Project Report: LAUpath AI

**Course**: COE548 - Large Language Models  
**Project**: Building a Specialized LLM Agent  
**Institution**: Lebanese American University (LAU)

---

## 1. Project Description: Detailed Explanation of What Your Project Does

### 1.1 Introduction and Problem Statement

LAUpath AI is a specialized Large Language Model (LLM) agent designed to assist newly graduated high school students in navigating the complex landscape of Lebanese American University (LAU). The transition from high school to university is a critical period in a student's academic journey, often characterized by uncertainty regarding program selection, admission requirements, eligibility assessment, and major choice. Traditional methods of obtaining this information—such as browsing through multiple PDF documents, consulting with advisors, or searching through fragmented online resources—can be time-consuming, overwhelming, and sometimes incomplete.

This project addresses these challenges by developing an intelligent, conversational AI assistant that leverages state-of-the-art LLM technology combined with Retrieval Augmented Generation (RAG) and custom analytical tools to provide personalized, accurate, and comprehensive guidance to prospective students. The system serves as a one-stop solution for students seeking information about LAU's programs, admission requirements, financial aid opportunities, and personalized academic recommendations.

### 1.2 Core Functionality and Features

LAUpath AI provides four primary functionalities, each implemented as a specialized tool integrated into the LLM agent:

#### 1.2.1 Retrieval Augmented Generation (RAG) System

The foundation of the system is a vectorized RAG mechanism that enables the agent to retrieve and synthesize information from a comprehensive collection of LAU's official documents. The RAG system processes seven key PDF documents covering:

- **Admission Requirements**: Detailed information for freshman and sophomore applicants
- **Academic Programs**: Complete listing of undergraduate programs and their descriptions
- **Tuition and Fees**: Comprehensive fee structures for all programs
- **Financial Aid and Scholarships**: Available financial assistance programs
- **English Proficiency Requirements**: Standardized test score requirements and remedial course policies
- **Minor Programs**: Information about available minor specializations

The RAG implementation uses ChromaDB as the vector database, Google Generative AI embeddings (`models/gemini-embedding-exp-03-07`) for semantic representation, and cosine similarity search to retrieve the top 5 most relevant document chunks for any given query. This ensures that the agent's responses are grounded in official LAU documentation, providing accurate and up-to-date information rather than relying solely on the LLM's training data.

#### 1.2.2 Student Profile Analyzer

The Student Profile Analyzer is a custom analytical tool that evaluates a student's academic credentials to provide comprehensive eligibility assessment and personalized recommendations. The tool accepts multiple input parameters:

- **High School GPA**: Evaluated on both Lebanese (0-20 scale) and US (0-4.0 scale) grading systems
- **Lebanese Baccalaureate Score**: Official Lebanese national exam results
- **SAT Score**: Standardized test performance (out of 1600)
- **English Proficiency Scores**: Supports TOEFL, IELTS, and Duolingo test results

The analyzer performs multi-dimensional evaluation to determine:

1. **Eligibility Status**: Categorizes students as having "high", "good", "acceptable", or "low" eligibility based on their academic performance
2. **Admission Probability**: Provides probabilistic assessment (high, medium-high, medium, or low) of admission likelihood
3. **Remedial English Requirements**: Automatically determines whether a student needs to enroll in remedial English courses based on LAU's official policies and the student's English proficiency scores
4. **Academic Strengths Identification**: Highlights areas where the student demonstrates strong performance
5. **Personalized Recommendations**: Suggests specific actions the student can take to improve their admission prospects

The tool implements rule-based logic derived from LAU's official admission policies, ensuring consistency and accuracy in evaluations. For instance, it checks English proficiency scores against specific thresholds (e.g., TOEFL ≥ 80, IELTS ≥ 6.5, Duolingo ≥ 105) to determine remedial course requirements.

#### 1.2.3 Major Recommendation Engine

The Major Recommendation Engine addresses one of the most challenging decisions students face: choosing an appropriate major. This tool employs a sophisticated matching algorithm that considers multiple student attributes:

- **Interests and Passions**: Student's hobbies, activities, and areas of curiosity
- **Academic Strengths**: Subjects and domains where the student excels
- **Career Goals**: Desired professional trajectory and long-term aspirations
- **Work Environment Preferences**: Preferred work settings (e.g., office, field, laboratory, remote)

The recommendation system maintains a comprehensive database of all LAU majors, dynamically extracted from the official "Undergraduate Programs and Tuition Fees" document using RAG. This ensures that recommendations are always based on currently available programs. The database includes detailed attributes for each major, such as:

- Field of study (e.g., Engineering, Business, Arts, Sciences)
- Required skills and competencies
- Typical career paths
- Work environment characteristics
- Academic focus areas

The tool employs a scoring algorithm that matches student profiles against major attributes, considering weighted factors for interests, strengths, career alignment, and work preferences. It returns the top three best-fit majors with detailed explanations of why each major is recommended, including specific alignment points between the student's profile and the major's characteristics.

Additionally, the system includes an interactive Career Test feature—a modal-based questionnaire inspired by the O*NET Interest Profiler. Students answer questions using emoji-based responses, and the test results are automatically fed into the recommendation engine to generate personalized major suggestions.

#### 1.2.4 Course Map Retriever

The Course Map Retriever tool provides students with direct access to curriculum information for specific majors. When a student inquires about a major's course plan, curriculum, or course map, the tool:

1. Identifies the major from the query (supporting both full names and common abbreviations)
2. Maps the major to the corresponding course map PDF file
3. Retrieves and displays the PDF document within the chat interface
4. Provides a download option for the student's convenience

The tool supports all engineering majors (Computer Engineering, Electrical Engineering, Mechanical Engineering, Civil Engineering, Industrial Engineering, Mechatronics Engineering, Petroleum Engineering) and handles various naming conventions and abbreviations (e.g., "COE" for Computer Engineering, "CIE" for Civil Engineering). If a course map is not available for a requested major, the tool gracefully informs the student without exposing technical implementation details.

### 1.3 Technical Architecture and Implementation

The system is built using a modular architecture that separates concerns across multiple layers:

#### 1.3.1 LLM Agent Framework

The core agent is implemented using LangChain, a powerful framework for building LLM applications. The agent uses Google's Gemini Pro model (`gemini-2.5-flash`) as its underlying language model, configured with a temperature of 0.3 to balance creativity with consistency. The agent is created using LangChain's `create_agent` function, which automatically handles tool selection, execution, and response generation.

The agent maintains conversation history through a message-based system that preserves context across interactions. Each conversation session is isolated, allowing multiple students to use the system simultaneously without context interference. The conversation history includes system messages, human messages, AI responses, and tool execution results, enabling the agent to maintain coherent, context-aware conversations.

#### 1.3.2 Vector Database and RAG Pipeline

The RAG system is initialized through a dedicated setup script (`setup_rag.py`) that:

1. **Document Loading**: Uses PyPDFLoader to extract text from all PDF documents in the `data/lau_documents/` directory
2. **Text Chunking**: Splits documents into manageable chunks (1000 characters with 200-character overlap) to optimize retrieval while maintaining context
3. **Embedding Generation**: Creates vector embeddings using Google Generative AI's embedding model
4. **Vector Storage**: Persists embeddings in ChromaDB with cosine similarity indexing for efficient retrieval

At runtime, when the `search_vector_db` tool is invoked, the system:
1. Generates an embedding vector for the user's query
2. Performs cosine similarity search against the vector database
3. Retrieves the top 5 most relevant document chunks
4. Returns the chunks with source attribution for transparency

#### 1.3.3 User Interface

The user interface is built using Streamlit, providing a clean, intuitive web-based chat interface that mimics modern conversational AI applications like ChatGPT. Key UI features include:

- **Chat Interface**: Real-time conversational interface with message history display
- **Chat Management**: Sidebar with chat history, allowing users to create new chats, switch between conversations, and delete previous chats with confirmation
- **Academic Profile Modal**: Pop-up window for entering and managing student academic information (GPA, test scores, etc.)
- **Career Test Modal**: Interactive questionnaire with emoji-based responses, navigation controls (previous/next), and automatic submission to the recommendation engine
- **Tool Visualization**: Real-time display of tool invocations with status indicators
- **PDF Display**: Inline rendering of course map PDFs with download functionality
- **Responsive Design**: Professional styling with custom CSS for modal windows, buttons, and layout

The interface is designed with user experience in mind, featuring clear visual hierarchy, intuitive navigation, and immediate feedback for all user actions.

#### 1.3.4 Persistent Chat History Management

A critical feature of LAUpath AI is its comprehensive chat history management system, which enables users to maintain multiple conversation threads and seamlessly navigate between them. This feature addresses the need for students to have separate conversations about different topics (e.g., one chat about admission requirements, another about major selection) while preserving context within each conversation.

**Chat Persistence Architecture**:
- **Storage Mechanism**: Each chat session is persisted as a JSON file in the `chat_history/` directory, ensuring conversations survive application restarts
- **Chat Metadata**: Each chat file contains:
  - Unique chat identifier (UUID)
  - Auto-generated title (derived from the first user message, truncated to 40 characters)
  - Creation timestamp (used internally for sorting chats in reverse chronological order)
  - Last update timestamp (tracks modifications; used as fallback for sorting if creation timestamp is missing)
  - Complete message history (user and assistant messages)

**Chat Management Features**:

1. **Chat Creation**: Users can create new chat sessions at any time through a dedicated "New Chat" button. New chats are automatically assigned a unique identifier and initialized with a fresh conversation context. The agent's internal message history is reset to include only the system message, ensuring no context leakage from previous conversations.

2. **Chat History Sidebar**: The sidebar displays all saved chat sessions in reverse chronological order (newest first), making it easy for users to access recent conversations. Each chat entry shows:
   - Auto-generated title based on the first message
   - Visual indication of the currently active chat
   - Delete button for removing unwanted chats

3. **Chat Switching**: Users can seamlessly switch between different chat sessions by clicking on any chat in the sidebar. When switching:
   - The current chat is automatically saved to disk
   - The selected chat's message history is loaded from persistent storage
   - The agent's internal conversation context is synchronized with the loaded messages
   - The UI updates to display the selected chat's messages

4. **Chat Deletion**: Users can delete individual chat sessions through a delete button associated with each chat entry. The deletion process includes:
   - **Confirmation Dialog**: A confirmation prompt prevents accidental deletions, asking the user to confirm before permanently removing a chat
   - **File Removal**: The chat's JSON file is deleted from the `chat_history/` directory
   - **Index Update**: The chat index in session state is updated to reflect the deletion
   - **Automatic Navigation**: If the deleted chat was the currently active chat, the system automatically switches to the most recent remaining chat, or creates a new chat if no chats remain

5. **Context Isolation**: Each chat maintains its own isolated conversation context. This is achieved through:
   - **Agent Message Synchronization**: When switching chats, the agent's internal `messages` list is synchronized with the loaded chat's history using the `load_messages()` method
   - **Session State Management**: Streamlit's session state maintains separate message lists for each chat
   - **System Message Preservation**: Each chat starts with the system message, ensuring consistent agent behavior across all conversations

6. **Automatic Saving**: Chat sessions are automatically saved to disk:
   - After each user message and agent response
   - When switching between chats
   - When creating a new chat
   - Before deleting a chat

This persistent chat history system ensures that students can:
- Maintain separate conversations for different topics without context mixing
- Resume previous conversations at any time
- Organize their interactions with the AI assistant
- Safely delete conversations they no longer need
- Access their conversation history across multiple sessions

The implementation demonstrates proper state management practices, combining Streamlit's session state for in-memory data with file-based persistence for long-term storage, creating a robust and user-friendly chat management system.

### 1.4 System Workflow and User Interaction

The typical user interaction flow follows these patterns:

1. **Initial Setup**: User launches the application and optionally creates an academic profile through the modal interface. The system automatically creates a new chat session or loads the most recent chat if one exists.

2. **Information Queries**: User asks questions about LAU (e.g., "What are the admission requirements?"), triggering the RAG tool to search official documents. The conversation is automatically saved after each exchange.

3. **Profile Analysis**: User requests profile analysis, either through the UI button or natural language, activating the profile analyzer tool. The analysis results are integrated into the conversation history.

4. **Major Exploration**: User takes the career test or directly requests major recommendations, engaging the recommendation engine. Test results and recommendations are preserved in the chat history.

5. **Curriculum Inquiry**: User asks about specific major curricula, invoking the course map retriever. The retrieved PDFs are displayed inline within the conversation.

6. **Follow-up Questions**: User continues the conversation with context-aware follow-up questions, leveraging the maintained conversation history within the current chat session.

7. **Chat Management**: Users can create new chats for different topics, switch between existing chats to resume previous conversations, or delete chats they no longer need. Each chat maintains its own isolated context, preventing information leakage between conversations.

Throughout this process, the agent intelligently selects appropriate tools based on user intent, executes them, and synthesizes the results into natural, helpful responses. The persistent chat history ensures that all interactions are preserved and can be revisited at any time, while the chat management features allow users to organize their conversations effectively.

### 1.5 Value Proposition and Impact

LAUpath AI provides significant value to prospective LAU students by:

1. **Democratizing Information Access**: Making comprehensive LAU information easily accessible through natural language queries, eliminating the need to navigate multiple documents
2. **Personalized Guidance**: Providing tailored recommendations based on individual student profiles rather than generic advice
3. **Time Efficiency**: Reducing the time students spend searching for information and making decisions
4. **Consistency**: Ensuring all students receive accurate, consistent information based on official LAU policies
5. **Accessibility**: Available 24/7 without requiring appointments with academic advisors
6. **Transparency**: Providing clear explanations for recommendations and eligibility assessments

The system serves as a practical demonstration of how LLM technology can be applied to solve real-world problems in educational contexts, combining the power of large language models with domain-specific tools and knowledge bases to create a specialized, useful application.

### 1.6 Technical Highlights and Best Practices

The implementation follows software engineering best practices:

- **Type Hints**: All functions include comprehensive type annotations for better code maintainability and IDE support
- **Docstrings**: Detailed documentation following Google-style docstrings for all functions and classes
- **Error Handling**: Robust try-except blocks with user-friendly error messages throughout the codebase
- **Modular Design**: Clear separation of concerns with dedicated modules for tools, agent logic, and UI
- **Configuration Management**: Environment variables for API keys and configuration parameters
- **Security**: API keys stored securely in `.env` files, never committed to version control
- **Conversation Management**: Proper isolation of chat sessions with persistent history storage
- **State Management**: Sophisticated chat history management with file-based persistence, automatic saving, and context synchronization between agent and UI
- **User Experience**: Intuitive chat management interface with confirmation dialogs for destructive actions and automatic navigation handling

This project demonstrates proficiency in modern LLM application development, combining theoretical understanding of language models with practical software engineering skills to create a production-ready system.

---

## 2. System Design: Component Interactions

This section presents a comprehensive system design diagram illustrating the architecture of LAUpath AI and the interactions between its various components. The system follows a layered architecture pattern with clear separation of concerns, enabling modularity, maintainability, and scalability.

### 2.1 Overall System Architecture

The system is organized into four primary layers: User Interface Layer, Agent Layer, Tool Layer, and Data Layer. The following diagram illustrates the high-level architecture and component relationships:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                                │
│                         (Streamlit - app.py)                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Chat Interface   │  │  Profile Modal    │  │  Career Test      │       │
│  │  - Message Display│  │  - GPA Input      │  │  - Questionnaire  │       │
│  │  - Input Field    │  │  - Test Scores    │  │  - Emoji Answers  │       │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘       │
│           │                       │                       │                  │
│           └───────────────────────┴───────────────────────┘                  │
│                                   │                                           │
│  ┌────────────────────────────────┴────────────────────────────────┐         │
│  │              Chat History Management                            │         │
│  │  - Chat Creation/Deletion                                       │         │
│  │  - Chat Switching                                               │         │
│  │  - Persistent Storage (JSON files)                               │         │
│  └─────────────────────────────────────────────────────────────────┘         │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                          │
                                          │ User Messages & Actions
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT LAYER                                      │
│                    (LAUpathAgent - LangChain)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    Conversation Management                           │ │
│  │  - Message History (self.messages)                                   │ │
│  │  - Context Preservation                                              │ │
│  │  - System Message Injection                                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    Tool Orchestration                                │ │
│  │  - Tool Selection Logic                                              │ │
│  │  - Tool Execution Coordination                                        │ │
│  │  - Response Synthesis                                                 │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    LLM Integration                                  │ │
│  │  - Google Gemini Pro (gemini-2.5-flash)                             │ │
│  │  - Temperature: 0.3                                                  │ │
│  │  - Response Generation                                               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┬───────────────┐
                │               │               │               │
                ▼               ▼               ▼               ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   RAG Tool       │  │  Profile         │  │  Major            │  │  Course Map       │
│                  │  │  Analyzer        │  │  Recommender      │  │  Retriever        │
│ search_vector_db │  │ analyze_student_ │  │ recommend_major   │  │ get_course_map   │
│                  │  │ profile          │  │                  │  │                  │
└────────┬─────────┘  └──────────────────┘  └──────────────────┘  └────────┬─────────┘
         │                                                                     │
         ▼                                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VECTOR DATABASE LAYER                                  │
│                            (ChromaDB)                                      │
│  - Document Embeddings                                                      │
│  - Cosine Similarity Search                                                 │
│  - Persistent Storage (vector_db/)                                           │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                │
│  │  LAU Documents (PDFs)    │  │  Course Map PDFs          │                │
│  │  data/lau_documents/      │  │  data/CourseMaps/        │                │
│  │  - Admission Requirements │  │  - COE_courseMap.pdf      │                │
│  │  - Program Information   │  │  - ELE_courseMap.pdf      │                │
│  │  - Financial Aid         │  │  - MEE_courseMap.pdf      │                │
│  │  - Scholarships          │  │  - CIE_courseMap.pdf      │                │
│  │  - English Proficiency    │  │  - INE_courseMap.pdf      │                │
│  │  - Minors                │  │  - MCE_courseMap.pdf      │                │
│  │                          │  │  - PTE_courseMap.pdf      │                │
│  └──────────────────────────┘  └──────────────────────────┘                │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │              Chat History Storage                                     │ │
│  │              chat_history/                                           │ │
│  │              - {chat_id}.json files                                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 User Request Processing Flow

The following diagram illustrates the complete flow of a user request through the system, from input to response:

```
┌─────────────┐
│   User      │
│  Input      │
└──────┬──────┘
       │
       │ 1. User types message or clicks action button
       ▼
┌─────────────────────────────────────────────────────────────┐
│         Streamlit UI (app.py)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Session State Management                             │   │
│  │  - st.session_state.messages                         │   │
│  │  - st.session_state.current_chat_id                  │   │
│  │  - st.session_state.student_profile                  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 2. Message added to session state
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         LAUpathAgent.send_message()                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Append HumanMessage to self.messages              │   │
│  │  2. Invoke LangChain agent with message               │   │
│  │  3. Agent analyzes intent and selects tool(s)         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Tool 1  │   │ Tool 2  │   │ Tool 3  │
   │ (RAG)   │   │(Profile)│   │(Major)  │
   └────┬────┘   └────┬────┘   └────┬────┘
        │            │            │
        │            │            │
        └────────────┴────────────┘
                     │
                     │ 3. Tool execution results
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Gemini LLM (Response Generation)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - Receives tool results                              │   │
│  │  - Synthesizes information                           │   │
│  │  - Generates natural language response                │   │
│  │  - Returns AIMessage with formatted output           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 4. Response returned
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Response Processing                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Save to session state                            │   │
│  │  2. Save to persistent storage (chat_history/)      │   │
│  │  3. Display in UI                                    │   │
│  │  4. Update chat metadata                            │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 5. User sees response
                        ▼
┌─────────────┐
│   User      │
│  Response  │
└─────────────┘
```

### 2.3 RAG System Flow

The Retrieval Augmented Generation system follows a two-phase process: initialization (setup) and runtime (query). The following diagram details both phases:

```
PHASE 1: INITIALIZATION (setup_rag.py)
┌─────────────────────────────────────────────────────────────┐
│                    PDF Documents                              │
│              data/lau_documents/*.pdf                         │
│  - Undergraduate Freshman Applicants.pdf                      │
│  - Undergraduate Sophomore Applicants.pdf                     │
│  - Undergraduate Programs and Tuition Fees.pdf               │
│  - Financial Aid.pdf                                          │
│  - Scholarships.pdf                                           │
│  - English Proficiency Scores.pdf                             │
│  - Minors.pdf                                                 │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 1. Load PDFs
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    PyPDFLoader                                │
│              Extract text from PDFs                          │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 2. Split into chunks
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    TextSplitter                              │
│  - Chunk size: 1000 characters                               │
│  - Overlap: 200 characters                                   │
│  - Preserves context across chunks                           │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 3. Generate embeddings
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         GoogleGenerativeAIEmbeddings                         │
│              (models/gemini-embedding-exp-03-07)            │
│  - Converts text chunks to vector embeddings                 │
│  - 768-dimensional vectors                                   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 4. Store in vector database
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChromaDB                                  │
│              vector_db/ directory                            │
│  - Collection: "lau_documents"                                │
│  - Index: Cosine similarity                                  │
│  - Persistent storage on disk                               │
└─────────────────────────────────────────────────────────────┘

PHASE 2: RUNTIME QUERY (search_vector_db tool)
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│              "What are admission requirements?"              │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 1. Generate query embedding
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         GoogleGenerativeAIEmbeddings                         │
│              Convert query to vector                         │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 2. Similarity search
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChromaDB                                  │
│  - Cosine similarity search                                  │
│  - Retrieve top 5 most similar chunks                        │
│  - Return chunks with source metadata                        │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 3. Return relevant context
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Result                                │
│  [Source: Undergraduate Freshman Applicants.pdf]             │
│  "Admission requirements include..."                         │
│  ---                                                          │
│  [Source: Undergraduate Programs and Tuition Fees.pdf]        │
│  "Program-specific requirements..."                           │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        │ 4. Context provided to LLM
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gemini LLM                                │
│  - Synthesizes retrieved context                            │
│  - Generates accurate, document-grounded response            │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Chat History Management Flow

The chat history management system ensures persistent storage and context isolation. The following diagram illustrates the complete flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Action                               │
│  - Create new chat                                           │
│  - Switch to existing chat                                   │
│  - Delete chat                                               │
│  - Send message                                              │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Chat Management Functions                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  create_new_chat()                                   │   │
│  │  - Generate UUID                                      │   │
│  │  - Initialize session state                          │   │
│  │  - Reset agent.messages                               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  load_chat_messages(chat_id)                         │   │
│  │  - Read JSON file from chat_history/                 │   │
│  │  - Convert to LangChain messages                     │   │
│  │  - Load into session state                           │   │
│  │  - Synchronize with agent.messages                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  delete_chat(chat_id)                                 │   │
│  │  - Show confirmation dialog                          │   │
│  │  - Delete JSON file                                   │   │
│  │  - Update chat index                                  │   │
│  │  - Auto-switch to another chat                        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  save_current_chat()                                  │   │
│  │  - Serialize messages                                  │   │
│  │  - Generate/update title                              │   │
│  │  - Write to JSON file                                 │   │
│  │  - Update chat index                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Persistent Storage                               │
│              chat_history/ directory                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  {uuid1}.json                                        │   │
│  │  {                                                    │   │
│  │    "id": "uuid1",                                    │   │
│  │    "title": "What are admission requirements?",      │   │
│  │    "created_at": "2024-01-15T10:30:00",              │   │
│  │    "updated_at": "2024-01-15T10:35:00",              │   │
│  │    "messages": [...]                                 │   │
│  │  }                                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  {uuid2}.json                                        │   │
│  │  {                                                    │   │
│  │    "id": "uuid2",                                    │   │
│  │    "title": "Help me choose a major",                │   │
│  │    "created_at": "2024-01-15T11:00:00",              │   │
│  │    "updated_at": "2024-01-15T11:15:00",              │   │
│  │    "messages": [...]                                 │   │
│  │  }                                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Context Synchronization                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Session State (Streamlit)                           │   │
│  │  - st.session_state.messages                         │   │
│  │  - st.session_state.current_chat_id                  │   │
│  │  - st.session_state.chats_index                      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Agent State (LAUpathAgent)                           │   │
│  │  - self.messages (synchronized via load_messages())  │   │
│  │  - Context isolation per chat                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 Tool Execution Flow

The following diagram shows how different tools are executed and how their results are integrated into the agent's response:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│  "Analyze my profile" / "What majors suit me?" / etc.       │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LangChain Agent                                 │
│              (Tool Selection Logic)                         │
│  - Analyzes user intent                                      │
│  - Determines which tool(s) to invoke                      │
└───────┬───────────────┬───────────────┬───────────────┬───────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  RAG Tool    │ │ Profile      │ │ Major        │ │ Course Map │
│              │ │ Analyzer     │ │ Recommender  │ │ Retriever    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                 │                 │
       │                │                 │                 │
       ▼                ▼                 ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ ChromaDB     │ │ Rule-based   │ │ Scoring      │ │ File System  │
│ Search       │ │ Analysis      │ │ Algorithm    │ │ Access       │
│              │ │              │ │              │ │              │
│ Returns:     │ │ Returns:     │ │ Returns:     │ │ Returns:     │
│ - Top 5      │ │ - Eligibility │ │ - Top 3      │ │ - PDF Path   │
│   chunks     │ │ - Remedial    │ │   majors     │ │ - PDF Data   │
│ - Sources    │ │   English     │ │ - Scores     │ │              │
│              │ │ - Strengths   │ │ - Reasons    │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                 │                 │
       └────────────────┴─────────────────┴─────────────────┘
                        │
                        │ Tool results
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              ToolMessage Objects                             │
│  - Wrapped in LangChain ToolMessage format                   │
│  - Include tool name and execution status                    │
│  - Added to agent.messages                                   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Gemini LLM                                       │
│  - Receives tool results as context                          │
│  - Synthesizes information                                   │
│  - Generates natural language response                       │
│  - Returns AIMessage                                         │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Display                                │
│  - Message rendered in UI                                    │
│  - Tool calls shown with status                              │
│  - PDFs displayed inline (if applicable)                     │
│  - Saved to chat history                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.6 Data Flow Summary

The system handles three primary types of data flow:

1. **User Input Flow**: User messages → Session State → Agent → Tools → LLM → Response → UI
2. **Chat History Flow**: User Action → Management Function → File System (JSON) → Session State → Agent State
3. **RAG Query Flow**: User Query → Embedding → Vector Search → Document Chunks → LLM Context → Response

Each flow maintains data integrity through proper serialization, state synchronization, and error handling mechanisms.

---

## 3. Installation and Setup Instructions

This section provides comprehensive step-by-step instructions for installing and setting up the LAUpath AI project environment. Following these instructions will ensure that all dependencies are correctly installed, the system is properly configured, and the application is ready for use.

### 3.1 Prerequisites

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

### 3.2 Step-by-Step Installation Process

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

This script performs the following operations:
1. **Loads PDF Documents**: Extracts text from all PDF files in `data/lau_documents/`
2. **Chunks Documents**: Splits documents into manageable segments (1000 characters with 200-character overlap)
3. **Generates Embeddings**: Creates vector embeddings using Google's embedding model
4. **Stores in ChromaDB**: Persists embeddings in the `vector_db/` directory

**Expected Output**: The script will display progress messages indicating:
- Number of PDF files found
- Number of document chunks created
- Successful creation of the vector database
- Location of the stored database (`vector_db/` directory)

**Processing Time**: This step typically takes 2-5 minutes depending on:
- Number and size of PDF documents
- Internet speed (for API calls to generate embeddings)
- System performance

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

**Default URL**: The application will be accessible at `http://localhost:8501`

If the browser does not open automatically, manually navigate to the URL displayed in the terminal.

### 3.3 Verification and Testing

After installation, verify that the system is functioning correctly:

1. **Application Launch**: The Streamlit interface should load without errors
2. **Chat Interface**: The chat interface should be visible and functional
3. **Tool Functionality**: Test each tool:
   - **RAG Tool**: Ask "What are the admission requirements?" and verify that relevant information is retrieved
   - **Profile Analyzer**: Enter academic information and request profile analysis
   - **Major Recommender**: Request major recommendations
   - **Course Map Retriever**: Ask for a course map for a specific major

4. **Chat History**: Create a new chat, send a message, and verify that the chat is saved and appears in the sidebar

### 3.4 File Path Configuration

The system uses default paths for data storage. If customization is needed, the following paths can be modified:

#### Vector Database Path

**Default**: `./vector_db/`

To change the vector database location:
1. Edit `setup_rag.py`: Modify the `VECTOR_DB_DIRECTORY` variable (approximately line 15)
2. Edit `app.py`: Modify the `VECTOR_DB_DIRECTORY` variable (approximately line 30)
3. Ensure both files use the same path

#### PDF Documents Path

**Default**: `./data/lau_documents/`

To change the PDF documents directory:
1. Edit `setup_rag.py`: Modify the `PDF_DIRECTORY` variable (approximately line 14)
2. Ensure all LAU-related PDFs are moved to the new directory
3. Re-run `setup_rag.py` to rebuild the vector database

#### Chat History Path

**Default**: `./chat_history/`

To change the chat history storage location:
1. Edit `app.py`: Modify the `CHAT_HISTORY_DIR` variable (approximately line 38)
2. Ensure the directory exists or will be created automatically

**Example Path Modifications** (using absolute paths):

```python
# In setup_rag.py
PDF_DIRECTORY = "C:/Users/YourName/Documents/LAU_docs"
VECTOR_DB_DIRECTORY = "C:/Users/YourName/Documents/LAU_vector_db"

# In app.py
VECTOR_DB_DIRECTORY = "C:/Users/YourName/Documents/LAU_vector_db"
CHAT_HISTORY_DIR = "C:/Users/YourName/Documents/LAU_chat_history"
```

### 3.5 Troubleshooting Common Issues

#### Issue: "GEMINI_API_KEY not found"

**Cause**: The API key is not properly configured in the `.env` file.

**Solution**:
1. Verify that the `.env` file exists in the project root directory
2. Check that the file contains: `GEMINI_API_KEY=your_actual_key`
3. Ensure there are no extra spaces or quotation marks around the key
4. Restart the application after making changes

#### Issue: "Vector database not found" or "No collection found"

**Cause**: The RAG system has not been initialized.

**Solution**:
1. Run `python setup_rag.py` to create the vector database
2. Verify that the `vector_db/` directory exists and contains files
3. Check that the `VECTOR_DB_DIRECTORY` path in `app.py` matches the location of the database

#### Issue: "No PDF files found"

**Cause**: PDF documents are missing or in the wrong directory.

**Solution**:
1. Verify that PDF files are in `data/lau_documents/` directory
2. Check that file names end with `.pdf` extension
3. Ensure files are not corrupted and can be opened manually
4. Verify the `PDF_DIRECTORY` path in `setup_rag.py` is correct

#### Issue: Import Errors

**Cause**: Dependencies are not installed or virtual environment is not activated.

**Solution**:
1. Ensure the virtual environment is activated (check for `(venv)` in terminal prompt)
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`
3. Verify Python version is 3.8 or higher
4. Check for conflicting package versions

#### Issue: Port Already in Use

**Cause**: Another application is using port 8501.

**Solution**:
1. Stop other Streamlit applications running on port 8501
2. Or specify a different port: `streamlit run app.py --server.port 8502`
3. Access the application at the new port URL

#### Issue: Slow Performance

**Cause**: Large vector database or system resource constraints.

**Solution**:
1. Ensure sufficient RAM is available (close other applications)
2. Check internet connection speed (affects API calls)
3. Consider reducing the number of PDF documents if not all are needed
4. Verify that the vector database was created successfully

### 3.6 Post-Installation Checklist

After completing the installation, verify the following:

- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed without errors
- [ ] `.env` file exists with valid API key
- [ ] All required PDF documents are in `data/lau_documents/`
- [ ] `setup_rag.py` completed successfully
- [ ] `vector_db/` directory exists and contains database files
- [ ] Application starts without errors
- [ ] Chat interface is functional
- [ ] RAG tool retrieves information correctly
- [ ] Profile analyzer tool works with test data
- [ ] Major recommender tool provides recommendations
- [ ] Course map retriever displays PDFs correctly
- [ ] Chat history is saved and can be loaded

### 3.7 Additional Notes

- **API Key Security**: Never share your API key or commit it to version control. The `.env` file is automatically excluded via `.gitignore`.
- **Virtual Environment**: Always activate the virtual environment before running the application. If the terminal is closed, reactivate it using the commands in Step 2.
- **Database Updates**: If PDF documents are updated or new ones are added, re-run `setup_rag.py` to update the vector database.
- **Port Configuration**: By default, Streamlit uses port 8501. If this port is unavailable, use the `--server.port` flag to specify an alternative port.
- **Performance**: The first run may be slower as the system initializes. Subsequent interactions should be faster.

---

## 4. File Path Instructions

All file paths in the LAUpath AI system are implemented as **relative paths** using Python's `pathlib.Path` module, which ensures cross-platform compatibility. This means the code will work on Windows, macOS, and Linux without any modifications.

### Path Variables

The following paths are defined in the code:

- **`PDF_DIRECTORY`**: Defined in `setup_rag.py` (line 22) as `"./data/lau_documents"` - location of LAU document PDFs
- **`VECTOR_DB_DIRECTORY`**: Defined in both `setup_rag.py` (line 23) and `app.py` (line 34) as `"./vector_db"` - vector database storage location
- **`CHAT_HISTORY_DIR`**: Defined in `app.py` (line 38) as `"./chat_history"` - chat history JSON files storage

### Why No Modification is Needed

Since all paths are relative to the project root directory (where `app.py` is located), they automatically work on any computer regardless of:
- Operating system (Windows, macOS, Linux)
- Installation location
- User directory structure

The `pathlib.Path` module automatically handles platform-specific path separators, so the same code works seamlessly across all systems. As long as the project structure is maintained (with `data/lau_documents/`, `vector_db/`, and `chat_history/` directories relative to the project root), no path modifications are necessary.

---

## 5. API Information

The LAUpath AI project uses only one API, which is available free of charge:

### Google Gemini API

**API Name**: Google Gemini API  
**Purpose**: Used for both LLM inference (language model responses) and text embeddings (vector generation for RAG)  
**Cost**: Free tier available with usage limits  
**API Key Location**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)  
**Usage in Project**:
- **LLM Model**: `gemini-2.5-flash` - Used by the LAUpathAgent for generating conversational responses
- **Embedding Model**: `models/gemini-embedding-exp-03-07` - Used for creating vector embeddings from PDF documents

**Note**: No paid APIs are required for this project. The free tier of Google Gemini API provides sufficient quota for development, testing, and moderate usage. For production deployments with high traffic, please refer to Google's current pricing information.

---

## 6. Additional Aspects

This section covers other significant aspects of the LAUpath AI project that enhance user experience and provide a comprehensive understanding of the system's capabilities.

### 6.1 User-Friendly Academic Profile Input

To simplify the process of entering academic information, LAUpath AI provides an intuitive form-based interface accessible through a modal pop-up window. Instead of requiring users to type their academic credentials in natural language, the system offers structured input fields where students can easily enter:

- **High School GPA**: Numeric input field supporting both Lebanese (0-20) and US (0-4.0) grading scales
- **Lebanese Baccalaureate Score**: Optional field for official Lebanese exam results
- **SAT Score**: Optional field for standardized test performance
- **English Proficiency**: Dropdown selection for test type (TOEFL, IELTS, Duolingo) with corresponding score input

This form-based approach eliminates ambiguity and ensures accurate data entry, making it easier for students to provide their academic information. Once entered, the profile is saved in the session state and can be used for eligibility analysis with a single click of the "Analyze My Profile" button.

### 6.2 Interactive Career Test with Emoji-Based Responses

One of the most engaging features of LAUpath AI is the interactive Career Test, which provides a fun and intuitive way for students to discover suitable majors. The test is inspired by the O*NET Interest Profiler and presents questions in a modal pop-up window with the following characteristics:

- **Emoji-Based Answers**: Students respond to questions by selecting from a set of emoji options (😊, 😐, 😕, etc.), making the test more engaging and less intimidating than traditional text-based questionnaires
- **Navigation Controls**: The test includes "Previous" and "Next" buttons, allowing students to review and modify their answers before submission
- **Progress Tracking**: Students can see their progress through the test questions
- **Automatic Major Recommendation**: Upon completion, the test results are automatically fed into the `recommend_major` tool, which analyzes the responses and provides personalized top 3 major recommendations

This emoji-based approach makes the career assessment process more enjoyable and accessible, particularly for students who may find traditional career tests overwhelming or tedious. The visual nature of emoji responses also helps students express their preferences more intuitively.

### 6.3 Seamless Integration of Profile Analysis and Major Recommendations

The system seamlessly integrates the academic profile input with the eligibility analysis and major recommendation features. Students can:

1. Enter their academic information once through the form
2. Request eligibility analysis, which provides detailed feedback on admission probability, remedial requirements, and academic strengths
3. Take the career test to explore major options based on interests and preferences
4. Receive comprehensive guidance that combines both academic eligibility and personal interests

This integrated approach ensures that students receive holistic guidance that considers both their academic standing and their career aspirations, leading to more informed decision-making.

---

## 7. Important Notes and Compliance

This section addresses the important considerations and best practices followed in the LAUpath AI project, ensuring security, privacy, and proper attribution.

### 7.1 API Key Security

The project follows strict security practices for API key management:

- **No API Keys in Code**: No actual API keys are included in any submitted files, code, or documentation. The code uses environment variables to access API keys.
- **Placeholder Usage**: Documentation and code examples use `[insert API key here]` or `your_actual_api_key_here` as placeholders.
- **Secure Storage**: Actual API keys are stored securely in `.env` files, which are automatically excluded from version control via `.gitignore`.
- **Environment Variables**: The project uses the `python-dotenv` library to load API keys from `.env` files, following recommended security practices.

**Security Implementation**:
- The `.env` file is listed in `.gitignore` to prevent accidental commits
- API keys are never hardcoded in source files
- Users are instructed to create their own `.env` file with their personal API key

### 7.2 Paid APIs

As documented in Section 5 (API Information), this project does not utilize any paid APIs. The only API used is Google Gemini API, which is available through a free tier that provides sufficient quota for development, testing, and moderate usage. No costs are associated with running this project.

### 7.3 Data Privacy

The LAUpath AI project is designed with data privacy in mind:

- **No Persistent User Data**: Student academic information entered through the profile form is stored only in Streamlit's session state during the active session. This data is not persisted to disk or transmitted to external servers beyond the necessary API calls.
- **Chat History**: Chat conversations are stored locally in JSON files on the user's machine in the `chat_history/` directory. This data remains under the user's control and is not shared with third parties.
- **No Personal Information Collection**: The system does not collect, store, or transmit any personally identifiable information (PII) beyond what the user voluntarily enters during their session.
- **Public Documents Only**: The RAG system uses only publicly available LAU documents (admission requirements, program information, etc.) that do not contain sensitive personal information.
- **User Control**: Users can delete their chat history at any time through the application interface, giving them full control over their data.

**Compliance**: The project complies with privacy standards by ensuring that all user data remains local and under user control, with no external data collection or storage.

### 7.4 Environment Setup

The project follows recommended practices for environment management:

- **Virtual Environments**: Installation instructions (Section 3) strongly recommend using Python virtual environments to isolate project dependencies and prevent conflicts with other projects.
- **Dependencies File**: A `requirements.txt` file is provided listing all project dependencies with version specifications, enabling easy and reproducible installation.
- **Cross-Platform Compatibility**: The project is designed to work on Windows, macOS, and Linux without modification, using relative paths and cross-platform libraries.

### 7.5 Licensing and Attribution

The project respects licensing agreements and provides appropriate attribution:

**Third-Party Libraries and Tools**:
- **LangChain**: Open-source framework for building LLM applications (Apache 2.0 License)
- **Streamlit**: Open-source framework for building web applications (Apache 2.0 License)
- **ChromaDB**: Open-source vector database (Apache 2.0 License)
- **Google Generative AI SDK**: Provided by Google for accessing Gemini API
- **PyPDF**: Open-source PDF processing library
- **python-dotenv**: Open-source environment variable management (BSD License)

**External Resources**:
- **LAU Documents**: The PDF documents used in the RAG system are official LAU publications used for educational purposes in this academic project.
- **O*NET Interest Profiler**: The career test design is inspired by the O*NET Interest Profiler methodology, adapted for the LAU context.

**Attribution**: All third-party libraries and tools are properly attributed through the `requirements.txt` file and are used in compliance with their respective open-source licenses. The project does not modify or redistribute any third-party code.

**Academic Use**: This project is created for educational purposes as part of the COE548 course at Lebanese American University. All external resources are used in accordance with fair use principles for academic research and education.

---

