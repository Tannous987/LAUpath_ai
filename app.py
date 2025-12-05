"""
LAUpath AI - Main Application

A specialized LLM agent to help undergraduate students navigate LAU (Lebanese American University).
Features include RAG-based information retrieval, student profile analysis, and major recommendations.
"""

import asyncio
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from streamlit_modal import Modal
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool

# Import custom tools
from tools import analyze_student_profile, recommend_major, get_course_map

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_DIRECTORY = "./vector_db"
COLLECTION_NAME = "lau_documents"
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
MODEL_NAME = "gemini-2.5-flash"
CHAT_HISTORY_DIR = "./chat_history"


@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for LAU-related documents similar to the query.
    
    This tool retrieves relevant information from LAU documents including:
    - Admission requirements
    - Program information
    - Tuition and fees
    - Scholarships and financial aid
    - English proficiency requirements
    - And other LAU-related information
    
    Args:
        query (str): The search query string to find relevant documents
    
    Returns:
        str: A concatenated string of the top 5 most similar document contents found
    """
    try:
        # Initialize embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=API_KEY
        )
        
        # Initialize/connect to vector database
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_DIRECTORY,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Perform similarity search and get top 5 results
        results = vector_store.similarity_search(query=query, k=5)
        
        # Combine all document contents into single string with source information
        result_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            result_parts.append(f"[Source: {source}]\n{doc.page_content}\n")
        
        result_str = "\n---\n".join(result_parts)
        
        return result_str if result_str else "No relevant documents found."
    
    except Exception as e:
        return f"Error searching vector database: {str(e)}"


class LAUpathAgent:
    """
    Main agent class for LAUpath AI.
    
    This class manages the LLM agent with integrated tools for:
    - RAG-based information retrieval
    - Student profile analysis
    - Major recommendations
    """
    
    def __init__(self, model_name: str = MODEL_NAME, temperature: float = 0.3):
        """
        Initialize LAUpathAgent with a language model and tools.
        
        Args:
            model_name (str): The Gemini model to use. Default is "gemini-2.5-flash".
            temperature (float): The temperature for response generation. Default is 0.3.
        """
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        
        self.api_key = API_KEY
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature
        )
        
        # Create agent with all tools
        self.agent = create_agent(
            self.llm,
            tools=[search_vector_db, analyze_student_profile, recommend_major, get_course_map]
        )
        
        # Initialize conversation history with system message
        self.messages = [self._get_system_message()]
    
    def _get_system_message(self) -> SystemMessage:
        """Get the system message for the agent."""
        return SystemMessage(content="""
<identity>
You are LAUpath AI, a specialized assistant designed to help undergraduate students navigate Lebanese American University (LAU). 
You are knowledgeable, friendly, and supportive, helping students make informed decisions about their academic journey.
</identity>

<purpose>
Your purpose is to:
1. Help students understand LAU's programs, requirements, and opportunities
2. Analyze student academic profiles and provide eligibility assessments
3. Recommend suitable majors based on student interests, strengths, and goals
4. Answer questions about admissions, financial aid, scholarships, and university policies
5. Guide students through the application process and academic planning

You have access to:
- A comprehensive vector database of LAU documents (admissions, programs, fees, etc.)
- A student profile analyzer tool that evaluates academic records and determines requirements
- A major recommendation engine that provides personalized major suggestions
- A course map retriever tool that provides curriculum PDFs for specific majors

Always be helpful, accurate, and encouraging. When you don't have specific information, use the search_vector_db tool to find it.
When students provide their academic information, use the analyze_student_profile tool to give them detailed feedback.
When students need guidance on choosing a major, use the recommend_major tool to provide personalized recommendations.
When students ask about a curriculum, course map, or course plan for a specific major, use the get_course_map tool to retrieve and display the appropriate PDF. 
IMPORTANT: When the tool successfully returns a course map, respond with: "Here is the course map for [major name]. You can download the PDF if you want." The PDF will be displayed automatically below your message. Do NOT mention file paths or technical details like "COURSE_MAP_PATH" in your response. If a course map is not available, inform the student politely that it's not currently available without showing any technical paths or error details.
</purpose>

<guidelines>
- Always be respectful and supportive
- Use tools when appropriate to provide accurate, up-to-date information
- Explain your reasoning when making recommendations
- Encourage students to explore their options
- Be clear about requirements and deadlines
- If information is not available, admit it and suggest alternatives
</guidelines>
""")
    
    def load_messages(self, messages: list) -> None:
        """
        Load conversation history for this chat session.
        This ensures each chat has its own isolated context.
        
        Args:
            messages (list): List of LangChain message objects (HumanMessage/AIMessage/ToolMessage)
        """
        # Start with system message, then add all loaded messages
        self.messages = [self._get_system_message()]
        # Filter out system messages from loaded messages and add the rest
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                self.messages.append(msg)
    
    def reset_messages(self) -> None:
        """Reset conversation history to just the system message (for new chats)."""
        self.messages = [self._get_system_message()]
    
    def send_message(self, message: str) -> list:
        """
        Send a message and get response from the agent.
        
        Args:
            message (str): The user's message
        
        Returns:
            list: List of new messages from this interaction (may include tool calls)
        """
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length to identify new messages later
        history_length = len(self.messages)
        
        # Get response from agent, including any tool usage
        try:
            self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        except Exception as e:
            # Handle errors gracefully
            error_message = AIMessage(content=f"I encountered an error: {str(e)}. Please try again or rephrase your question.")
            self.messages.append(error_message)
        
        # Extract only the new messages from this interaction
        new_messages = self.messages[history_length:]
        
        return new_messages


def get_text(content: Any) -> str:
    """
    Normalize Gemini/LangChain content to a plain string.
    
    Args:
        content: The content to normalize (can be str, list, or other types)
    
    Returns:
        str: Normalized string representation of the content
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # content is a list of parts like {'type': 'text', 'text': '...'}
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(part.get("text", ""))
        return "".join(pieces)
    # Fallback
    return str(content)


def ensure_chat_history_dir() -> Path:
    """
    Ensure that the chat history directory exists.

    Returns:
        Path: Path object for the chat history directory.
    """
    chat_dir = Path(CHAT_HISTORY_DIR)
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir


def _chat_file_path(chat_id: str) -> Path:
    """Get the file path for a given chat ID."""
    return ensure_chat_history_dir() / f"{chat_id}.json"


def load_chat_index() -> List[Dict[str, Any]]:
    """
    Load metadata for all saved chats.
    Sorted by created_at timestamp (newest first).

    Returns:
        List[Dict[str, Any]]: List of chat metadata dictionaries, sorted by creation date (newest first).
    """
    chat_dir = ensure_chat_history_dir()
    chats: List[Dict[str, Any]] = []
    for file in chat_dir.glob("*.json"):
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            chats.append(
                {
                    "id": data.get("id"),
                    "title": data.get("title") or "New chat",
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        except Exception:
            continue
    
    # Sort by created_at timestamp (newest first), fallback to updated_at if created_at is missing
    chats.sort(
        key=lambda x: x.get("created_at") or x.get("updated_at") or "",
        reverse=True
    )
    return chats


def load_chat_messages(chat_id: str) -> List[Any]:
    """
    Load chat messages for a given chat ID and convert them to LangChain messages.

    Args:
        chat_id (str): ID of the chat to load.

    Returns:
        List[Any]: List of LangChain message objects (HumanMessage/AIMessage).
    """
    path = _chat_file_path(chat_id)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    raw_messages = data.get("messages", [])
    messages: List[Any] = []
    for msg in raw_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        # We intentionally skip tool/system messages for persistence simplicity.
    return messages


def save_current_chat() -> None:
    """
    Save the current chat (from session state) to disk.

    This function persists only user and assistant messages for clarity.
    """
    chat_id = st.session_state.get("current_chat_id")
    if not chat_id:
        return

    messages = st.session_state.get("messages", [])
    if not isinstance(messages, list):
        return

    # Serialize messages
    serialized_messages = []
    title = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            serialized_messages.append({"role": "user", "content": content})
            # Use first user message as chat title (truncated)
            if not title:
                simple = str(content).strip().split("\n")[0]
                title = simple[:40] + ("..." if len(simple) > 40 else "")
        elif isinstance(msg, AIMessage) and msg.content:
            serialized_messages.append({"role": "assistant", "content": get_text(msg.content)})

    if not title:
        title = "New chat"

    now = datetime.utcnow().isoformat()

    data = {
        "id": chat_id,
        "title": title,
        "created_at": st.session_state.get("current_chat_created_at", now),
        "updated_at": now,
        "messages": serialized_messages,
    }

    # Persist file
    path = _chat_file_path(chat_id)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        return

    # Update in-memory chat index and sort by created_at (newest first)
    chats = st.session_state.get("chats_index", [])
    existing = next((c for c in chats if c.get("id") == chat_id), None)
    if existing:
        existing["title"] = title
        existing["updated_at"] = now
    else:
        chats.append(
            {
                "id": chat_id,
                "title": title,
                "created_at": data["created_at"],
                "updated_at": now,
            }
        )
    # Sort chats by created_at (newest first) to ensure new chats appear at top
    chats.sort(
        key=lambda x: x.get("created_at") or x.get("updated_at") or "",
        reverse=True
    )
    st.session_state.chats_index = chats
    st.session_state.current_chat_created_at = data["created_at"]


def delete_chat(chat_id: str) -> None:
    """
    Delete a chat by removing its file and updating the chat index.
    
    Args:
        chat_id (str): The ID of the chat to delete
    """
    # Delete the chat file
    chat_file = _chat_file_path(chat_id)
    if chat_file.exists():
        try:
            chat_file.unlink()
        except Exception:
            pass  # File might already be deleted
    
    # Remove from session state index
    chats = st.session_state.get("chats_index", [])
    st.session_state.chats_index = [c for c in chats if c.get("id") != chat_id]
    
    # If this was the current chat, switch to another chat or create a new one
    if st.session_state.get("current_chat_id") == chat_id:
        remaining_chats = st.session_state.chats_index
        if remaining_chats:
            # Switch to the most recent chat
            new_chat_id = remaining_chats[0]["id"]
            st.session_state.current_chat_id = new_chat_id
            loaded_messages = load_chat_messages(new_chat_id)
            st.session_state.messages = loaded_messages
            if "agent" in st.session_state:
                st.session_state.agent.load_messages(loaded_messages)
        else:
            # No chats left, create a new one
            create_new_chat()


def create_new_chat() -> None:
    """
    Create a new empty chat session and set it as current.
    """
    chat_id = str(uuid4())
    now = datetime.utcnow().isoformat()
    st.session_state.current_chat_id = chat_id
    st.session_state.current_chat_created_at = now
    st.session_state.messages = []
    # Reset agent's internal messages for the new chat
    if "agent" in st.session_state:
        st.session_state.agent.reset_messages()
    # Persist immediately so it shows up in index
    save_current_chat()


def load_student_profile() -> Optional[Dict[str, Any]]:
    """
    Load student profile from session state.
    
    Returns:
        Optional[Dict]: Student profile data if available, None otherwise
    """
    return st.session_state.get("student_profile", None)


def save_student_profile(profile: Dict[str, Any]) -> None:
    """
    Save student profile to session state.
    
    Args:
        profile (Dict): Student profile data to save
    """
    st.session_state.student_profile = profile


async def main():
    """Main Streamlit application function."""
    # Page configuration
    st.set_page_config(
        page_title="LAUpath AI",
        page_icon="🎓",
        layout="wide",
    )
    # Global UI styling
    st.markdown(
        """
        <style>
        /* Center main content and limit width for a cleaner chat look */
        .main .block-container {
            max-width: 900px;
            padding-top: 1rem;
            padding-bottom: 4rem;
        }
        /* Make sidebar narrower */
        [data-testid="stSidebar"] {
            width: 260px !important;
        }
        /* Tighter radio spacing in sidebar for chat list */
        div[role="radiogroup"] > label {
            padding-top: 2px;
            padding-bottom: 2px;
        }

       
        section[data-testid="stSidebar"] button[kind="primary"] {
            background-color: #374151 !important;  /* dark grey */
            color: #ffffff !important;
            border-color: #374151 !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #1F2937 !important;  /* even darker */
            border-color: #1F2937 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Ensure vector database exists
    if not os.path.exists(VECTOR_DB_DIRECTORY):
        st.error("⚠️ Vector database not found!")
        st.info("Please run `python setup_rag.py` first to set up the RAG system.")
        st.stop()

    # Initialize chat history structures
    ensure_chat_history_dir()
    if "chats_index" not in st.session_state:
        st.session_state.chats_index = load_chat_index()
    if "current_chat_id" not in st.session_state:
        if st.session_state.chats_index:
            # Use most recent chat
            st.session_state.current_chat_id = st.session_state.chats_index[0]["id"]
            st.session_state.current_chat_created_at = st.session_state.chats_index[0].get(
                "created_at", datetime.utcnow().isoformat()
            )
        else:
            create_new_chat()

    # Sidebar - chat history and controls (ChatGPT-style)
    with st.sidebar:
        st.title("💬 LAUpath Chats")

        if st.button("➕ New chat", use_container_width=True):
            create_new_chat()
            # Reload chat index to ensure new chat is at top (sorted by created_at)
            st.session_state.chats_index = load_chat_index()
            st.rerun()

        chats = st.session_state.get("chats_index", [])
        if chats:
            st.markdown("**Conversations**")
            current_chat_id = st.session_state.get("current_chat_id")
            
            # Initialize pending deletion state
            if "pending_deletion" not in st.session_state:
                st.session_state.pending_deletion = None
            
            # Show confirmation dialog if a deletion is pending
            if st.session_state.pending_deletion:
                pending_chat_id = st.session_state.pending_deletion
                pending_chat = next((c for c in chats if c.get("id") == pending_chat_id), None)
                pending_title = pending_chat.get("title") if pending_chat else "this chat"
                
                st.warning(f"⚠️ Delete '{pending_title}'?")
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅confirm", key="confirm_delete", use_container_width=True):
                        delete_chat(pending_chat_id)
                        st.session_state.pending_deletion = None
                        st.rerun()
                with confirm_col2:
                    if st.button("❌cancel", key="cancel_delete", use_container_width=True):
                        st.session_state.pending_deletion = None
                        st.rerun()
                st.markdown("---")
            
            # Display each chat with a delete button
            for chat in chats:
                chat_id = chat["id"]
                chat_title = chat.get("title") or "New chat"
                is_selected = (chat_id == current_chat_id)
                
                # Skip showing delete button if this chat is pending deletion (already shown in confirmation)
                if st.session_state.pending_deletion == chat_id:
                    continue
                
                # Create a row with chat button and delete button
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Chat button
                    button_key = f"chat_btn_{chat_id}"
                    if st.button(
                        chat_title,
                        key=button_key,
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        if chat_id != current_chat_id:
                            st.session_state.current_chat_id = chat_id
                            loaded_messages = load_chat_messages(chat_id)
                            st.session_state.messages = loaded_messages
                            # Sync agent's internal messages with this chat's history
                            if "agent" in st.session_state:
                                st.session_state.agent.load_messages(loaded_messages)
                            st.rerun()
                
                with col2:
                    # Delete button
                    delete_key = f"delete_btn_{chat_id}"
                    if st.button("🗑️", key=delete_key, help="Delete this chat"):
                        st.session_state.pending_deletion = chat_id
                        st.rerun()
            
        else:
            st.info("No chats yet. Start a new conversation!")

        st.markdown("---")
        st.caption("Chats are saved and will persist even if you close the app.")

    # Title and description
    st.title("🎓 LAUpath AI")
    st.markdown("**Your AI guide to Lebanese American University**")
    st.markdown("Get personalized guidance on admissions, programs, majors, and more!")

    # Initialize toggle state for profile/tools panel
    if "show_profile_panel" not in st.session_state:
        st.session_state.show_profile_panel = False

    # Initialize agent if not already in session state
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = LAUpathAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()

    # Initialize message history in session state from current chat if not already set
    if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
        current_id = st.session_state.get("current_chat_id")
        if current_id:
            loaded_messages = load_chat_messages(current_id)
            st.session_state.messages = loaded_messages
            # Sync agent's internal messages with loaded chat history
            if "agent" in st.session_state:
                st.session_state.agent.load_messages(loaded_messages)
        else:
            st.session_state.messages = []
            # Reset agent's messages for new empty chat
            if "agent" in st.session_state:
                st.session_state.agent.reset_messages()
    # Handle quick actions (use saved profile data, independent of UI location)
    profile_for_tools = load_student_profile()
    if "quick_action" in st.session_state:
        action = st.session_state.quick_action
        del st.session_state.quick_action
        
        if action == "analyze_profile" and profile_for_tools:
            # Format profile data for the tool
            profile_str = f"""Please analyze my student profile using the analyze_student_profile tool with the following data:
- School GPA: {profile_for_tools.get('school_gpa')}
- Lebanese Exam Score: {profile_for_tools.get('lebanese_exam_score')}
- SAT Score: {profile_for_tools.get('sat_score')}
- English Proficiency Type: {profile_for_tools.get('english_proficiency_type')}
- English Proficiency Score: {profile_for_tools.get('english_proficiency_score')}

Please use the analyze_student_profile tool with these exact values."""
            prompt = profile_str
        elif action == "recommend_major":
            prompt = "I need help choosing a major. Can you guide me through the recommendation process? Please use the recommend_major tool to help me."
        else:
            prompt = None
        
        if prompt:
            st.session_state.messages.append(HumanMessage(content=prompt))
            messages = st.session_state.agent.send_message(prompt)
            st.session_state.messages.extend(messages)
            # Persist updated chat history
            save_current_chat()
            st.rerun()
    
    # Display all previous messages from session state
    for message in st.session_state.messages:
        # Handle AI message with content (regular response)
        if isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.markdown(get_text(message.content))
        
        # Handle AI message without content (tool call)
        elif isinstance(message, AIMessage) and not message.content and hasattr(message, 'tool_calls') and message.tool_calls:
            with st.chat_message("assistant"):
                tool_name = message.tool_calls[0]['name']
                tool_args = str(message.tool_calls[0]['args'])
                with st.status(f"🔧 Using tool: {tool_name}", expanded=False):
                    st.code(tool_args, language="json")
        
        # Handle tool execution result message
        elif isinstance(message, ToolMessage):
            with st.chat_message("assistant"):
                content = message.content
                # Check if this is a course map path
                if content.startswith("COURSE_MAP_PATH:"):
                    # Parse the content: COURSE_MAP_PATH:path|major_name
                    parts = content.replace("COURSE_MAP_PATH:", "").strip().split("|")
                    pdf_path = parts[0].strip()
                    major_display = parts[1].strip() if len(parts) > 1 else "the requested major"
                    
                    pdf_file = Path(pdf_path)
                    if pdf_file.exists():
                        # Read PDF file
                        with open(pdf_file, "rb") as pdf_file_obj:
                            pdf_bytes = pdf_file_obj.read()
                        
                        # Display PDF in an embedded viewer using iframe
                        import base64
                        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'''
                        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                                width="100%" 
                                height="600px" 
                                style="border: 1px solid #ddd; border-radius: 5px;">
                        </iframe>
                        '''
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        
                        # Also provide download button
                        st.download_button(
                            label="📥 Download Course Map PDF",
                            data=pdf_bytes,
                            file_name=pdf_file.name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        # If file doesn't exist, show user-friendly message
                        st.info("The course map for this major is not currently available.")
                else:
                    with st.status("📊 Tool result", expanded=False):
                        # Truncate very long results for display
                        if len(content) > 1000:
                            st.markdown(content[:1000] + "\n\n... (truncated, full result used by agent)")
                        else:
                            st.markdown(content)
        
        # Handle user message
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # Initialize modals for academic profile and career test
    modal = Modal(
        "📋 Academic Profile",
        key="academic-profile-modal",
        padding=30,
        max_width=600
    )
    
    career_test_modal = Modal(
        "🎯 Career Interest Test",
        key="career-test-modal",
        padding=30,
        max_width=700
    )

    # Add custom CSS for professional modal styling

    # Profile button placed near the input area
    bottom_left, bottom_middle, bottom_right = st.columns([3, 1, 1])
    with bottom_right:
        open_modal = st.button("👤 Academic Profile", key="toggle_profile_bottom", use_container_width=True)
    with bottom_middle:
        open_career_test = st.button("🎯 Career Test", key="toggle_career_test", use_container_width=True)
    
    # Open modals if buttons were clicked
    if open_modal:
        modal.open()
    if open_career_test:
        career_test_modal.open()
    
    profile = load_student_profile()
    
    # Display modal content
    if modal.is_open():
        with modal.container():
            with st.form("student_profile_form"):
                #st.markdown("### Provide or update your academic details for personalized guidance.")
                
                school_gpa = st.number_input(
                    "High School GPA",
                    min_value=0.0,
                    max_value=20.0,
                    value=(profile.get("school_gpa") or 0.0) if profile else 0.0,
                    step=0.1,
                    help="Enter your high school GPA (Lebanese system: 0-20, US system: 0-4.0)",
                )

                lebanese_exam = st.number_input(
                    "Lebanese Baccalaureate Score (Optional)",
                    min_value=0.0,
                    max_value=20.0,
                    value=(profile.get("lebanese_exam_score") or 0.0) if profile else 0.0,
                    step=0.1,
                )

                sat_score = st.number_input(
                    "SAT Score (Optional)",
                    min_value=0,
                    max_value=1600,
                    value=(profile.get("sat_score") or 0) if profile else 0,
                    step=10,
                )

                st.subheader("English Proficiency")
                english_test_type = st.selectbox(
                    "English Test Type",
                    ["None", "TOEFL", "IELTS", "Duolingo"],
                    index=(
                        ["None", "TOEFL", "IELTS", "Duolingo"].index(
                            profile.get("english_proficiency_type")
                        )
                        if profile and profile.get("english_proficiency_type") in ["TOEFL", "IELTS", "Duolingo"]
                        else 0
                    ),
                    help="Select the type of English proficiency test you took",
                )

                english_score = st.number_input(
                    "English Proficiency Score",
                    min_value=0.0,
                    max_value=200.0,
                    value=(profile.get("english_proficiency_score") or 0.0) if profile else 0.0,
                    step=0.1,
                    help="Enter your English proficiency test score",
                )

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
                with col2:
                    analyze_btn = st.form_submit_button("🔍 Check Eligibility", use_container_width=True)

                if submitted:
                    profile = {
                        "school_gpa": school_gpa if school_gpa > 0 else None,
                        "lebanese_exam_score": lebanese_exam if lebanese_exam > 0 else None,
                        "sat_score": sat_score if sat_score > 0 else None,
                        "english_proficiency_type": english_test_type if english_test_type != "None" else None,
                        "english_proficiency_score": english_score if english_score > 0 else None,
                    }
                    save_student_profile(profile)
                    #st.success("✅ Profile saved! You can now ask me to analyze it.")
                    modal.close()
                    st.rerun()
                
                if analyze_btn:
                    # Save profile first if there are values
                    profile_data = {
                        "school_gpa": school_gpa if school_gpa > 0 else None,
                        "lebanese_exam_score": lebanese_exam if lebanese_exam > 0 else None,
                        "sat_score": sat_score if sat_score > 0 else None,
                        "english_proficiency_type": english_test_type if english_test_type != "None" else None,
                        "english_proficiency_score": english_score if english_score > 0 else None,
                    }
                    # Check if at least one field has a value
                    if any(v is not None for v in profile_data.values()):
                        save_student_profile(profile_data)
                        st.session_state.quick_action = "analyze_profile"
                        modal.close()
                        st.rerun()
                    else:
                        st.warning("Please enter your academic information first.")

#             st.markdown(
#                 """
# **Tips**
# - Use your saved profile for eligibility analysis  
# - Ask follow-up questions about the analysis or recommendations  
# """
#             )

    # Career Test Modal
    if career_test_modal.is_open():
        with career_test_modal.container():
            # Initialize career test state
            if "career_test_answers" not in st.session_state:
                st.session_state.career_test_answers = {}
            if "career_test_current_question" not in st.session_state:
                st.session_state.career_test_current_question = 0
            
            # Career test questions specifically designed for LAU majors
            career_questions = [
                # Engineering-focused questions
                {
                    "question": "How interested are you in programming and software development?",
                    "target_majors": ["Computer Engineering (COE)", "Computer and Information Engineering (CIE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How much do you enjoy working with computer hardware and circuits?",
                    "target_majors": ["Computer Engineering (COE)", "Electrical Engineering (ELE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in networks, cybersecurity, and information systems?",
                    "target_majors": ["Computer and Information Engineering (CIE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How much do you like working with electrical systems and power?",
                    "target_majors": ["Electrical Engineering (ELE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in optimizing processes and improving efficiency?",
                    "target_majors": ["Industrial Engineering (INE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How much do you enjoy designing and building mechanical systems?",
                    "target_majors": ["Mechanical Engineering (MCE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in oil, gas, and energy industries?",
                    "target_majors": ["Petroleum Engineering (PTE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Business and Management
                {
                    "question": "How much do you enjoy business strategy and management?",
                    "target_majors": ["Business Administration"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in entrepreneurship and starting businesses?",
                    "target_majors": ["Business Administration"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Health Sciences
                {
                    "question": "How much do you want to become a doctor and treat patients?",
                    "target_majors": ["Medicine"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in providing direct patient care as a nurse?",
                    "target_majors": ["Nursing"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Social Sciences
                {
                    "question": "How much do you enjoy studying human behavior and mental health?",
                    "target_majors": ["Psychology"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Arts and Design
                {
                    "question": "How interested are you in designing buildings and spaces?",
                    "target_majors": ["Architecture"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How much do you enjoy creating visual designs and graphics?",
                    "target_majors": ["Graphic Design"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Communication
                {
                    "question": "How interested are you in journalism and news reporting?",
                    "target_majors": ["Journalism"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # Education
                {
                    "question": "How much do you enjoy teaching and working with students?",
                    "target_majors": ["Education"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                # General interest questions
                {
                    "question": "How much do you enjoy solving complex mathematical problems?",
                    "target_majors": ["Computer Engineering (COE)", "Computer and Information Engineering (CIE)", "Electrical Engineering (ELE)", "Industrial Engineering (INE)", "Mechanical Engineering (MCE)", "Petroleum Engineering (PTE)"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How interested are you in working in a laboratory or research setting?",
                    "target_majors": ["Computer Engineering (COE)", "Computer and Information Engineering (CIE)", "Electrical Engineering (ELE)", "Mechanical Engineering (MCE)", "Medicine", "Psychology"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                },
                {
                    "question": "How much do you prefer working with people versus working with things?",
                    "target_majors": ["Business Administration", "Medicine", "Nursing", "Psychology", "Education", "Journalism"],
                    "emoji_options": ["😊", "😐", "😕", "😞"]
                }
            ]
            
            total_questions = len(career_questions)
            current_q = st.session_state.career_test_current_question
            question_data = career_questions[current_q]
            
            # Progress indicator
            progress = (current_q + 1) / total_questions
            st.progress(progress)
            st.caption(f"Question {current_q + 1} of {total_questions}")
            
            # Display question
            st.markdown(f"### {question_data['question']}")
            st.markdown("---")
            
            # Emoji options with labels
            emoji_labels = {
                "😊": "Strongly Like",
                "😐": "Somewhat Like",
                "😕": "Somewhat Dislike",
                "😞": "Strongly Dislike"
            }
            
            # Create columns for emoji buttons (equal width)
            col1, col2, col3, col4 = st.columns(4)
            selected_emoji = None
            
            # Format button text: emoji on first line (centered with padding), text on second line
            # Using newlines and spacing to center emoji visually
            button_texts = [
                f"\n{question_data['emoji_options'][0]}\n\n{emoji_labels['😊']}",
                f"\n{question_data['emoji_options'][1]}\n\n{emoji_labels['😐']}",
                f"\n{question_data['emoji_options'][2]}\n\n{emoji_labels['😕']}",
                f"\n{question_data['emoji_options'][3]}\n\n{emoji_labels['😞']}"
            ]
            
            with col1:
                if st.button(button_texts[0], 
                            key=f"emoji_0_q{current_q}", 
                            use_container_width=True):
                    selected_emoji = question_data['emoji_options'][0]
            with col2:
                if st.button(button_texts[1], 
                            key=f"emoji_1_q{current_q}", 
                            use_container_width=True):
                    selected_emoji = question_data['emoji_options'][1]
            with col3:
                if st.button(button_texts[2], 
                            key=f"emoji_2_q{current_q}", 
                            use_container_width=True):
                    selected_emoji = question_data['emoji_options'][2]
            with col4:
                if st.button(button_texts[3], 
                            key=f"emoji_3_q{current_q}", 
                            use_container_width=True):
                    selected_emoji = question_data['emoji_options'][3]
            
            # Save answer if emoji was clicked
            if selected_emoji:
                st.session_state.career_test_answers[current_q] = {
                    "answer": selected_emoji,
                    "target_majors": question_data.get('target_majors', []),
                    "score": {"😊": 4, "😐": 2, "😕": 1, "😞": 0}[selected_emoji]
                }
                if current_q < total_questions - 1:
                    st.session_state.career_test_current_question += 1
                    st.rerun()
            
            # Navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
            with nav_col1:
                if current_q > 0:
                    if st.button("← Previous", key="prev_question", use_container_width=True):
                        st.session_state.career_test_current_question -= 1
                        st.rerun()
            
            with nav_col2:
                # Show current answer if exists
                if current_q in st.session_state.career_test_answers:
                    answer = st.session_state.career_test_answers[current_q]["answer"]
                    st.info(f"Selected: {answer} {emoji_labels[answer]}")
            
            with nav_col3:
                if current_q < total_questions - 1:
                    if st.button("Next →", key="next_question", use_container_width=True):
                        if current_q not in st.session_state.career_test_answers:
                            st.warning("Please select an answer before proceeding.")
                        else:
                            st.session_state.career_test_current_question += 1
                            st.rerun()
                else:
                    # Submit button on last question
                    if st.button("Submit Test", key="submit_test", use_container_width=True, type="primary"):
                        if current_q not in st.session_state.career_test_answers:
                            st.warning("Please select an answer before submitting.")
                        else:
                            # Calculate major scores directly from test answers
                            major_scores = {}
                            
                            # Score each major based on answers
                            for q_idx, answer_data in st.session_state.career_test_answers.items():
                                target_majors = answer_data.get("target_majors", [])
                                score = answer_data["score"]
                                
                                # Distribute score to all target majors for this question
                                for major in target_majors:
                                    if major not in major_scores:
                                        major_scores[major] = 0
                                    major_scores[major] += score
                            
                            # Get top 3 majors
                            sorted_majors = sorted(major_scores.items(), key=lambda x: x[1], reverse=True)
                            top_majors = [major for major, score in sorted_majors[:3] if score > 0]
                            
                            # If we have top majors, use them; otherwise generate from interests
                            if top_majors:
                                # Build interests and strengths from top majors
                                interests_list = []
                                strengths_list = []
                                
                                # Map majors to interests and strengths (comprehensive list)
                                major_to_interests = {
                                    "Computer Engineering (COE)": "programming, hardware, embedded systems, computer architecture",
                                    "Computer and Information Engineering (CIE)": "networks, cybersecurity, information systems, software",
                                    "Computer Science": "programming, algorithms, software, computing, technology",
                                    "Electrical Engineering (ELE)": "electronics, power systems, circuits, electrical systems",
                                    "Industrial Engineering (INE)": "optimization, process improvement, operations, efficiency",
                                    "Mechanical Engineering (MCE)": "mechanical design, manufacturing, thermodynamics, machines",
                                    "Mechatronics Engineering": "robotics, automation, mechanical systems, electronics, control",
                                    "Civil Engineering": "construction, infrastructure, design, building, structures",
                                    "Chemical Engineering": "chemistry, processes, manufacturing, materials, reactions",
                                    "Petroleum Engineering (PTE)": "oil, gas, energy, drilling, reservoir engineering",
                                    "Business Administration": "business, management, finance, entrepreneurship, strategy",
                                    "Business": "business, management, finance, marketing, entrepreneurship",
                                    "Business Emphasis Hospitality & Tourism Management": "hospitality, tourism, hotel management, customer service, travel",
                                    "Economics": "economics, finance, markets, policy, analysis",
                                    "Medicine": "health, biology, helping people, medical diagnosis, patient care",
                                    "Nursing": "healthcare, patient care, helping, wellness, biology",
                                    "Pharmacy": "pharmacy, medicine, drugs, health, patient care",
                                    "Nutrition and Dietetics": "nutrition, health, food science, wellness, diet planning",
                                    "Biology": "biology, life sciences, research, nature, experiments",
                                    "Bioinformatics": "biology, computing, data analysis, genetics, programming, research",
                                    "Chemistry": "chemistry, experiments, molecules, reactions, research",
                                    "Applied Physics": "physics, research, experiments, mathematics, scientific analysis",
                                    "Mathematics": "mathematics, problem-solving, logic, analysis, theoretical concepts",
                                    "Psychology": "human behavior, mental health, counseling, research, therapy",
                                    "Political Science/International Affairs": "politics, international relations, government, policy, diplomacy",
                                    "Architecture": "design, building, creativity, space, urban planning",
                                    "Graphic Design": "art, design, creativity, visual, media, branding",
                                    "Interior Design": "interior design, space, decoration, architecture, creativity",
                                    "Fashion Design": "fashion, design, clothing, creativity, style, textiles",
                                    "Studio Art": "art, painting, sculpture, visual arts, creativity, exhibition",
                                    "Journalism": "writing, media, news, communication, storytelling, investigation",
                                    "Multimedia Journalism": "journalism, multimedia, video, digital media, storytelling, news",
                                    "TV & Film": "television, film, video production, cinematography, directing, media",
                                    "Communication": "communication, media, public relations, advertising, marketing",
                                    "Performing Arts": "theater, acting, dance, music, performance, entertainment",
                                    "Education": "teaching, education, helping, children, learning, curriculum",
                                    "English": "literature, writing, language, reading, analysis",
                                    "Translation": "languages, translation, linguistics, communication, cultural exchange"
                                }
                                
                                for major in top_majors:
                                    if major in major_to_interests:
                                        interests_list.append(major_to_interests[major])
                                        # Extract major name for strengths
                                        major_name = major.split("(")[0].strip() if "(" in major else major
                                        strengths_list.append(major_name.lower())
                                
                                interests = ", ".join(interests_list) if interests_list else "general academic interests"
                                academic_strengths = ", ".join(strengths_list) if strengths_list else "general academic strengths"
                                career_goals = f"Career in {', '.join([m.split('(')[0].strip() if '(' in m else m for m in top_majors])}"
                                preferred_work_environment = "office, lab, field, studio, hospital, clinic"  # Comprehensive
                            else:
                                # Fallback if no clear matches
                                interests = "general academic and professional interests"
                                academic_strengths = "general academic strengths"
                                career_goals = "exploring career options"
                                preferred_work_environment = "various work environments"
                            
                            # Get student profile if available
                            profile = load_student_profile()
                            profile_json = json.dumps(profile) if profile else None
                            
                            # Trigger the recommend_major tool via agent
                            test_results_prompt = f"""Based on my career interest test results, please recommend specific LAU majors for me. 
My test results show interest in: {', '.join(top_majors) if top_majors else 'various fields'}.
My interests: {interests}
My academic strengths: {academic_strengths}
My career goals: {career_goals}
Preferred work environment: {preferred_work_environment}

Please use the recommend_major tool with these details to suggest the top 3 specific majors available at LAU (like Computer Engineering, Electrical Engineering, etc.) that best fit my interests. Do not suggest general categories - only specific LAU majors."""
                            
                            st.session_state.messages.append(HumanMessage(content=test_results_prompt))
                            messages = st.session_state.agent.send_message(test_results_prompt)
                            st.session_state.messages.extend(messages)
                            save_current_chat()
                            
                            # Reset test state
                            st.session_state.career_test_answers = {}
                            st.session_state.career_test_current_question = 0
                            career_test_modal.close()
                            st.rerun()

    # Get user input from chat interface
    prompt = st.chat_input("Ask me anything about LAU, your profile, or major selection...")
    
    if prompt:
        # Add user's message to session state history
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Display user's message in chat UI
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Send message to agent and get response messages (may include tool usage)
        with st.spinner("Thinking..."):
            messages = st.session_state.agent.send_message(prompt)
        
        # Add all new messages (including tool calls) to session state history
        st.session_state.messages.extend(messages)
        
        # Process and display response messages
        for message in messages:
            if isinstance(message, AIMessage) and message.content:
                with st.chat_message("assistant"):
                    st.markdown(get_text(message.content))
            elif isinstance(message, AIMessage) and not message.content and hasattr(message, 'tool_calls') and message.tool_calls:
                with st.chat_message("assistant"):
                    tool_name = message.tool_calls[0]['name']
                    tool_args = str(message.tool_calls[0]['args'])
                    with st.status(f"🔧 Using tool: {tool_name}", expanded=False):
                        st.code(tool_args, language="json")
            elif isinstance(message, ToolMessage):
                with st.chat_message("assistant"):
                    content = message.content
                    # Check if this is a course map path
                    if content.startswith("COURSE_MAP_PATH:"):
                        # Parse the content: COURSE_MAP_PATH:path|major_name
                        parts = content.replace("COURSE_MAP_PATH:", "").strip().split("|")
                        pdf_path = parts[0].strip()
                        major_display = parts[1].strip() if len(parts) > 1 else "the requested major"
                        
                        pdf_file = Path(pdf_path)
                        if pdf_file.exists():
                            # Read PDF file
                            with open(pdf_file, "rb") as pdf_file_obj:
                                pdf_bytes = pdf_file_obj.read()
                            
                            # Display PDF in an embedded viewer using iframe
                            import base64
                            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f'''
                            <iframe src="data:application/pdf;base64,{pdf_base64}" 
                                    width="100%" 
                                    height="600px" 
                                    style="border: 1px solid #ddd; border-radius: 5px;">
                            </iframe>
                            '''
                            st.markdown(pdf_display, unsafe_allow_html=True)
                            
                            # Also provide download button
                            st.download_button(
                                label="📥 Download Course Map PDF",
                                data=pdf_bytes,
                                file_name=pdf_file.name,
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            # If file doesn't exist, show user-friendly message
                            st.info("The course map for this major is not currently available.")
                    else:
                        with st.status("📊 Tool result", expanded=False):
                            if len(content) > 1000:
                                st.markdown(content[:1000] + "\n\n... (truncated)")
                            else:
                                st.markdown(content)
        # Persist updated chat history
        save_current_chat()

        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())

