"""
LAUpath AI - Main Application

A specialized LLM agent to help undergraduate students navigate LAU (Lebanese American University).
Features include RAG-based information retrieval, student profile analysis, and major recommendations.
"""

import asyncio
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool

# Import custom tools
from tools import analyze_student_profile, recommend_major

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_DIRECTORY = "./vector_db"
COLLECTION_NAME = "lau_documents"
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
MODEL_NAME = "gemini-2.5-flash"


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
            tools=[search_vector_db, analyze_student_profile, recommend_major]
        )
        
        # Initialize conversation history with system message
        self.messages = [SystemMessage(content="""
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

Always be helpful, accurate, and encouraging. When you don't have specific information, use the search_vector_db tool to find it.
When students provide their academic information, use the analyze_student_profile tool to give them detailed feedback.
When students need guidance on choosing a major, use the recommend_major tool to provide personalized recommendations.
</purpose>

<guidelines>
- Always be respectful and supportive
- Use tools when appropriate to provide accurate, up-to-date information
- Explain your reasoning when making recommendations
- Encourage students to explore their options
- Be clear about requirements and deadlines
- If information is not available, admit it and suggest alternatives
</guidelines>
""")]
    
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
        layout="wide"
    )
    
    # Title and description
    st.title("🎓 LAUpath AI")
    st.markdown("**Your AI guide to Lebanese American University**")
    st.markdown("Get personalized guidance on admissions, programs, majors, and more!")
    
    # Sidebar for student profile
    with st.sidebar:
        st.header("📋 Student Profile")
        
        # Check if vector database exists
        if not os.path.exists(VECTOR_DB_DIRECTORY):
            st.error("⚠️ Vector database not found!")
            st.info("Please run `python setup_rag.py` first to set up the RAG system.")
            st.stop()
        
        # Student profile form
        with st.expander("📝 Enter Your Academic Information", expanded=False):
            with st.form("student_profile_form"):
                st.subheader("Academic Information")
                
                school_gpa = st.number_input(
                    "High School GPA",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.1,
                    help="Enter your high school GPA (Lebanese system: 0-20, US system: 0-4.0)"
                )
                
                lebanese_exam = st.number_input(
                    "Lebanese Baccalaureate Score (Optional)",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.1
                )
                
                sat_score = st.number_input(
                    "SAT Score (Optional)",
                    min_value=0,
                    max_value=1600,
                    value=0,
                    step=10
                )
                
                st.subheader("English Proficiency")
                english_test_type = st.selectbox(
                    "English Test Type",
                    ["None", "TOEFL", "IELTS", "Duolingo"],
                    help="Select the type of English proficiency test you took"
                )
                
                english_score = st.number_input(
                    "English Proficiency Score",
                    min_value=0.0,
                    max_value=200.0,
                    value=0.0,
                    step=0.1,
                    help="Enter your English proficiency test score"
                )
                
                submitted = st.form_submit_button("💾 Save Profile")
                
                if submitted:
                    profile = {
                        "school_gpa": school_gpa if school_gpa > 0 else None,
                        "lebanese_exam_score": lebanese_exam if lebanese_exam > 0 else None,
                        "sat_score": sat_score if sat_score > 0 else None,
                        "english_proficiency_type": english_test_type if english_test_type != "None" else None,
                        "english_proficiency_score": english_score if english_score > 0 else None
                    }
                    save_student_profile(profile)
                    st.success("Profile saved! You can now ask me to analyze it.")
        
        # Display saved profile
        profile = load_student_profile()
        if profile:
            st.success("✓ Profile loaded")
            with st.expander("View Profile"):
                for key, value in profile.items():
                    if value:
                        st.text(f"{key.replace('_', ' ').title()}: {value}")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔍 Analyze My Profile"):
            if profile:
                # This will be handled in the chat
                st.session_state.quick_action = "analyze_profile"
            else:
                st.warning("Please enter your academic information first.")
        
        if st.button("🎯 Get Major Recommendations"):
            st.session_state.quick_action = "recommend_major"
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask me about LAU programs, admissions, or requirements
        - Upload your grades to get personalized analysis
        - Get major recommendations based on your interests
        - I can search LAU documents for specific information
        """)


    # Initialize agent if not already in session state
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = LAUpathAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()
    
    # Initialize message history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Handle quick actions
    if "quick_action" in st.session_state:
        action = st.session_state.quick_action
        del st.session_state.quick_action
        
        if action == "analyze_profile" and profile:
            # Format profile data for the tool
            profile_str = f"""Please analyze my student profile using the analyze_student_profile tool with the following data:
- School GPA: {profile.get('school_gpa')}
- Lebanese Exam Score: {profile.get('lebanese_exam_score')}
- SAT Score: {profile.get('sat_score')}
- English Proficiency Type: {profile.get('english_proficiency_type')}
- English Proficiency Score: {profile.get('english_proficiency_score')}

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
                with st.status("📊 Tool result", expanded=False):
                    # Truncate very long results for display
                    content = message.content
                    if len(content) > 1000:
                        st.markdown(content[:1000] + "\n\n... (truncated, full result used by agent)")
                    else:
                        st.markdown(content)
        
        # Handle user message
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
    
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
                    with st.status("📊 Tool result", expanded=False):
                        content = message.content
                        if len(content) > 1000:
                            st.markdown(content[:1000] + "\n\n... (truncated)")
                        else:
                            st.markdown(content)
        
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())

