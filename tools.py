"""
Custom tools for LAUpath AI agent.

This module contains custom tools that enhance the LLM agent's capabilities:
1. Student Profile Analyzer - Analyzes student academic records
2. Major Recommendation Engine - Provides personalized major recommendations
3. Course Map Retriever - Retrieves course map PDFs for specific majors
"""

from langchain.tools import tool
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for RAG access
API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_DIRECTORY = "./vector_db"
COLLECTION_NAME = "lau_documents"
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"


@tool
def analyze_student_profile(
    school_gpa: float,
    lebanese_exam_score: Optional[float] = None,
    sat_score: Optional[int] = None,
    english_proficiency_score: Optional[float] = None,
    english_proficiency_type: Optional[str] = None
) -> str:
    """
    Analyzes a student's academic profile to determine eligibility, remedial requirements, and admission status.
    
    This tool evaluates:
    - Overall eligibility based on grades
    - Need for remedial English courses
    - Admission probability
    - Academic strengths and areas for improvement
    
    Args:
        school_gpa (float): Student's high school GPA (typically 0-20 scale for Lebanese system or 0-4.0 for US system)
        lebanese_exam_score (Optional[float]): Official Lebanese Baccalaureate exam score
        sat_score (Optional[int]): SAT total score (out of 1600)
        english_proficiency_score (Optional[float]): English proficiency test score
        english_proficiency_type (Optional[str]): Type of English test (e.g., "TOEFL", "IELTS", "Duolingo")
    
    Returns:
        str: A detailed analysis of the student's profile including eligibility, recommendations, and requirements
    """
    try:
        analysis = {
            "eligibility": "pending",
            "remedial_english_required": False,
            "admission_probability": "medium",
            "strengths": [],
            "recommendations": [],
            "details": {}
        }
        
        # Analyze GPA
        if school_gpa >= 16.0 or school_gpa >= 3.5:  # High GPA (Lebanese 20-point or US 4.0 scale)
            analysis["eligibility"] = "high"
            analysis["admission_probability"] = "high"
            analysis["strengths"].append("Strong academic performance")
        elif school_gpa >= 14.0 or school_gpa >= 3.0:
            analysis["eligibility"] = "good"
            analysis["admission_probability"] = "medium-high"
            analysis["strengths"].append("Good academic performance")
        elif school_gpa >= 12.0 or school_gpa >= 2.5:
            analysis["eligibility"] = "acceptable"
            analysis["admission_probability"] = "medium"
            analysis["recommendations"].append("Consider strengthening academic performance")
        else:
            analysis["eligibility"] = "low"
            analysis["admission_probability"] = "low"
            analysis["recommendations"].append("Significant improvement needed in academic performance")
        
        analysis["details"]["gpa"] = school_gpa
        
        # Analyze Lebanese exam score
        if lebanese_exam_score is not None:
            if lebanese_exam_score >= 15.0:
                analysis["strengths"].append("Excellent Lebanese Baccalaureate results")
                if analysis["eligibility"] == "pending":
                    analysis["eligibility"] = "high"
            elif lebanese_exam_score >= 12.0:
                analysis["strengths"].append("Good Lebanese Baccalaureate results")
            else:
                analysis["recommendations"].append("Consider retaking Lebanese Baccalaureate exam")
            analysis["details"]["lebanese_exam"] = lebanese_exam_score
        
        # Analyze SAT score
        if sat_score is not None:
            if sat_score >= 1400:
                analysis["strengths"].append("Excellent SAT score")
                if analysis["eligibility"] == "pending":
                    analysis["eligibility"] = "high"
            elif sat_score >= 1200:
                analysis["strengths"].append("Good SAT score")
            elif sat_score >= 1000:
                analysis["recommendations"].append("Consider retaking SAT for better opportunities")
            else:
                analysis["recommendations"].append("SAT score below typical admission standards")
            analysis["details"]["sat"] = sat_score
        
        # Analyze English proficiency
        if english_proficiency_score is not None and english_proficiency_type is not None:
            english_type = english_proficiency_type.upper()
            
            # TOEFL iBT (0-120)
            if english_type == "TOEFL":
                if english_proficiency_score < 80:
                    analysis["remedial_english_required"] = True
                    analysis["recommendations"].append("Remedial English course required (TOEFL < 80)")
                elif english_proficiency_score >= 100:
                    analysis["strengths"].append("Strong English proficiency (TOEFL)")
            
            # IELTS (0-9)
            elif english_type == "IELTS":
                if english_proficiency_score < 6.5:
                    analysis["remedial_english_required"] = True
                    analysis["recommendations"].append("Remedial English course required (IELTS < 6.5)")
                elif english_proficiency_score >= 7.5:
                    analysis["strengths"].append("Strong English proficiency (IELTS)")
            
            # Duolingo (10-160)
            elif english_type == "DUOLINGO":
                if english_proficiency_score < 105:
                    analysis["remedial_english_required"] = True
                    analysis["recommendations"].append("Remedial English course required (Duolingo < 105)")
                elif english_proficiency_score >= 120:
                    analysis["strengths"].append("Strong English proficiency (Duolingo)")
            
            analysis["details"]["english_proficiency"] = {
                "type": english_proficiency_type,
                "score": english_proficiency_score
            }
        else:
            # If no English proficiency score provided, recommend taking a test
            analysis["recommendations"].append("English proficiency test score required for admission")
            analysis["details"]["english_proficiency"] = "not_provided"
        
        # Generate summary
        summary_parts = [f"**Student Profile Analysis**\n\n"]
        summary_parts.append(f"**Eligibility Status:** {analysis['eligibility'].upper()}\n")
        summary_parts.append(f"**Admission Probability:** {analysis['admission_probability'].upper()}\n\n")
        
        if analysis["strengths"]:
            summary_parts.append("**Academic Strengths:**\n")
            for strength in analysis["strengths"]:
                summary_parts.append(f"✓ {strength}\n")
            summary_parts.append("\n")
        
        if analysis["remedial_english_required"]:
            summary_parts.append("⚠️ **Remedial English Required:** Yes\n\n")
        else:
            summary_parts.append("✓ **Remedial English Required:** No\n\n")
        
        if analysis["recommendations"]:
            summary_parts.append("**Recommendations:**\n")
            for rec in analysis["recommendations"]:
                summary_parts.append(f"• {rec}\n")
            summary_parts.append("\n")
        
        summary_parts.append("**Profile Details:**\n")
        for key, value in analysis["details"].items():
            summary_parts.append(f"- {key.replace('_', ' ').title()}: {value}\n")
        
        return "".join(summary_parts)
    
    except Exception as e:
        return f"Error analyzing student profile: {str(e)}"


@tool
def recommend_major(
    interests: str,
    academic_strengths: str,
    career_goals: str,
    preferred_work_environment: str,
    student_profile_data: Optional[str] = None
) -> str:
    """
    Provides personalized major recommendations based on student interests, strengths, and goals.
    
    This tool conducts an assessment and recommends suitable majors by analyzing:
    - Student interests and passions
    - Academic strengths and performance
    - Career aspirations
    - Work environment preferences
    - Academic profile (if available)
    
    Args:
        interests (str): Student's interests, hobbies, and subjects they enjoy (comma-separated or free text)
        academic_strengths (str): Subjects or areas where the student excels (e.g., "Math, Science, Programming")
        career_goals (str): Desired career path or professional goals
        preferred_work_environment (str): Preferred work setting (e.g., "office", "field work", "research lab", "creative studio")
        student_profile_data (Optional[str]): JSON string of student profile data for context
    
    Returns:
        str: Personalized major recommendations with explanations and fit scores
    """
    try:
        # Parse student profile if provided
        profile = {}
        if student_profile_data:
            try:
                profile = json.loads(student_profile_data)
            except:
                pass
        
        # First, get available majors from LAU PDF using RAG
        available_majors_info = ""
        try:
            if API_KEY:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=API_KEY
                )
                vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=VECTOR_DB_DIRECTORY,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                # Search for undergraduate programs
                results = vector_store.similarity_search(
                    query="undergraduate programs majors degrees LAU", 
                    k=10
                )
                available_majors_info = "\n".join([doc.page_content for doc in results])
        except Exception as e:
            # If RAG fails, continue with static database
            pass
        
        # Extract major names from the PDF content
        lau_majors_list = []
        if available_majors_info:
            # Known LAU majors from the PDF (comprehensive list)
            known_majors = [
                "Applied Physics", "Architecture", "Bioinformatics", "Biology", "Business",
                "Business Emphasis Hospitality & Tourism Management", "Chemical Engineering", "Chemistry",
                "Civil Engineering", "Communication", "Computer Engineering", "Computer Science", "Economics",
                "Education", "Electrical Engineering", "English", "Fashion Design", "Graphic Design",
                "Industrial Engineering", "Interior Design", "Mathematics", "Mechanical Engineering",
                "Mechatronics Engineering", "Multimedia Journalism", "Nutrition and Dietetics",
                "Performing Arts", "Petroleum Engineering", "Pharmacy", "Political Science/International Affairs",
                "Psychology", "Studio Art", "Translation", "TV & Film"
            ]
            
            # Check which majors are mentioned in the PDF content
            available_majors_info_lower = available_majors_info.lower()
            for major in known_majors:
                # Check if major name appears in PDF (flexible matching)
                major_keywords = major.lower().split()
                if any(keyword in available_majors_info_lower for keyword in major_keywords if len(keyword) > 3):
                    lau_majors_list.append(major)
        
        # Comprehensive LAU majors database with all programs from PDF
        majors_db = {
            # Engineering Majors
            "Computer Engineering (COE)": {
                "interests": ["programming", "hardware", "embedded systems", "computer architecture", "digital design", "circuits"],
                "strengths": ["math", "physics", "logic", "problem-solving", "analytical thinking", "electronics"],
                "careers": ["computer engineer", "hardware engineer", "embedded systems engineer", "firmware developer"],
                "environment": ["office", "lab", "tech company"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Electrical Engineering (ELE)": {
                "interests": ["electronics", "power systems", "circuits", "signal processing", "electromagnetics", "renewable energy"],
                "strengths": ["math", "physics", "analytical thinking", "problem-solving", "circuit analysis"],
                "careers": ["electrical engineer", "power systems engineer", "electronics engineer", "control systems engineer"],
                "environment": ["office", "field", "lab", "power plant"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Industrial Engineering (INE)": {
                "interests": ["optimization", "process improvement", "manufacturing", "operations research", "quality control", "supply chain"],
                "strengths": ["math", "analytics", "problem-solving", "organization", "statistics"],
                "careers": ["industrial engineer", "operations analyst", "quality engineer", "supply chain manager"],
                "environment": ["office", "factory", "warehouse"],
                "riasec": ["Conventional", "Investigative"]
            },
            "Mechanical Engineering (MCE)": {
                "interests": ["mechanics", "design", "manufacturing", "thermodynamics", "machines", "robotics"],
                "strengths": ["math", "physics", "mechanics", "problem-solving", "design"],
                "careers": ["mechanical engineer", "design engineer", "manufacturing engineer", "robotics engineer"],
                "environment": ["office", "factory", "field", "lab"],
                "riasec": ["Realistic", "Investigative"]
            },
            "Petroleum Engineering (PTE)": {
                "interests": ["oil", "gas", "drilling", "reservoir engineering", "energy", "geology"],
                "strengths": ["math", "physics", "chemistry", "problem-solving", "geology"],
                "careers": ["petroleum engineer", "reservoir engineer", "drilling engineer", "production engineer"],
                "environment": ["field", "office", "oil rig"],
                "riasec": ["Realistic", "Investigative"]
            },
            # Business and Management
            "Business Administration": {
                "interests": ["business", "management", "finance", "marketing", "entrepreneurship", "strategy"],
                "strengths": ["communication", "leadership", "analytics", "strategy", "organization"],
                "careers": ["manager", "entrepreneur", "consultant", "executive", "business analyst"],
                "environment": ["office", "corporate"],
                "riasec": ["Enterprising", "Conventional"]
            },
            # Health Sciences
            "Medicine": {
                "interests": ["health", "biology", "helping people", "science", "anatomy", "diagnosis"],
                "strengths": ["science", "biology", "chemistry", "memorization", "analytical thinking"],
                "careers": ["doctor", "physician", "surgeon", "medical researcher"],
                "environment": ["hospital", "clinic", "field"],
                "riasec": ["Social", "Investigative"]
            },
            "Nursing": {
                "interests": ["healthcare", "helping", "patient care", "biology", "wellness"],
                "strengths": ["science", "empathy", "communication", "biology", "patience"],
                "careers": ["nurse", "healthcare provider", "patient care specialist"],
                "environment": ["hospital", "clinic"],
                "riasec": ["Social", "Conventional"]
            },
            # Social Sciences
            "Psychology": {
                "interests": ["human behavior", "mental health", "counseling", "research", "therapy"],
                "strengths": ["communication", "empathy", "analysis", "writing", "research"],
                "careers": ["psychologist", "therapist", "counselor", "researcher"],
                "environment": ["office", "clinic", "research"],
                "riasec": ["Social", "Investigative"]
            },
            # Arts and Design
            "Architecture": {
                "interests": ["design", "art", "building", "creativity", "space", "urban planning"],
                "strengths": ["math", "art", "visual", "design", "creativity", "spatial thinking"],
                "careers": ["architect", "designer", "urban planner"],
                "environment": ["studio", "field", "office"],
                "riasec": ["Artistic", "Realistic"]
            },
            "Graphic Design": {
                "interests": ["art", "design", "creativity", "visual", "media", "branding"],
                "strengths": ["art", "creativity", "visual", "design", "communication"],
                "careers": ["graphic designer", "artist", "creative director", "brand designer"],
                "environment": ["studio", "office", "creative"],
                "riasec": ["Artistic", "Enterprising"]
            },
            # Communication
            "Journalism": {
                "interests": ["writing", "media", "news", "communication", "storytelling", "investigation"],
                "strengths": ["writing", "communication", "research", "interviewing", "critical thinking"],
                "careers": ["journalist", "reporter", "writer", "media producer"],
                "environment": ["field", "office", "studio"],
                "riasec": ["Artistic", "Enterprising"]
            },
            # Education
            "Education": {
                "interests": ["teaching", "education", "helping", "children", "learning", "curriculum"],
                "strengths": ["communication", "patience", "organization", "subject knowledge", "empathy"],
                "careers": ["teacher", "educator", "professor", "administrator"],
                "environment": ["school", "classroom", "office"],
                "riasec": ["Social", "Conventional"]
            },
            # Additional LAU Majors from PDF
            "Applied Physics": {
                "interests": ["physics", "research", "experiments", "mathematics", "scientific analysis"],
                "strengths": ["math", "physics", "analytical thinking", "problem-solving", "research"],
                "careers": ["physicist", "research scientist", "engineer", "data analyst"],
                "environment": ["lab", "research", "office"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Bioinformatics": {
                "interests": ["biology", "computing", "data analysis", "genetics", "programming", "research"],
                "strengths": ["biology", "programming", "math", "data analysis", "research"],
                "careers": ["bioinformatician", "computational biologist", "data scientist", "researcher"],
                "environment": ["lab", "office", "research"],
                "riasec": ["Investigative", "Conventional"]
            },
            "Biology": {
                "interests": ["biology", "life sciences", "research", "nature", "experiments"],
                "strengths": ["biology", "chemistry", "research", "analytical thinking", "observation"],
                "careers": ["biologist", "researcher", "lab technician", "environmental scientist"],
                "environment": ["lab", "field", "research"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Business": {
                "interests": ["business", "management", "finance", "marketing", "entrepreneurship"],
                "strengths": ["communication", "leadership", "analytics", "strategy"],
                "careers": ["business manager", "entrepreneur", "consultant", "executive"],
                "environment": ["office", "corporate"],
                "riasec": ["Enterprising", "Conventional"]
            },
            "Business Emphasis Hospitality & Tourism Management": {
                "interests": ["hospitality", "tourism", "hotel management", "customer service", "travel"],
                "strengths": ["communication", "organization", "customer service", "leadership"],
                "careers": ["hotel manager", "tourism director", "event planner", "hospitality consultant"],
                "environment": ["hotel", "office", "travel"],
                "riasec": ["Enterprising", "Social"]
            },
            "Chemical Engineering": {
                "interests": ["chemistry", "processes", "manufacturing", "materials", "reactions"],
                "strengths": ["chemistry", "math", "physics", "problem-solving", "analytical thinking"],
                "careers": ["chemical engineer", "process engineer", "materials engineer", "research engineer"],
                "environment": ["lab", "factory", "office"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Chemistry": {
                "interests": ["chemistry", "experiments", "molecules", "reactions", "research"],
                "strengths": ["chemistry", "math", "analytical thinking", "research", "lab skills"],
                "careers": ["chemist", "researcher", "lab technician", "pharmaceutical scientist"],
                "environment": ["lab", "research", "office"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Civil Engineering": {
                "interests": ["construction", "infrastructure", "design", "building", "structures"],
                "strengths": ["math", "physics", "design", "problem-solving", "project management"],
                "careers": ["civil engineer", "structural engineer", "construction manager", "project engineer"],
                "environment": ["field", "office", "construction site"],
                "riasec": ["Realistic", "Investigative"]
            },
            "Communication": {
                "interests": ["communication", "media", "public relations", "advertising", "marketing"],
                "strengths": ["communication", "writing", "creativity", "presentation", "interpersonal"],
                "careers": ["communications specialist", "public relations manager", "marketing coordinator", "media planner"],
                "environment": ["office", "studio", "field"],
                "riasec": ["Enterprising", "Artistic"]
            },
            "Computer Science": {
                "interests": ["programming", "algorithms", "software", "computing", "technology"],
                "strengths": ["programming", "math", "logic", "problem-solving", "analytical thinking"],
                "careers": ["software developer", "data scientist", "systems analyst", "programmer"],
                "environment": ["office", "remote", "tech company"],
                "riasec": ["Investigative", "Conventional"]
            },
            "Economics": {
                "interests": ["economics", "finance", "markets", "policy", "analysis"],
                "strengths": ["math", "analytics", "critical thinking", "research", "statistics"],
                "careers": ["economist", "financial analyst", "policy analyst", "researcher"],
                "environment": ["office", "research", "government"],
                "riasec": ["Investigative", "Enterprising"]
            },
            "English": {
                "interests": ["literature", "writing", "language", "reading", "analysis"],
                "strengths": ["writing", "communication", "critical thinking", "analysis", "creativity"],
                "careers": ["writer", "editor", "teacher", "content creator", "researcher"],
                "environment": ["office", "classroom", "library"],
                "riasec": ["Artistic", "Social"]
            },
            "Fashion Design": {
                "interests": ["fashion", "design", "clothing", "creativity", "style", "textiles"],
                "strengths": ["creativity", "design", "visual", "artistic", "fashion sense"],
                "careers": ["fashion designer", "stylist", "fashion buyer", "costume designer"],
                "environment": ["studio", "office", "fashion house"],
                "riasec": ["Artistic", "Enterprising"]
            },
            "Interior Design": {
                "interests": ["interior design", "space", "decoration", "architecture", "creativity"],
                "strengths": ["design", "creativity", "spatial thinking", "visual", "artistic"],
                "careers": ["interior designer", "space planner", "decorator", "design consultant"],
                "environment": ["studio", "field", "office"],
                "riasec": ["Artistic", "Realistic"]
            },
            "Mathematics": {
                "interests": ["mathematics", "problem-solving", "logic", "analysis", "theoretical concepts"],
                "strengths": ["math", "logic", "analytical thinking", "problem-solving", "abstract thinking"],
                "careers": ["mathematician", "actuary", "statistician", "data analyst", "researcher"],
                "environment": ["office", "research", "academic"],
                "riasec": ["Investigative", "Conventional"]
            },
            "Mechatronics Engineering": {
                "interests": ["robotics", "automation", "mechanical systems", "electronics", "control systems"],
                "strengths": ["math", "physics", "electronics", "mechanics", "programming"],
                "careers": ["mechatronics engineer", "robotics engineer", "automation engineer", "control systems engineer"],
                "environment": ["lab", "factory", "office"],
                "riasec": ["Investigative", "Realistic"]
            },
            "Multimedia Journalism": {
                "interests": ["journalism", "multimedia", "video", "digital media", "storytelling", "news"],
                "strengths": ["writing", "communication", "multimedia", "creativity", "technology"],
                "careers": ["multimedia journalist", "video producer", "digital reporter", "content creator"],
                "environment": ["studio", "field", "office"],
                "riasec": ["Artistic", "Enterprising"]
            },
            "Nutrition and Dietetics": {
                "interests": ["nutrition", "health", "food science", "wellness", "diet planning"],
                "strengths": ["science", "biology", "chemistry", "communication", "organization"],
                "careers": ["nutritionist", "dietitian", "health educator", "food scientist"],
                "environment": ["hospital", "clinic", "office", "research"],
                "riasec": ["Social", "Investigative"]
            },
            "Performing Arts": {
                "interests": ["theater", "acting", "dance", "music", "performance", "entertainment"],
                "strengths": ["creativity", "performance", "expression", "communication", "artistic"],
                "careers": ["actor", "performer", "director", "entertainer", "theater professional"],
                "environment": ["theater", "studio", "stage"],
                "riasec": ["Artistic", "Enterprising"]
            },
            "Pharmacy": {
                "interests": ["pharmacy", "medicine", "drugs", "health", "patient care"],
                "strengths": ["chemistry", "biology", "math", "memorization", "attention to detail"],
                "careers": ["pharmacist", "pharmaceutical researcher", "clinical pharmacist", "pharmacy manager"],
                "environment": ["pharmacy", "hospital", "clinic", "research"],
                "riasec": ["Investigative", "Conventional"]
            },
            "Political Science/International Affairs": {
                "interests": ["politics", "international relations", "government", "policy", "diplomacy"],
                "strengths": ["critical thinking", "writing", "research", "communication", "analysis"],
                "careers": ["diplomat", "policy analyst", "political consultant", "researcher", "government official"],
                "environment": ["office", "government", "international"],
                "riasec": ["Enterprising", "Investigative"]
            },
            "Studio Art": {
                "interests": ["art", "painting", "sculpture", "visual arts", "creativity", "exhibition"],
                "strengths": ["art", "creativity", "visual", "artistic", "expression"],
                "careers": ["artist", "art teacher", "curator", "art director", "gallery manager"],
                "environment": ["studio", "gallery", "office"],
                "riasec": ["Artistic", "Enterprising"]
            },
            "Translation": {
                "interests": ["languages", "translation", "linguistics", "communication", "cultural exchange"],
                "strengths": ["languages", "writing", "communication", "cultural knowledge", "attention to detail"],
                "careers": ["translator", "interpreter", "linguist", "language specialist"],
                "environment": ["office", "field", "international"],
                "riasec": ["Conventional", "Artistic"]
            },
            "TV & Film": {
                "interests": ["television", "film", "video production", "cinematography", "directing", "media"],
                "strengths": ["creativity", "visual", "technical", "storytelling", "communication"],
                "careers": ["filmmaker", "director", "producer", "cinematographer", "video editor"],
                "environment": ["studio", "field", "production"],
                "riasec": ["Artistic", "Enterprising"]
            }
        }
        
        # Filter majors to only include those found in PDF if RAG was successful
        if lau_majors_list:
            # Create a filtered database with only PDF majors
            filtered_majors_db = {}
            for major_name, major_data in majors_db.items():
                # Check if major name matches any in PDF
                major_base = major_name.split("(")[0].strip() if "(" in major_name else major_name
                # Try to match with PDF majors
                matched = False
                for pdf_major in lau_majors_list:
                    pdf_base = pdf_major.split("/")[0].strip() if "/" in pdf_major else pdf_major
                    if (major_base.lower() in pdf_base.lower() or 
                        pdf_base.lower() in major_base.lower() or
                        major_base.lower() == pdf_base.lower()):
                        matched = True
                        break
                if matched:
                    filtered_majors_db[major_name] = major_data
            # If we found matches, use filtered; otherwise use full database
            if filtered_majors_db:
                majors_db = filtered_majors_db
        
        # Normalize inputs for matching
        interests_lower = interests.lower()
        strengths_lower = academic_strengths.lower()
        goals_lower = career_goals.lower()
        env_lower = preferred_work_environment.lower()
        
        # Score each major
        major_scores = {}
        for major, criteria in majors_db.items():
            score = 0
            
            # Check interests match
            for interest_keyword in criteria["interests"]:
                if interest_keyword in interests_lower:
                    score += 2
            
            # Check strengths match
            for strength_keyword in criteria["strengths"]:
                if strength_keyword in strengths_lower:
                    score += 2
            
            # Check career goals match
            for career_keyword in criteria["careers"]:
                if career_keyword in goals_lower:
                    score += 3
            
            # Check environment match
            for env_keyword in criteria["environment"]:
                if env_keyword in env_lower:
                    score += 1
            
            major_scores[major] = score
        
        # Sort majors by score
        sorted_majors = sorted(major_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations - only include majors with score > 0
        valid_majors = [(m, s) for m, s in sorted_majors if s > 0]
        
        if not valid_majors:
            return "Based on your responses, I couldn't find a strong match with LAU majors. I recommend exploring LAU's programs using the search function or speaking with an academic advisor to discover majors that align with your interests."
        
        recommendations = ["**Top 3 Major Recommendations at LAU**\n\n"]
        
        # Top 3 recommendations (only specific LAU majors)
        top_majors = valid_majors[:3]
        max_score = max([s for _, s in valid_majors]) if valid_majors else 1
        
        for i, (major, score) in enumerate(top_majors, 1):
            # Calculate fit percentage
            fit_percentage = int((score / max_score) * 100) if max_score > 0 else 0
            fit_level = "Excellent" if fit_percentage >= 80 else "Good" if fit_percentage >= 60 else "Moderate"
            
            recommendations.append(f"**{i}. {major}**\n")
            recommendations.append(f"   Fit Level: {fit_level} ({fit_percentage}% match)\n")
            
            # Explain why with specific details
            reasons = []
            if any(kw in interests_lower for kw in majors_db[major]["interests"]):
                reasons.append("aligns with your interests")
            if any(kw in strengths_lower for kw in majors_db[major]["strengths"]):
                reasons.append("matches your academic strengths")
            if any(kw in goals_lower for kw in majors_db[major]["careers"]):
                reasons.append("supports your career goals")
            if any(kw in env_lower for kw in majors_db[major]["environment"]):
                reasons.append("matches your preferred work environment")
            
            if reasons:
                recommendations.append(f"   Why: This major {', '.join(reasons)}.\n")
            
            # Add career paths
            careers = majors_db[major]["careers"]
            if careers:
                recommendations.append(f"   Career Paths: {', '.join(careers[:3])}\n")
            
            recommendations.append("\n")
        
        # Additional considerations (only if there are more valid majors)
        if len(valid_majors) > 3:
            recommendations.append("**Other LAU Majors to Consider:**\n")
            for major, score in valid_majors[3:6]:
                recommendations.append(f"• {major}\n")
        
        recommendations.append("\n**Next Steps:**\n")
        recommendations.append("1. Research the recommended majors in detail\n")
        recommendations.append("2. Use the course map tool to see the curriculum for each major\n")
        recommendations.append("3. Speak with LAU academic advisors about these programs\n")
        recommendations.append("4. Review LAU's admission requirements for your chosen major\n")
        
        return "".join(recommendations)
    
    except Exception as e:
        return f"Error generating major recommendations: {str(e)}"


@tool
def get_course_map(major_name: str) -> str:
    """
    Retrieves the course map (curriculum) PDF for a specific major.
    
    This tool maps major names to their corresponding course map PDF files.
    When a student asks about a curriculum or course map, this tool finds and returns
    the path to the appropriate PDF file.
    
    Args:
        major_name (str): The name of the major (e.g., "Computer Engineering", "COE", 
                         "Civil Engineering", "CIE", "Electrical Engineering", etc.)
    
    Returns:
        str: The file path to the course map PDF, or an error message if not found.
             The path will be in format: "COURSE_MAP_PATH:data/CourseMaps/[major]_courseMap.pdf"
    """
    try:
        # Course maps directory
        course_maps_dir = Path("./data/CourseMaps")
        
        if not course_maps_dir.exists():
            return "Error: Course maps directory not found. Please ensure data/CourseMaps folder exists."
        
        # Normalize major name for matching
        major_lower = major_name.lower().strip()
        
        # Mapping of major names/abbreviations to file prefixes
        # This maps various ways students might refer to majors to the actual file names
        major_mappings = {
            # Computer Engineering variations
            "computer engineering": "COE",
            "coe": "COE",
            "comp eng": "COE",
            "computer eng": "COE",
            
            # Civil Engineering variations
            "civil engineering": "CIE",
            "cie": "CIE",
            "civil eng": "CIE",
            "civ eng": "CIE",
            
            # Electrical Engineering variations
            "electrical engineering": "ELE",
            "ele": "ELE",
            "electrical eng": "ELE",
            "elec eng": "ELE",
            
            # Industrial Engineering variations
            "industrial engineering": "INE",
            "ine": "INE",
            "industrial eng": "INE",
            "ind eng": "INE",
            
            # Mechanical Engineering variations
            "mechanical engineering": "MCE",
            "mce": "MCE",
            "mechanical eng": "MCE",
            "mech eng": "MCE",
            
            # Petroleum Engineering variations (if PTE is petroleum)
            "petroleum engineering": "PTE",
            "pte": "PTE",
            "petroleum eng": "PTE",
            "pet eng": "PTE",
            
            # Additional variations that might be used
            "meee": "MEE",  # If MEE is a separate major
            "mee": "MEE",
        }
        
        # Find matching major
        matched_prefix = None
        for key, prefix in major_mappings.items():
            if key in major_lower:
                matched_prefix = prefix
                break
        
        if not matched_prefix:
            # Try direct match with uppercase
            matched_prefix = major_name.upper().strip()
        
        # Look for the course map file
        # Try different file name patterns
        possible_files = [
            f"{matched_prefix}_courseMap.pdf",
            f"{matched_prefix}_CourseMap.pdf",
            f"{matched_prefix}_CourseMap.pdf.pdf",  # Handle double extension
            f"{matched_prefix}_coursemap.pdf",
        ]
        
        found_file = None
        for filename in possible_files:
            file_path = course_maps_dir / filename
            if file_path.exists():
                found_file = file_path
                break
        
        # If not found with prefix, search all files
        if not found_file:
            all_files = list(course_maps_dir.glob("*.pdf"))
            # Try fuzzy matching on file names
            for file_path in all_files:
                file_stem = file_path.stem.upper()
                if matched_prefix in file_stem or file_stem.startswith(matched_prefix):
                    found_file = file_path
                    break
        
        if found_file:
            # Return path in a format that can be easily parsed by the UI
            # Use relative path for portability
            # Format: COURSE_MAP_PATH:path|major_display_name
            relative_path = str(found_file).replace("\\", "/")
            # Get a clean major name for display
            major_display = matched_prefix if matched_prefix else major_name
            return f"COURSE_MAP_PATH:{relative_path}|{major_display}"
        else:
            # Return user-friendly error message without exposing paths
            return f"The course map for '{major_name}' is not currently available."
    
    except Exception as e:
        return f"Unable to retrieve the course map at this time. Please try again later."

