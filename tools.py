"""
Custom tools for LAUpath AI agent.

This module contains custom tools that enhance the LLM agent's capabilities:
1. Student Profile Analyzer - Analyzes student academic records
2. Major Recommendation Engine - Provides personalized major recommendations
"""

from langchain.tools import tool
from typing import Dict, List, Optional, Any
import json
import os


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
        
        # Major database with characteristics
        majors_db = {
            "Computer Science": {
                "interests": ["programming", "technology", "computers", "software", "coding", "algorithms"],
                "strengths": ["math", "logic", "problem-solving", "analytical thinking"],
                "careers": ["software engineer", "developer", "data scientist", "tech"],
                "environment": ["office", "remote", "tech company"]
            },
            "Engineering (Various)": {
                "interests": ["building", "design", "physics", "mechanics", "innovation"],
                "strengths": ["math", "science", "physics", "problem-solving"],
                "careers": ["engineer", "design", "construction", "manufacturing"],
                "environment": ["field", "office", "lab"]
            },
            "Business Administration": {
                "interests": ["business", "management", "finance", "marketing", "entrepreneurship"],
                "strengths": ["communication", "leadership", "analytics", "strategy"],
                "careers": ["manager", "entrepreneur", "consultant", "executive"],
                "environment": ["office", "corporate"]
            },
            "Medicine": {
                "interests": ["health", "biology", "helping people", "science", "anatomy"],
                "strengths": ["science", "biology", "chemistry", "memorization"],
                "careers": ["doctor", "physician", "surgeon", "medical"],
                "environment": ["hospital", "clinic", "field"]
            },
            "Nursing": {
                "interests": ["healthcare", "helping", "patient care", "biology"],
                "strengths": ["science", "empathy", "communication", "biology"],
                "careers": ["nurse", "healthcare", "patient care"],
                "environment": ["hospital", "clinic"]
            },
            "Psychology": {
                "interests": ["human behavior", "mental health", "counseling", "research"],
                "strengths": ["communication", "empathy", "analysis", "writing"],
                "careers": ["psychologist", "therapist", "counselor", "researcher"],
                "environment": ["office", "clinic", "research"]
            },
            "Architecture": {
                "interests": ["design", "art", "building", "creativity", "space"],
                "strengths": ["math", "art", "visual", "design", "creativity"],
                "careers": ["architect", "designer", "urban planning"],
                "environment": ["studio", "field", "office"]
            },
            "Graphic Design": {
                "interests": ["art", "design", "creativity", "visual", "media"],
                "strengths": ["art", "creativity", "visual", "design"],
                "careers": ["designer", "artist", "creative", "media"],
                "environment": ["studio", "office", "creative"]
            },
            "Journalism": {
                "interests": ["writing", "media", "news", "communication", "storytelling"],
                "strengths": ["writing", "communication", "research", "interviewing"],
                "careers": ["journalist", "reporter", "writer", "media"],
                "environment": ["field", "office", "studio"]
            },
            "Education": {
                "interests": ["teaching", "education", "helping", "children", "learning"],
                "strengths": ["communication", "patience", "organization", "subject knowledge"],
                "careers": ["teacher", "educator", "professor", "administrator"],
                "environment": ["school", "classroom", "office"]
            }
        }
        
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
        
        # Generate recommendations
        recommendations = ["**Major Recommendations Based on Your Profile**\n\n"]
        
        # Top 3 recommendations
        top_majors = sorted_majors[:3]
        
        for i, (major, score) in enumerate(top_majors, 1):
            fit_level = "Excellent" if score >= 8 else "Good" if score >= 5 else "Moderate"
            recommendations.append(f"**{i}. {major}** (Fit: {fit_level})\n")
            recommendations.append(f"   Match Score: {score}/10\n")
            
            # Explain why
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
            recommendations.append("\n")
        
        # Additional considerations
        if len(sorted_majors) > 3:
            recommendations.append("**Other Majors to Consider:**\n")
            for major, score in sorted_majors[3:6]:
                if score > 0:
                    recommendations.append(f"• {major} (Score: {score})\n")
        
        recommendations.append("\n**Next Steps:**\n")
        recommendations.append("1. Research the recommended majors in detail\n")
        recommendations.append("2. Speak with academic advisors about these programs\n")
        recommendations.append("3. Consider taking introductory courses to explore your interests\n")
        recommendations.append("4. Review LAU's program requirements using the RAG system\n")
        
        return "".join(recommendations)
    
    except Exception as e:
        return f"Error generating major recommendations: {str(e)}"

