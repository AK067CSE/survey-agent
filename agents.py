# agents.py

import llm_clients
import tools
import json

def research_agent(topic: str) -> str:
    """
    Takes a topic, searches the web, and returns a summary of the findings.
    """
    print("--- RESEARCH AGENT ACTIVATED ---")
    prompt = f"You are an expert researcher. Your goal is to provide a concise summary of key points, concepts, and common questions related to the topic: '{topic}'. This summary will be used by other AI agents to generate a survey. Focus on actionable insights."
    
    search_results = tools.simple_web_search(f"key aspects and questions for a survey about {topic}")
    
    # Use a fast model to summarize the search results
    summary_prompt = f"Based on the following search results, create a summary of key points for creating a survey about '{topic}'.\n\nSearch Results:\n{search_results}\n\nSummary:"
    summary = llm_clients.call_groq(summary_prompt, model="llama3-70b-8192")
    return summary

def creative_question_agent(topic: str, research_summary: str) -> str:
    """
    Uses OpenAI to generate creative, open-ended questions.
    """
    print("--- CREATIVE (OPENAI) AGENT ACTIVATED ---")
    prompt = f"""
    You are a world-class survey designer specializing in qualitative feedback. 
    Based on the topic '{topic}' and the following research summary, generate 5-7 insightful, open-ended questions.
    These questions should encourage detailed, thoughtful responses. Do not generate multiple-choice questions.

    Research Summary:
    {research_summary}

    Generate the questions as a JSON list of strings. For example: ["Question 1?", "Question 2?"]
    """
    response = llm_clients.call_openai(prompt)
    return response

def structured_question_agent(topic: str, research_summary: str) -> str:
    """
    Uses Gemini to generate structured questions (Multiple Choice, Likert Scale).
    """
    print("--- STRUCTURED (GEMINI) AGENT ACTIVATED ---")
    prompt = f"""
    You are a survey methodologist specializing in quantitative data.
    Based on the topic '{topic}' and the following research summary, generate 5-7 structured questions.
    Include a mix of multiple-choice and 5-point Likert scale (Strongly Disagree to Strongly Agree) questions.

    Research Summary:
    {research_summary}

    Format the output as a JSON list of objects. Each object must have a 'type' ('multiple_choice' or 'likert') and a 'question' string. For multiple_choice, also include an 'options' list.
    Example:
    [
        {{"type": "multiple_choice", "question": "What is your primary role?", "options": ["Engineer", "Manager", "Designer"]}},
        {{"type": "likert", "question": "The new policy is easy to understand."}}
    ]
    """
    response = llm_clients.call_gemini(prompt)
    return response

def compiler_agent(topic: str, open_ended_qs: str, structured_qs: str) -> str:
    """
    Uses Groq to compile, format, and de-duplicate all generated questions.
    """
    print("--- COMPILER (GROQ) AGENT ACTIVATED ---")
    prompt = f"""
    You are an expert survey editor. Your job is to combine and format the following sets of questions into a single, cohesive survey about '{topic}'.
    Remove any duplicate or very similar questions. Organize them into logical sections.
    Present the final survey in a clean, readable Markdown format.

    Open-Ended Questions (from Creative Agent):
    {open_ended_qs}

    Structured Questions (from Structured Agent):
    {structured_qs}

    Compile them into a final survey. Use Markdown for formatting (e.g., ## Section Title, 1., a.).
    """
    response = llm_clients.call_groq(prompt, model="llama3-70b-8192")
    return response

def insight_agent(conversation_history: list) -> str:
    """
    Uses Groq to provide follow-up recommendations.
    """
    print("--- INSIGHT (GROQ) AGENT ACTIVATED ---")
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    prompt = f"""
    You are a strategic research consultant. Based on the conversation history below, provide 2-3 follow-up recommendations.
    These could be suggestions for related survey topics, different audiences to survey, or how to analyze the potential results. Keep it brief and actionable.

    Conversation History:
    {history_str}

    Recommendations:
    """
    response = llm_clients.call_groq(prompt)
    return response

def orchestrator_agent(user_prompt: str, history: list) -> dict:
    """
    The main orchestrator that manages the entire workflow.
    """
    print("--- ORCHESTRATOR ACTIVATED ---")
    
    # 1. Research Agent
    research_summary = research_agent(user_prompt)
    
    # 2. Parallel Generation Agents
    open_ended_json = creative_question_agent(user_prompt, research_summary)
    structured_json = structured_question_agent(user_prompt, research_summary)

    # 3. Compiler Agent
    final_survey = compiler_agent(user_prompt, open_ended_json, structured_json)

    # 4. Insight Agent
    current_conversation = history + [
        {"role": "assistant", "content": f"Here is the survey about {user_prompt}:\n{final_survey}"}
    ]
    recommendations = insight_agent(current_conversation)

    return {
        "research_summary": research_summary,
        "final_survey": final_survey,
        "recommendations": recommendations
    }
