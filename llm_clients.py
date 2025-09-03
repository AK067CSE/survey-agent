# llm_clients.py

import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from groq import Groq

# --- Initialize API Clients ---

# It's best practice to initialize clients once
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def get_gemini_client():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# --- Wrapper Functions for API Calls ---

def call_openai(prompt, model="gpt-4o-mini"):
    """Calls the OpenAI API."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI: {e}"

def call_gemini(prompt):
    """Calls the Google Gemini API."""
    try:
        model = get_gemini_client()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"

def call_groq(prompt, model="llama-3.3-70b-versatile"):
    """Calls the Groq API for fast responses."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq: {e}"
