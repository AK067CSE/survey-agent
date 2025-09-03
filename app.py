# app.py

import streamlit as st
from agents import orchestrator_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Survey Generator",
    page_icon="🤖",
    layout="wide"
)

# --- Page Title and Introduction ---
st.title("🤖 Multi-Agent Survey Generator")
st.markdown("Describe the topic for your survey, and our team of AI agents will research, design, and compile a comprehensive survey for you.")

# --- Session State for History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("e.g., A survey about employee satisfaction with a new remote work policy"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Agent Processing ---
    with st.chat_message("assistant"):
        with st.spinner("🤖 The AI agents are collaborating... Please wait."):
            try:
                # Call the orchestrator
                response_data = orchestrator_agent(prompt, st.session_state.messages)
                
                # --- Display Results ---
                st.subheader("📝 Here is your generated survey:")
                st.markdown(response_data["final_survey"])

                with st.expander("🔍 View Research Summary"):
                    st.markdown(response_data["research_summary"])
                
                st.subheader("💡 Follow-up Recommendations")
                st.markdown(response_data["recommendations"])

                # Add the full response to session state for context
                full_response_md = f"""
                ### Survey on '{prompt}'
                {response_data['final_survey']}
                ---
                ### Follow-up Recommendations
                {response_data['recommendations']}
                """
                st.session_state.messages.append({"role": "assistant", "content": full_response_md})

                # --- Download Button ---
                st.download_button(
                    label="⬇️ Download Survey as Markdown",
                    data=response_data["final_survey"],
                    file_name=f"survey_{prompt.replace(' ', '_')[:20]}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
