import streamlit as st
from transformers import pipeline
import time

# Set page configuration for a professional look
st.set_page_config(page_title="GrowTech", page_icon="💻", layout="wide")

# Custom CSS for gradient background and styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }
    .stTextInput > div > input, .stTextArea > div > textarea {
        background-color: #ffffff;
        color: #333;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #ff6f61;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e55a50;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load the Hugging Face model (DistilGPT2)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

model = load_model()

# Base prompt for assistant
base_prompt = """You are GrowTech, a friendly AI tutor for technical skills. Provide clear, beginner-friendly answers with Python code examples. Break explanations into steps, avoid jargon, and suggest next steps. For debugging, identify errors and suggest fixes. Respond to: """

# Sidebar
with st.sidebar:
    st.header("🚀 GrowTech Options")
    skill_level = st.selectbox("Your Skill Level", ["Beginner", "Intermediate", "Advanced"], help="Select your experience level")
    st.markdown("---")
    st.write("*Example Queries:*")
    st.write("- Explain Python loops")
    st.write("- Debug my code: print(x[5])")
    st.write("- Create a 4-week Python learning plan")
    st.markdown("---")
    st.markdown("*About GrowTech*")
    st.write("A free, open-source app to master coding skills! 🎓")

# Main content
st.title("GrowTech: Your Technical Skills Mentor 🌟")
st.markdown("Ask coding questions, debug code, or get a personalized learning path to level up your skills!")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Coding Q&A", "🐞 Debug Code", "📚 Learning Path"])

# Tab 1: Q&A
with tab1:
    st.header("Ask a Coding Question")
    query = st.text_input("Enter your question (e.g., 'Explain Python loops'):", key="qa_input")
    if st.button("Get Answer", key="qa_button"):
        if query:
            with st.spinner("GrowTech is thinking... 🤔"):
                full_prompt = base_prompt + query
                response = model(full_prompt, max_length=300, num_return_sequences=1)[0]["generated_text"]
                st.markdown("### Answer")
                st.write(response)
                st.success("Done! Try another question or check other tabs! 🚀")
        else:
            st.error("Please enter a question!")

# Tab 2: Debugging
with tab2:
    st.header("Debug Your Code")
    code_input = st.text_area("Paste your code here:", key="debug_input", height=150)
    if st.button("Debug Code", key="debug_button"):
        if code_input:
            with st.spinner("Analyzing your code... 🔍"):
                prompt = f"Debug this code and explain errors in simple terms: {code_input}"
                response = model(prompt, max_length=300)[0]["generated_text"]
                st.markdown("### Debug Report")
                st.write(response)
                st.success("Debug complete! Fix your code and test again! 🛠")
        else:
            st.error("Please paste some code to debug!")

# Tab 3: Learning Path
with tab3:
    st.header("Get a Personalized Learning Path")
    goal = st.text_input("Enter your goal (e.g., 'Learn Python'):", key="path_input")
    if st.button("Generate Path", key="path_button"):
        if goal:
            with st.spinner("Crafting your learning path... 📝"):
                prompt = f"Suggest a 4-week learning path for a {skill_level.lower()} to achieve: {goal}"
                response = model(prompt, max_length=300)[0]["generated_text"]
                st.markdown("### Your Learning Path")
                st.write(response)
                st.success("Path generated! Start learning and share feedback! 🎉")
        else:
            st.error("Please enter a learning goal!")

# Feedback section
with st.expander("📣 Share Your Feedback"):
    with st.form("feedback_form"):
        feedback = st.text_area("Tell us how we can improve GrowTech!", height=100)
        submitted = st.form_submit_button("Submit Feedback")
        if submitted and feedback:
            st.success("Thanks for your feedback! We'll use it to make GrowTech better! 😊")

# Footer
st.markdown("---")
st.markdown("*GrowTech* - Built by TechBit Innovators | Open-source on [code.swecha.org](https://code.swecha.org) | Powered by Hugging Face 🤗")
