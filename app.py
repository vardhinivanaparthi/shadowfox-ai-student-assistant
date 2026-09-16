import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# App title
st.title("🎓 ShadowFox AI Student Assistant")

st.write("Your simple AI-powered study assistant.")

# Feature selection
feature = st.selectbox(
    "Choose a feature:",
    [
        "Note Summarizer",
        "Quiz Generator",
        "Concept Explainer"
    ]
)

# Student input
notes = st.text_area(
    "📝 Enter your notes or topic:",
    height=250,
    placeholder="Enter your study notes or a topic here..."
)

# ---------------- NOTE SUMMARIZER ----------------

if feature == "Note Summarizer":

    if st.button("✨ Summarize Notes"):

        if not notes.strip():
            st.warning("Please enter some notes first.")

        else:
            with st.spinner("AI is generating your summary..."):

                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful AI study assistant. "
                                    "Summarize student notes clearly using "
                                    "simple language and bullet points."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Summarize these study notes:\n\n{notes}"
                            }
                        ]
                    )

                    summary = response.choices[0].message.content

                    st.subheader("📚 AI Summary")
                    st.write(summary)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# ---------------- QUIZ GENERATOR ----------------

elif feature == "Quiz Generator":

    if st.button("❓ Generate Quiz"):

        if not notes.strip():
            st.warning("Please enter some notes first.")

        else:
            with st.spinner("AI is creating your quiz..."):

                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful AI study assistant. "
                                    "Create a simple quiz from the student's notes. "
                                    "Generate 5 questions with multiple-choice options "
                                    "and provide the correct answers at the end."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Create a quiz from these study notes:\n\n{notes}"
                            }
                        ]
                    )

                    quiz = response.choices[0].message.content

                    st.subheader("❓ AI Generated Quiz")
                    st.write(quiz)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# ---------------- CONCEPT EXPLAINER ----------------

elif feature == "Concept Explainer":

    if st.button("💡 Explain Concept"):

        if not notes.strip():
            st.warning("Please enter a topic first.")

        else:
            with st.spinner("AI is explaining the concept..."):

                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a friendly AI tutor. "
                                    "Explain concepts to college students "
                                    "using simple language, examples, and "
                                    "easy-to-understand points."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Explain this concept simply:\n\n{notes}"
                            }
                        ]
                    )

                    explanation = response.choices[0].message.content

                    st.subheader("💡 AI Explanation")
                    st.write(explanation)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")