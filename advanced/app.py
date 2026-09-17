import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils import extract_text_from_file, chunk_text

load_dotenv()

client = Groq()

st.set_page_config(
    page_title="ShadowFox Document Q&A",
    page_icon="📄"
)

st.title("📄 ShadowFox Document Q&A Assistant")

st.write("Upload a PDF or TXT document and ask questions about its content.")

uploaded_file = st.file_uploader(
    "📤 Upload your document",
    type=["pdf", "txt"]
)

if uploaded_file:

    try:
        document_text = extract_text_from_file(uploaded_file)

        if not document_text.strip():
            st.error("The uploaded document contains no readable text.")
            st.stop()

        chunks = chunk_text(document_text)

        st.success(f"Document processed successfully! {len(chunks)} text chunks created.")

        question = st.text_input(
            "❓ Ask a question about the document:"
        )

        if st.button("🔍 Ask Question"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("AI is searching the document..."):

                    context = "\n\n".join(chunks[:5])

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a document question-answering assistant. "
                                    "Answer the user's question using only the provided "
                                    "document context. If the answer is not found in "
                                    "the context, say that the information is not "
                                    "available in the document."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Document context:\n\n{context}\n\n"
                                    f"Question: {question}"
                                )
                            }
                        ]
                    )

                    answer = response.choices[0].message.content

                    st.subheader("💡 Answer")
                    st.write(answer)

    except Exception as e:
        st.error(f"Something went wrong: {e}")