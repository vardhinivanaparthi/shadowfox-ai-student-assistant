import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils import (
    extract_text_from_pdf,
    chunk_text,
    create_faiss_index,
    search_similar_chunks
)


# Load API key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------- APP TITLE ----------------

st.title("📄 ShadowFox Document Q&A Assistant")

st.write(
    "Upload a PDF or TXT document and ask questions about its content."
)


# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📤 Upload your document",
    type=["pdf", "txt"]
)


if uploaded_file:

    st.success("Document uploaded successfully! ✅")


    # ---------------- EXTRACT TEXT ----------------

    try:

        if uploaded_file.name.lower().endswith(".pdf"):

            text = extract_text_from_pdf(uploaded_file)

        else:

            text = uploaded_file.read().decode("utf-8")


        if not text.strip():

            st.error("Could not extract text from the document.")

        else:

            # ---------------- CHUNK TEXT ----------------

            chunks = chunk_text(text)

            st.info(
                f"Document processed successfully! "
                f"Created {len(chunks)} chunks."
            )


            # ---------------- CREATE FAISS INDEX ----------------

            with st.spinner("Creating document search index..."):

                index, embeddings = create_faiss_index(chunks)


            st.success("Document is ready for questions! ✅")


            # ---------------- QUESTION ----------------

            question = st.text_input(
                "🔎 Ask a question about your document:"
            )


            if st.button("🔍 Ask Question"):

                if not question.strip():

                    st.warning("Please enter a question.")

                else:

                    with st.spinner("Searching the document and generating answer..."):

                        try:

                            # Find relevant chunks
                            relevant_chunks = search_similar_chunks(
                                question,
                                chunks,
                                index,
                                top_k=3
                            )


                            context = "\n\n".join(
                                relevant_chunks
                            )


                            # Ask AI using retrieved context
                            response = client.chat.completions.create(

                                model="openai/gpt-oss-20b",

                                messages=[

                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a document question-answering "
                                            "assistant. Answer the user's question "
                                            "using ONLY the provided document context. "
                                            "If the answer is not present in the context, "
                                            "say that the information is not available "
                                            "in the document."
                                        )
                                    },

                                    {
                                        "role": "user",
                                        "content": (
                                            f"Document context:\n\n"
                                            f"{context}\n\n"
                                            f"Question:\n{question}"
                                        )
                                    }

                                ]
                            )


                            answer = response.choices[0].message.content


                            # ---------------- DISPLAY ANSWER ----------------

                            st.subheader("💡 AI Answer")

                            st.write(answer)


                            # ---------------- SHOW SOURCES ----------------

                            with st.expander("📚 View relevant document sections"):

                                for i, chunk in enumerate(
                                    relevant_chunks,
                                    start=1
                                ):

                                    st.markdown(
                                        f"**Relevant Section {i}**"
                                    )

                                    st.write(chunk)


                        except Exception as e:

                            st.error(
                                f"AI error: {e}"
                            )


    except Exception as e:

        st.error(
            f"Document processing error: {e}"
        )