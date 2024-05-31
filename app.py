import streamlit as st
from transformers import pipeline, TFAutoModelForQuestionAnswering, AutoTokenizer
import pdfplumber
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model for question answering
logging.info("Loading model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf_file:
            text = ""
            for page in pdf_file.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error("Error extracting text from PDF: %s", e)
        return None

# Function to answer user questions
def answer_question(text, question):
    try:
        answer = qa_pipeline(question=question, context=text)
        return answer['answer']
    except Exception as e:
        logging.error("Error answering question: %s", e)
        return None

def main():
    st.title("PDF Question Answering")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully!")
        st.write("Now, ask your question:")
        
        question = st.text_input("Question:")
        
        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please ask a question.")
            else:
                try:
                    text = extract_text_from_pdf(uploaded_file)
                    if text is None:
                        st.error("Failed to extract text from PDF.")
                    else:
                        answer = answer_question(text, question)
                        if answer is None:
                            st.error("Failed to get an answer.")
                        else:
                            st.success("Answer: " + answer)
                except Exception as e:
                    st.error("An error occurred: {}".format(str(e)))

if __name__ == "__main__":
    main()
