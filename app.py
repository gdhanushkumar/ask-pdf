import streamlit as st
from transformers import pipeline
import pdfplumber

# Load pre-trained model for question answering
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf_file:
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

# Function to answer user questions
def answer_question(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    answer = qa_model(question=question, context=text)
    return answer['answer']

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
                    answer = answer_question(uploaded_file, question)
                    st.success("Answer: " + answer)
                except Exception as e:
                    st.error("An error occurred: {}".format(str(e)))

if __name__ == "__main__":
    main()
