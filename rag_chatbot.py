import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import numpy as np

# Load sample telecom standards documents
# In a real implementation, you would load PDFs or other documents
telecom_standards = [
    "5G NR: 3GPP TS 38.300 - Overall architecture",
    "O-RAN: O-RAN Architecture Description v07.00 - The O-RAN architecture consists of Non-Real Time RIC, Near-Real Time RIC, O-CU, O-DU and O-RU.",
    "5G Security: 3GPP TS 33.501 - Security architecture and procedures for 5G system",
    "Network Slicing: 3GPP TS 28.530 - Management and orchestration of network slicing",
    "Quality of Service: 3GPP TS 23.203 - Policy and charging control architecture"
]

# Initialize embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Create documents and vector store
documents = []
for standard in telecom_standards:
    documents.extend(text_splitter.split_documents([{"page_content": standard}]))

vector_store = FAISS.from_documents(documents, embedding_model)

# Initialize LLM for question answering
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create RAG prompt template
RAG_PROMPT_TEMPLATE = """[INST] 
You are a telecom standards expert. Use the following context to answer the question at the end.
Context: {context}
Question: {question}
Answer: 
[/INST]"""

def rag_chatbot(question, top_k=3):
    # Retrieve relevant context
    docs = vector_store.similarity_search(question, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Format prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Generate response
    response = pipe(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return response[0]['generated_text'].split("[/INST]")[-1].strip()

# Example usage
if __name__ == "__main__":
    questions = [
        "What is the O-RAN architecture?",
        "Which 3GPP specification covers 5G security?",
        "What is network slicing?"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        answer = rag_chatbot(question)
        print(f"Answer: {answer}\n")
