from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document 
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA 
import os
from dotenv import load_dotenv
from src.prompt import *



## OpenAI authentication
load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


## File Processing Function
def file_processing(file_path):
    loader=PyPDFLoader(file_path)
    data=loader.load()
    
    question_gen=''
    for page in data:
        question_gen +=page.page_content
    
    splitter_ques_gen=TokenTextSplitter(
        model_name='gpt-3.5-turbo',
        chunk_size=10000,
        chunk_overlap=200
    )
    
    chunks_ques_gen=splitter_ques_gen.split_text(question_gen)
    
    document_ques_gen=[Document(page_document=t) for t in chunks_ques_gen]
    
    splitter_ans_gen=TokenTextSplitter(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=100
    )
    
    document_answer_gen=splitter_ans_gen.split_documents(
        document_ques_gen
    )
    
    return document_ques_gen,document_answer_gen


## LLM Pipeline
def llm_pipeline(file_path):
    
    document_ques_gen,document_answer_gen=file_processing(file_path)
    
    llm_ques_gen_pipeline=ChatOpenAI(
        temprature=0.3,
        model='gpt-3.5-turbo'
        
    )
    
    PROMPT_QUESTIONS= PromptTemplate(template=prompt_template,input_variables=['text'])
    
    # Define the refine prompt template
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Load the summarize chain for question generation
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # Generate questions
    ques = ques_gen_chain.run(document_ques_gen)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    
    
    # Initialize the LLM with specific temperature and model
    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # Split the input questions into a list and filter by specific criteria
    ques_list = ques.split("\n")
    filtered_ques_list = [
        element for element in ques_list
        if element.endswith("?") or element.endswith(".")
    ]

    # Create the answer generation chain
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Return the chain and the filtered questions list
    return answer_generation_chain, filtered_ques_list