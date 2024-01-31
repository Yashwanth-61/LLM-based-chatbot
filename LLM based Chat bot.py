import os
import getpass
os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
loader = WebBaseLoader("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2568977/")
docs = loader.load()
text_content = docs[0].page_content
docs = [Document(page_content=text_content, metadata={"source": "local"})]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
                     documents=docs,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )
vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )
retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})
print(len(retriever.get_relevant_documents("climate")))
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("How can AI address climate challenges?")
# Define the Streamlit app
def main():
    st.title("Q&A Chatbot")

    # Define a text input box for user input
    user_input = st.text_input("You:", "")

    if st.button("Ask"):
        # Invoke the RAG pipeline with the user's question
        bot_response = rag_chain.invoke({"question": user_input})

        # Display the bot's response
        st.text("Bot: " + bot_response)

# Run the Streamlit app
if __name__ == "__main__":
    main()