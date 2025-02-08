from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from flask import Flask,request,jsonify
from flask_restful import Api, Resource, reqparse
from dotenv import load_dotenv
from langchain_community.llms import Ollama

app = Flask(__name__)
api = Api(app)

#Function to create vector store
def create_vector_store():
    loader=WebBaseLoader(web_paths=("https://brainlox.com/courses/category/technical",))
    text_documents=loader.load()
    
  
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=text_splitter.split_documents(text_documents)
       
    embeddings = OllamaEmbeddings(model="llama3")
    vector_stores=FAISS.from_documents(docs,embeddings)
    return vector_stores

vector_stores=create_vector_store()


prompt = ChatPromptTemplate.from_template("""
Answer the following input based only on the provided context. 
Think step by step before providing a detailed answer.  
<context>
{context}
</context>
Input: {input}""")

llm=Ollama(model='llama3')


retriever = vector_stores.as_retriever()


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

Chat = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}  # Adding the custom prompt
)

class ChatBotAPI(Resource):
    def post(self):
        try:
            data = request.get_json()
            print("Received data:", data)

            user_input = data.get("input") 
            
            if not user_input:
                return {"error": "Missing 'input' parameter"}, 400
            
            response = Chat.invoke({"input": user_input})
    
            return ({"response": response["answer"]}) 

        except Exception as e:
            return({"error": str(e)}), 500
        
api.add_resource(ChatBotAPI, "/chat")
if __name__ == "__main__":
    app.run(debug=True)

