import streamlit as st 
from snowflake.snowpark import Session
from config import *
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone


conn = {
    "user"  : snowflake_user,
    "password": snowflake_password,
    "account": snowflake_account,
    "warehouse": snowflake_warehouse,
    "database": snowflake_database,
    "schema": snowflake_schema
}

session = Session.builder.configs(conn).create()
dataframe = session.sql(query)
st.write(dataframe)

presigned_urls_list = [row[0] for row in dataframe.collect()]
st.write(presigned_urls_list)


loader = UnstructuredURLLoader(urls=presigned_urls_list)
documents = loader.load()

st.write("Documents : ",len(documents))


def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  docs=text_splitter.split_documents(documents)
  return docs
docs=split_docs(documents)
st.write("Vectors : ",len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

pinecone.init(
    api_key=api_key,
    environment=environment
)

index = Pinecone.from_documents(docs,embeddings,index_name=index_name)
