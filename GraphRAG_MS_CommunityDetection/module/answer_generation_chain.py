from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnablePick, RunnableBinding, RunnableAssign, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

llm = ChatOpenAI(model='gpt-4o',
                 temperature=0,
                 openai_api_key=OPENAI_API_KEY
                 )
questionPrompt = PromptTemplate.from_template(
  """
Use Only the following context to answer the following question
 
    Question:
    {input}
 
    Context:
    {context}
 
    Answer as if you have been asked the original question.
    Do not use your pre-trained knowledge.
 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If multiple documents are provided as context, Answer using context more similar to the question.
   
    Answer in detail based on the provided context.
    provide source on the metadata provided at the end of your answer.
 
    Include source on the metadata provided Example:
    If provided metadata is['Entities_ID': '특약', 'Reports': 'The insurance rider "(무)위암·폐암·간암진단특약" provides coverage specifically for the diagnosis of stomach cancer, lung cancer, and liver cancer.', 'score': 0.7551493644714355, 'outline': '1. 기준금리']
    Response: (식별된 엔티티 : 특약, 커뮤니티 리포트 요약: The insurance rider "(무)위암·폐암·간암진단특약" provides coverage specifically for the diagnosis of stomach cancer, lung cancer, and liver cancer., 유사도 점수: 0.7551493644714355)
 
    Answer must be in Korean
  """
)

answer_chain = (
  questionPrompt
  | llm
  | StrOutputParser()
)