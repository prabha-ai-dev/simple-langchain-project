from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline

hf_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",   
    max_new_tokens=50
)

model = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)