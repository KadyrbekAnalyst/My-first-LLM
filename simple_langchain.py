from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Integrate with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate(
    input_variables=["city"],
    template="Describe a perfect day in {city}?",
)

llmchain = LLMChain(llm=llm, prompt=prompt)
print(llmchain.run("Paris"))
