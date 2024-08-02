from transformers import pipeline

# For sentiment analysis
classifier = pipeline("sentiment-analysis")

result = classifier("I love using Hugging Face's transformers!")
print(result)