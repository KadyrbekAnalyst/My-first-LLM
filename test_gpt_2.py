from transformers import pipeline

generator = pipeline('text-generation', model='gpt-2')

prompt = "Extract keywords and generate SQL query: Select all users from New York"
result = generator(prompt, max_length=100)
sql_query = result[0]['generated_text']
