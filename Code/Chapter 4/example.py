from llama_index.llms import Ollama

llm = Ollama(model="llama3:8b", temperature=0, request_timeout=120.0) # llama3:8b, qwen2:7b, phi3:8b, gemma2:9b
reasoner = MultiplexToTReasoner()
question = "Why is the sky blue?"
result = await reasoner.process_question(question, llm)
print(reasoner.format_response(result))
