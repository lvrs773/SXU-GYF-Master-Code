import re
import numpy as np
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from sentence_transformers import util
import re
from typing import List
import requests
import json

def question_pred(model, prompt, model_name, temperatures):
    res = []
    for temp in temperatures:
        response = model.submit_request(prompt, temperature=temp, split_by='Answer:')
        response = [res for res in response if res != '']
        response = ', '.join(response)
        res.append((temp, response))
        
    return {model_name:res}

# reconstruct question from the answer using the reconstruction models (llama3-8b, qwen2-7b, phi3-8b, gemma2-9b) using multiprocessing pool for parallelism 
def reconstruct_pool(models, prompt, temperatures):
    with Pool() as pool:
        tmp = pool.starmap(question_pred, [(models[0], prompt, 'llama3:8b', temperatures), (models[1], prompt, 'qwen2:7b', temperatures), (models[2], prompt, 'phi3:8b', temperatures), (models[3], prompt, 'gemma2:9b', temperatures)])
    return tmp


class OurMethod:
    def __init__(self, answer_model, reconstruction_models, embedding_model, embedding_threshold, t_0=0.6):
        self.answer_model = answer_model
        self.reconstruction_models = reconstruction_models
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold
        self.t_0 = t_0

    # for natural questions and hotpot qa
    @staticmethod
    def create_few_shot_prompt1(question_answer_list, inverse=False):
        few_shot_prompt = ""
        for question,content, answer in question_answer_list:
            question = 'Question: ' + question
            answer = 'Answer: ' + answer
            if inverse:
                few_shot_prompt += answer + "\n" + question + "\n\n"
            else:
                few_shot_prompt += question + "\n" + answer + "\n\n"

        return few_shot_prompt
    
    # for commonsense qa
    def create_few_shot_prompt2(question_choices_answer_list, inverse=False):
        few_shot_prompt = ""
        for question, answer in question_choices_answer_list:
            question = 'Question: ' + question
            choices = 'Choices' + choices
            answer = 'Answer: ' + answer
            if inverse:
                few_shot_prompt += answer + "\n" + question + "\n\n"
            else:
                few_shot_prompt += question + "\n" + choices + "\n" + answer + "\n\n"

        return few_shot_prompt

    def generate_answer(prompt, triplet_tuple, search_content,  model, temperature: float = 0.6, max_tokens: int = 1000) -> str:
        # Extract entities from triplet for context
        try:
            # Parse the triplet format <head, relation, tail>
            if triplet_tuple.startswith('<') and triplet_tuple.endswith('>'):
                triplet_content = triplet_tuple[1:-1]
                parts = [part.strip() for part in triplet_content.split(',')]
                if len(parts) == 3:
                    head, relation, tail = parts
                else:
                    head, relation, tail = "unknown", "unknown", "unknown"
            else:
                head, relation, tail = "unknown", "unknown", "unknown"
        except:
            head, relation, tail = "unknown", "unknown", "unknown"
        
        # Construct the enhanced prompt with all available information
        enhanced_prompt = f"""
        Based on the following information, please provide a comprehensive answer:
        
        ORIGINAL QUESTION: {prompt}
        
        KNOWLEDGE TRIPLET: Head: {head}, Relation: {relation}, Tail: {tail}
        
        RELEVANT SEARCH CONTENT: {search_content}
        
        Please generate a detailed and accurate answer that synthesizes all available information.
        Focus on providing clear, factual information that directly addresses the original question.
        """
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": enhanced_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()            
            result = response.json()
            return result.get('response', 'No response generated from the model.')
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama service: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Error parsing response from Ollama: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def reconstruct_question(self, predicted_answer, inverse_prompt_prefix , temperatures):
        answer_question_instructions = 'Follow the format below, and please only predict the question that corresponds to the last answer.\n\n'
        answer_question_prompt = answer_question_instructions + inverse_prompt_prefix + 'Answer: ' + predicted_answer + '\n' + 'Question: '
        
        res = reconstruct_pool(self.reconstruction_models, answer_question_prompt, temperatures)
        return res

    def model_run(self, query, few_shot_examples, iterations=5, variable_temp=False):
        # create instructions for the model
        question_answer_instructions = 'Follow the format below, and please only predict the answer that corresponds to the last question.\n\n'
        # prompt prefix (instructions + few shot examples)
        # naturalquestions, hotpotqa
        prompt_prefix = question_answer_instructions + self.create_few_shot_prompt1(few_shot_examples)
        # prompt_prefix = question_answer_instructions + self.create_few_shot_prompt2(few_shot_examples)
        triplet_tuple = self.answer_model.submit_request(prompt_prefix + "Please extract the problem from the input content, and then summarize the triplet form of the problem based on the input problem. The output format is: \"<Head Entity, relationship, Tail Entity>\"")
        search_content = self.answer_model.submit_request(prompt_prefix + "Please search for the text content related to the question from the input questions")
        answer = self.answer_model.submit_request("question: " + prompt_prefix + "\n" + "triplet: " + triplet_tuple + "\n" + + "content:" + search_content + "Please answer this question based on the input question, triples and related text")

        # create inverse prompt prefix (instructions + few shot examples) for predicting questions from the answer (reconstruction step)
        inverse_prompt_prefix = self.create_few_shot_prompt1(few_shot_examples, inverse=True)
        #inverse_prompt_prefix = self.create_few_shot_prompt2(few_shot_examples, inverse=True)

        # add few shot prefix to the question to create the prompt
        prompt = prompt_prefix + 'Question: ' + query + '\n' + 'Answer: ' + answer + "Please think step by step when answering the questions and output each step when answering the questions"
        # submit the prompt to the model
        response = self.answer_model.submit_request(prompt, split_by='Question:')
        # remove empty strings from response
        response = [res for res in response if res != '']
        # remove leading '-' from the response
        response = list(map(lambda x: re.sub("^([-]*)", "", x), response))
        # concatenate the response to a single string
        predicted_answer = ', '.join(response)
        
        if len(response) == 0:
            print('Response is empty')

        temperatures = [self.t_0] * iterations if not variable_temp else [self.t_0 + (1-self.t_0) * i/iterations for i in range(0, iterations)]
        # predict questions from the answer (get list of questions according to the answer)
        predicted_questions = self.reconstruct_question(predicted_answer, inverse_prompt_prefix, temperatures)

        # get embedding vector for original question
        original_question_embedding = self.embedding_model.submit_embedding_request(query)
        
        # get embedding vector for each reconstructed question and calculate cosine similarity (const temperature)
        pred_questions_embedding = []
        for model_pred in predicted_questions:
            model_name = list(model_pred.keys())[0]
            model_pred_res = model_pred[model_name]
            res = []
            for temp, pred_question in model_pred_res:
                embedding_pred_question = self.embedding_model.submit_embedding_request(pred_question)
                questions_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                res.append((temp, pred_question, questions_cosine_similarity))
            pred_questions_embedding.append({model_name:res})
        
        cosine_scores = [score for res in pred_questions_embedding for key, value in res.items() for _, _, score in value]
        avg_cosine_score = np.average(cosine_scores)
        

        print(f'Query:\n{query}')
        print(f'Predicted Answer:\n{predicted_answer}\n')

        return avg_cosine_score < self.embedding_threshold
