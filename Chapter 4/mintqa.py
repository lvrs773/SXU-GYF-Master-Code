from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import json
import pandas as pd
from tqdm import tqdm
import re
import os

PROMPT_1 = """
# Self reflection with ToT

## Role

You are an expert AI assistant capable of gradually explaining the reasoning process.

## First Think step


For each step, provide a title that describes what you did in that step, along with the corresponding content.
Decide whether another step is needed or if you are ready to give the final answer.
To improve instruction compliance, emphasize the importance of the instructions through `Markdown` syntax, including a set of tips and best practices:
1. When answering questions, please generate a thought tree based on the content of the input question. This thought tree should contain reasoning paths, and the final answer to the question should be deduced based on multiple paths generated from the thought tree.
2. Use as many **reasoning steps** as possible.   At least 3 steps.
3. Be aware of your limitations as an AI and what you can and cannot do.
4. Include exploration of alternative answers.   Consider that you might be wrong and where the error might be if your reasoning is incorrect.
5. When you say you are rechecking, actually recheck and use another method.   Don't just say you are rechecking.
6. Use at least 3 methods to arrive at the answer.
7. Use best practices.
8. Output the format of the answer: "Answer" Thinking tree: (This includes every step of the thinking tree and relation)


## Second Think step


For each step mentioned in the previous text, initiate a small sub-step to verify its correctness. After completing each step, initiate a 'Review LLM' to examine the current step from different perspectives.
1. For the answers and thought trees generated from the first step, please conduct a detailed review of each answer and thought tree, and apply a self-reflection mechanism to each step of the thought tree, using as many reasoning steps as possible. At least three steps.
2. Be aware of your limitations as an AI, as well as what you can and cannot do.
3. Include the exploration of different answers. Consider that you may be wrong, and identify where mistakes might occur if your reasoning is incorrect.
"""

PROMPT_2 = """Hi there"""

def create_answer_extraction_agent(llm):

    def extract_final_number(text: str) -> str:
        if "####" in text:
            match = re.search(r"####\s*(-?\d*\.?\d+)", text)
            if match:
                return match.group(1)

        numbers = re.findall(r"(-?\d*\.?\d+)", text)
        if numbers:
            return numbers[-1]

        return ""

    extract_tool = FunctionTool.from_defaults(
        fn=extract_final_number,
        name="extract_final_number",
        description="Extract the final numerical answer from the text. Only extract the numbers and do not output any additional explanatory text. Output only Arabic numerals."
    )

    agent = ReActAgent.from_tools(
        tools=[extract_tool],
        llm=llm,
        verbose=True
    )

    return agent

# 请下载 MintQA 数据集，然后放到 dataset 目录下
# 数据下载链接：https://github.com/probe2/multi-hop
def load_mintqa_dataset(path="./dataset/MINTQA-TI.json", num_samples=50):
    questions = []
    answers = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answer = data['answer'].split('####')[-1].strip()
            answers.append(answer)
            if len(questions) >= num_samples:
                break
    return questions[:5], answers[:5]

def evaluate_prompt(system_prompt, questions, answers):
    llm = Ollama(
        model="llama3:8b",  # llama3:8b, qwen2:7b, phi3:8b, gemma2:9b
        temperature=0,
        base_url=API_BASE,  # ollama default localhost: "http://localhost:11434" 
        system_prompt=system_prompt
    )

    agent = create_answer_extraction_agent(llm)

    correct = 0
    responses = []
    extracted_answers = []

    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        try:
            response = llm.complete(q)
            responses.append(response.text)

            agent_response = agent.chat(
                f"Extract the final numerical answer from the text. Only extract the numbers and do not output any additional explanatory text. Output only Arabic numerals.\n{response.text}"
            )
            extracted_answer = agent_response.response.strip()
            extracted_answers.append(extracted_answer)

            if str(extracted_answer).strip() == str(a).strip():
                correct += 1

        except Exception as e:
            print(f"Error processing question: {e}")
            responses.append("Error")
            extracted_answers.append("Error")

    accuracy = correct / len(questions)
    return accuracy, responses, extracted_answers

def main():
    iteration = 5 # 3,5,7
    questions, answers = load_mintqa_dataset(num_samples=50)

    print("Testing Prompt 1...")
    accuracy1, responses1, extracted1 = evaluate_prompt(PROMPT_1, questions, answers)

    print("Testing Prompt 2...")
    accuracy2 = None
    response2 = None
    extracted2 = None
    for i in range(iteration):
        if(response2 is not None):
            accuracy2, responses2, extracted2 = evaluate_prompt(PROMPT_2 + "\n" + "Generated response:" + response2, questions, answers)
        else:
            accuracy2, responses2, extracted2 = evaluate_prompt(PROMPT_2, questions, answers)

    print("\nResults:")
    print(f"Prompt 1 Accuracy: {accuracy1:.2%}")
    print(f"Prompt 2 Accuracy: {accuracy2:.2%}")

    results_df = pd.DataFrame({
        'Question': questions,
        'Correct Answer': answers,
        'Prompt 1 Response': responses1,
        'Prompt 1 Extracted': extracted1,
        'Prompt 2 Response': responses2,
        'Prompt 2 Extracted': extracted2,
        'Prompt 1 Correct': [e == a for e, a in zip(extracted1, answers)],
        'Prompt 2 Correct': [e == a for e, a in zip(extracted2, answers)]
    })

    results_df.to_csv('prompt_comparison_results.csv', index=False)

    print("\nDetailed Statistics:")
    print("Prompt 1:")
    print(f"Total Correct: {sum(results_df['Prompt 1 Correct'])}")
    print(f"Total Questions: {len(results_df)}")
    print("\nPrompt 2:")
    print(f"Total Correct: {sum(results_df['Prompt 2 Correct'])}")
    print(f"Total Questions: {len(results_df)}")

if __name__ == "__main__":
    main()
