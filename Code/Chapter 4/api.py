from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.prompts import PromptTemplate
from llama_index.llms import Ollama
from typing import List, Dict
import logging

class MultiplexToTReasoner:
    def __init__(self):
        # Define the prompt word template for the first thought
        self.first_think_template = PromptTemplate(
            """
            # Role
            You are an expert AI assistant capable of gradually explaining the reasoning process.

            # Task
            Please help analyze the following question:
            {question}

            # First Think step
            For each step, provide a title that describes what you did in that step, along with the corresponding content.
            Decide whether another step is needed or if you are ready to give the final answer.
            To improve instruction compliance, emphasize the importance of the instructions through `Markdown` syntax, including a set of tips and best practices:

            Remember:
            1. When answering questions, please generate a thought tree based on the content of the input question. This thought tree should contain reasoning paths, and the final answer to the question should be deduced based on multiple paths generated from the thought tree.
            2. Use as many **reasoning steps** as possible.   At least 3 steps.
            3. Be aware of your limitations as an AI and what you can and cannot do.
            4. Include exploration of alternative answers.   Consider that you might be wrong and where the error might be if your reasoning is incorrect.
            5. When you say you are rechecking, actually recheck and use another method.   Don't just say you are rechecking.
            6. Use at least 3 methods to arrive at the answer.
            7. Use best practices.
            8. Output the format of the answer: "Answer" Thinking tree: (This includes every step of the thinking tree and relation)

            Please provide your step-by-step analysis:
            """
        )

        # Define the prompt word template for the second thought
        self.second_think_template = PromptTemplate(
            """
            # Review and Verification

            Based on the previous analysis:
            {first_analysis}

            # Second Think step
            For each step mentioned in the previous text, initiate a small sub-step to verify its correctness. After completing each step, initiate a 'Review LLM' to examine the current step from different perspectives.

            Remember:
            1. For the answers and thought trees generated from the first step, please conduct a detailed review of each answer and thought tree, and apply a self-reflection mechanism to each step of the thought tree, using as many reasoning steps as possible. At least three steps.
            2. Be aware of your limitations as an AI, as well as what you can and cannot do.
            3. Include the exploration of different answers. Consider that you may be wrong, and identify where mistakes might occur if your reasoning is incorrect.'''


            Please provide your detailed review:
            """
        )

    async def process_question(self, question: str, llm_service) -> Dict:
        """
        Deal with the problem and return the two-stage thinking results
        """
        try:
            # The first stage of thinking
            first_response = await llm_service.complete(
                self.first_think_template.format(question=question)
            )
            
            # The second stage of thinking
            second_response = await llm_service.complete(
                self.second_think_template.format(first_analysis=first_response)
            )

            return {
                "question": question,
                "first_think": first_response,
                "second_think": second_response
            }
        except Exception as e:
            logging.error(f"Error in processing question: {str(e)}")
            raise

    def format_response(self, response: Dict) -> str:
        """
        Format the response result
        """
        formatted_response = f"""
        # Question
        {response['question']}

        # First Analysis
        {response['first_think']}

        # Review and Verification
        {response['second_think']}
        """
        return formatted_response

# 使用示例
async def main():
    llm = Ollama(model="llama3:8b", temperature=0, request_timeout=60.0) # llama3:8b, qwen2:7b, phi3:8b, gemma2:9b
    reasoner = MultiplexToTReasoner()
    question = "What would happen if the moon suddenly disappeared?"
    result = await reasoner.process_question(question, llm)
    formatted_output = reasoner.format_response(result)
    print(formatted_output)

# 使用方法
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
