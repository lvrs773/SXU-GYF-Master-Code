import pandas as pd
from unidecode import unidecode
from models.model import ModelPipe


class CommonsenseQAModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for commonsenseqa dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
        choices_1 = "[{\"label\": \"A\", \"text\": \"doctor\"}, {\"label\": \"B\", \"text\": \"bookstore\"}, {\"label\": \"C\", \"text\": \"market\"}, {\"label\": \"D\", \"text\": \"train station\"}, {\"label\": \"E\", \"text\": \"mortuary\"}]"
        answer_1 = "B. Bookstore is the security measure for convenient for two direction travel"

        question_2 = "What are you waiting alongside with when you're in a reception area?"
        choices_2 = "[{\"label\": \"A\", \"text\": \"motel\"}, {\"label\": \"B\", \"text\": \"chair\"}, {\"label\": \"C\", \"text\": \"hospital\"}, {\"label\": \"D\", \"text\": \"people\"}, {\"label\": \"E\", \"text\": \"hotels\"}]"
        answer_2 = "D. Examine thing is the aiting alongside with when you're in a reception area"

        question_3 = "Who is a police officer likely to work for?"
        choices_3 = "[{\"label\": \"A\", \"text\": \"keep cloesd\"}, {\"label\": \"B\", \"text\": \"train\"}, {\"label\": \"C\", \"text\": \"ignition switch\"}, {\"label\": \"D\", \"text\": \"drawer\"}, {\"label\": \"E\", \"text\": \"firearm\"}]"
        answer_3 = "C. City is the police officer working area"


        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append((eval('question_' + str(i)), eval('choices' + str(i)), (eval('answer_' + str(i)))))

        return qa_list

    def read_dataset(self):
        comnonsenseqa_data_path = './datasets/commonsenseqa_pre_processingA.jsonl' # 跑第二组，第三组，替换成对应的文件名称
        comnonsenseqa_data_path = pd.read_json(comnonsenseqa_data_path, lines=True)
        
        return comnonsenseqa_data_path

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        comnonsenseqa_data_dataset = self.read_dataset()
        for index, comnonsenseqa in comnonsenseqa_data_dataset.iterrows():
            if index > 4000:
                break
            # get comnonsenseqa data
            question = eval(comnonsenseqa['"question"'])
            choices = eval(comnonsenseqa['"choices"'])
            answer = eval(comnonsenseqa['"answer"'])

            yield question, choices, answer

    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """
        answer_simple_heuristic = sum([1 for x in kwargs.values() if x.lower() in predicted_answer.lower()])

        return answer_simple_heuristic

    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        original_name = kwargs['question'].lower()
        predicted_question = predicted_question.lower()
        return original_name in predicted_question or original_name.split(':')[0] in predicted_question

