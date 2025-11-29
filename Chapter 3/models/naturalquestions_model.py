import pandas as pd
from models.model import ModelPipe


class NaturalQuestionsModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for naturalquestions dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "Where does the last name painter come from?"
        answer_1 = "Cornwall"

        question_2 = "What do the 3 dots mean in math?"
        answer_2 = "And so on."

        question_3 = "Where is most distortion found on a globe?"
        answer_3 = "Mercator projection."

        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append(eval('question_' + str(i)), (eval('answer_' + str(i))))

        return qa_list

    def read_dataset(self):
        naturalquestions_data_path = './datasets/naturalquestions_pre_processingA.jsonl' # 跑第二组，第三组，替换成对应的文件名称
        naturalquestions_data = pd.read_json(naturalquestions_data_path, lines=True)
        
        return naturalquestions_data

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        naturalquestions_data_dataset = self.read_dataset()
        for index, naturalquestion in naturalquestions_data_dataset.iterrows():
            if index > 4000:
                break
            # get naturalquestion data
            question_text = eval(naturalquestion['"question_text"'])
            question_content = eval(naturalquestion['"question_content"'])

            yield question_text, question_content

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
