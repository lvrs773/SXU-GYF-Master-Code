import utils
import pandas as pd
from datetime import datetime
from models.model import ModelPipe



class HotpotQAModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for hotpotqa dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "Were Scott Derrickson and Ed Wood of the same nationality?"
        answer_1 = "Yes.Scott Derrickson and Ed Wood are the same nationality"

        question_2 = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
        answer_2 = "Shirley Temple were held by the woman who portrayed Corliss Archer in the film Kiss and Tell"

        question_3 = "What actors played in the 2009 movie Inglourious Basterds?"
        answer_3 = "The actors are Brad Pitt, Diane Kruger, Eli Roth, Mélanie Laurent, Christoph Waltz, Michael Fassbender, Daniel Brühl, Til Schweiger, Gedeon Burkhard, Jacky Ido, B.J. Novak, Omar Doom."

        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append(eval('question_' + str(i)), (eval('answer_' + str(i))))

        return qa_list

    def read_dataset(self):
        hotpotqa_data_path = './datasets/hotpotqa_pre_processingA.jsonl' # 跑第二组，第三组，替换成对应的文件名称
        hotpotqa_data = pd.read_csv(hotpotqa_data_path).dropna()

        return hotpotqa_data

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        hotpotqa_dataset = self.read_dataset()
        for index, hotpotqa in hotpotqa_dataset.iterrows():
            if index > 4000:
                break
            # get hotpotqa data
            question = eval(hotpotqa['"question"'])
            answer = eval(hotpotqa['"answer"'])

            yield question, answer

    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """

        predicted_cast = utils.spacy_extract_entities(predicted_answer)
        intersection, union = utils.calculate_intersection_and_union(kwargs['answer'], predicted_cast)

        threshold = 0.8
        answer_simple_heuristic = len(intersection) / len(predicted_cast) > threshold if len(predicted_cast) != 0 else True
        return answer_simple_heuristic

    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        original_name = kwargs['question'].lower()
        predicted_question = predicted_question.lower()
        return original_name in predicted_question
