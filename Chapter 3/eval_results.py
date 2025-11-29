import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score , precision_recall_curve

import utils
from cos_baselines import embedding_model_thresholds

heuristic_thresholds = {'naturalquestions':0.8, 'hotpotqa':2, 'commonsenseqa':None}

def naturalquestions_answer_heuristic(predicted_answer, gt_answer, threshold=0.8):
    predicted_cast = utils.spacy_extract_entities(predicted_answer)
    intersection, union = utils.calculate_intersection_and_union(gt_answer['naturalquestions_cast'], predicted_cast)

    answer_simple_heuristic = len(intersection) / len(predicted_cast) > threshold if len(predicted_cast) != 0 else True
    return answer_simple_heuristic

def hotpotqa_answer_heuristic(predicted_answer, gt_answer, threshold=2):
    answer_simple_heuristic = sum([1 for x in gt_answer.values() if utils.check_entity_in_sentence(x, predicted_answer)])

    return answer_simple_heuristic >= threshold

def commonsenseqa_answer_heuristic(predicted_answer, gt_answer):
    for x in gt_answer.values():
        for i in x:
            if utils.check_entity_in_sentence(i, predicted_answer):
                return True
    return False



def calc_ans_heuristic(predicted_answer, gt_answer, dataset_name, heuristic_threshold):
    if dataset_name == 'naturalquestions':
        gt = [naturalquestions_answer_heuristic(x, y, heuristic_threshold) for x, y in zip(predicted_answer, gt_answer)]
    elif dataset_name == 'hotpotqa':
        gt = [hotpotqa_answer_heuristic(x, y, heuristic_threshold) for x, y in zip(predicted_answer, gt_answer)]
    elif dataset_name == 'commonsenseqa':
        gt = [commonsenseqa_answer_heuristic(x, y) for x, y in zip(predicted_answer, gt_answer)]
    else:
        gt = []

    return gt


def auc_plot(gt, pred, title, file_name, save_path='./'):
    pred = [x if x <= 1 else 1.0 for x in pred]
    
    pred = 1 - np.array(pred)
    gt = np.array(gt)
    gt = 1 - gt
    
    # calculate roc curve
    fpr, tpr, _ = roc_curve(gt, pred)

    ns_probs = [0 for _ in range(len(gt))]
    ns_fpr, ns_tpr, _ = roc_curve(gt, ns_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # add the auc score and optimal threshold to the plot
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    # plt.text(0.7, 0.02, f'AUC: {roc_auc_score(gt, pred):.3f}\nOptimal Threshold: {optimal_threshold:.3f}\nBalanced Acc: {bal_acc:.3f}', fontsize=8, bbox=props)
    plt.text(0.1, 0.02, f'AUC: {roc_auc_score(gt, pred):.3f}\n', fontsize=8, bbox=props)

    # title
    plt.title(title)

    # show the plot
    save_path = os.path.join(save_path, file_name)
    plt.savefig(save_path)
    plt.cla()


def calc_auc_and_bal_acc(gt, pred):
    pred = [x if x <= 1 else 1.0 for x in pred]
    gt = 1 - gt
    
    pred = 1 - np.array(pred)
    gt = np.array(gt)


    # calculate optimal threshold
    optimal_threshold = embedding_model_thresholds['ada002']

    bal_acc = balanced_accuracy_score(gt, np.array(pred) > optimal_threshold)
    auc = roc_auc_score(gt, pred)
    
    return auc, bal_acc

def calc_auc_and_acc(base_dir, heuristic_threshold, avg_max='avg', k_range=1, dataset_name='naturalquestions'):
    current_dir = os.path.join(base_dir, 'res_pkl')
    save_dir = os.path.join(base_dir, 'res', f'k={k_range}', avg_max)
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(current_dir):
        sample_sizes = [int(i.split('_')[-1].split('.')[0]) for i in os.listdir(current_dir)]
        sample_size = int(f.split('_')[-1].split('.')[0])
        file_path = os.path.join(current_dir, f)

        if dataset_name == 'commonsenseqa':
            if sample_size != max(sample_sizes):
                continue
        else:
            if sample_size != 3000:
                continue
        with open(file_path, 'rb') as handle:
            results = pkl.load(handle)

            gt_answers = [res['answer_args'] for res in results]
            pred_ans = [res['predicted_answer'] for res in results]
            

            gt = calc_ans_heuristic(pred_ans, gt_answers, dataset_name, heuristic_threshold)


            exp_type = 'predicted_questions_const'
            exp_type = 'predicted_questions_var'
            
            predicted_questions_cosine = [{key: value for m_res in res[exp_type] for key, value in m_res.items()} for res in results]
            
            pred_questions_cosine_llama3_8b = [res['llama3:8b'][:k_range] for res in predicted_questions_cosine]
            pred_questions_cosine_llama3_8b = [[item[2] for item in inner_list] for inner_list in pred_questions_cosine_llama3_8b]
            
            pred_questions_cosine_qwen2_7b = [res['qwen2:7b'][:k_range] for res in predicted_questions_cosine]
            pred_questions_cosine_qwen2_7b = [[item[2] for item in inner_list] for inner_list in pred_questions_cosine_qwen2_7b]
            
            pred_questions_cosine_phi3_8b = [res['phi3:8b'][:k_range] for res in predicted_questions_cosine]
            pred_questions_cosine_phi3_8b = [[item[2] for item in inner_list] for inner_list in pred_questions_cosine_phi3_8b]
            
            pred_questions_cosine_gemma2_9b = [res['gemma2:9b'][:k_range] for res in predicted_questions_cosine]
            pred_questions_cosine_gemma2_9b = [[item[2] for item in inner_list] for inner_list in pred_questions_cosine_gemma2_9b]

            pred_questions_cosine_ensemble = [res1 + res2 + res3 for res1, res2, res3 in zip(pred_questions_cosine_llama3_8b, pred_questions_cosine_qwen2_7b, pred_questions_cosine_phi3_8b, pred_questions_cosine_gemma2_9b)]

            for f, f_name in zip([np.max, np.average], ['max', 'avg']):
                print(f_name + '\n')
                pred_questions_cosine_llama3_8b_ = [f(x) for x in pred_questions_cosine_llama3_8b]
                pred_questions_cosine_qwen2_7b_ = [f(x) for x in pred_questions_cosine_qwen2_7b]
                pred_questions_cosine_phi3_8b_ = [f(x) for x in pred_questions_cosine_phi3_8b]
                pred_questions_cosine_ensemble_ = [f(x) for x in pred_questions_cosine_ensemble]


                print(f'k={k_range}, sample_size={sample_size}, heuristic_threshold={heuristic_threshold}')
                print(f'Hallucination rate: {1 - (sum(gt)/len(gt)):.3f}')
                llama3_8b_res = calc_auc_and_bal_acc(gt, pred_questions_cosine_llama3_8b_)
                print(f'llama3_8b:\n  AUC: {llama3_8b_res[0]:.3f}, Balanced Acc: {llama3_8b_res[1]:.3f}')
                qwen2_7b_res = calc_auc_and_bal_acc(gt, pred_questions_cosine_qwen2_7b_)
                print(f'llama7:\n  AUC: {qwen2_7b_res[0]:.3f}, Balanced Acc: {qwen2_7b_res[1]:.3f}')
                phi3_8b_res = calc_auc_and_bal_acc(gt, pred_questions_cosine_phi3_8b_)
                print(f'llama13:\n  AUC: {phi3_8b_res[0]:.3f}, Balanced Acc: {phi3_8b_res[1]:.3f}')
                ensemble_res = calc_auc_and_bal_acc(gt, pred_questions_cosine_ensemble_)
                print(f'ensemble:\n  AUC: {ensemble_res[0]:.3f}, Balanced Acc: {ensemble_res[1]:.3f}')
                print('\n\n')


def eval_exp(exp_dir, n_models, k_range=1, dataset_name='naturalquestions'):
    calc_auc_and_acc(exp_dir,
                        k_range=k_range,
                        dataset_name=dataset_name,
                        heuristic_threshold=heuristic_thresholds[dataset_name],
                        n_models=n_models
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='naturalquestions', choices=['naturalquestions', 'hotpotqa', 'commonsenseqa'],
                        help='dataset name')

    parser.add_argument('--ans_model', type=str, default='llama3-8b', choices=['llama3-8b', 'qwen2-7b', 'phi3-8b', 'gemma2-9b'],
                        help='llm model name')

    parser.add_argument('--embedding_model_name', type=str, default='sbert', choices=['ada002', 'sbert', 'e5','bert'],
                        help='embedding model name')
    args = parser.parse_args()

    question_models_name = ['llama3-8b', 'qwen2-7b', 'phi3-8b', 'gemma2-9b']
    exp_dir = f'{args.dataset_name}_experiments'
    save_dir = os.path.join('.', exp_dir, args.embedding_model_name, args.answer_model_name, '-'.join(question_models_name))

    print(f'exp_dir: {exp_dir}')
    print(f'Ans model: {args.ans_model}, Dataset: {args.dataset_name}')
    eval_exp(exp_dir, k_range=5, dataset_name=args.dataset_name, n_models=len(question_models_name))

        
