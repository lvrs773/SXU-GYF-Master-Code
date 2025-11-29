import os
import csv
import json
import re

# 数据集下载链接：https://ai.google.com/research/NaturalQuestions/download
# 下载 naturalquestions_simplified_dev-datasets.jsonl.gz 数据集，然后解压获取对应的数据集文件 naturalquestions_simplified_dev-datasets.jsonl
# 实验过程中在所选的三个数据集中共抽取了3000个问题进行测试，所抽取的每个数据集包含1000个问题，抽取的问题分为 A、B、C 组

def pre_process_naturalquestions(startindex, endindex, read_file_path, save_file_path):    
    try:
        with open(read_file_path, 'r', encoding='utf-8') as infile, \
             open(save_file_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                if i < startindex:
                    continue
                if i > endindex:
                    break
                
                try:
                    data = json.loads(line.strip())
                    
                    question_text = data.get('question_text', '')
                    document_html = data.get('document_html', '')
                    
                    new_data = {
                        'question_text': question_text,
                        'document_html': document_html
                    }
                    
                    outfile.write(json.dumps(new_data) + '\n')
                    
                except json.JSONDecodeError:
                    try:
                        question_match = re.search(r'"question_text":\s*"([^"]*)"', line)
                        html_match = re.search(r'"document_html":\s*"([^"]*)"', line)
                        
                        if question_match and html_match:
                            question_text = question_match.group(1)
                            document_html = html_match.group(1)
                            
                            new_data = {
                                'question_text': question_text,
                                'document_html': document_html
                            }
                            
                            outfile.write(json.dumps(new_data) + '\n')
                        else:
                            print(f"Error: Line {i} failed processing - {str(e)}")
                    except Exception as e:
                        print(f"Error: Line {i} failed processing - {str(e)}")
                
                except Exception as e:
                    print(f"Error: Line {i} failed processing - {str(e)}")
            
            print(f"Successfully processed the row {startindex} to {endindex}, and saved the result to {save_file_path}")
    
    except FileNotFoundError:
        print(f"Error: Input file cannot be found {read_file_path}")
    except Exception as e:
        print(f"Error: An exception occurred while processing the file - {str(e)}")

if __name__ == '__main__':
    read_file_path = './datasets/naturalquestions_simplified_dev-datasets.jsonl'
    save_file_pathA = './datasets/naturalquestions_pre_processingA.jsonl'
    save_file_pathB = './datasets/naturalquestions_pre_processingB.jsonl'
    save_file_pathC = './datasets/naturalquestions_pre_processingC.jsonl'
    if not os.path.isfile(save_file_pathA):
        print('Pre processing group A natural questions...')
        pre_process_naturalquestions(1000, 2000, read_file_path, save_file_pathA)
        print('group A natural questions all created.')
    else:
        print('The group A natural questions all already exists')

    if not os.path.isfile(save_file_pathB):
        print('Pre processing group B natural questions...')
        pre_process_naturalquestions(5000, 6000, read_file_path, save_file_pathB)
        print('group B natural questions all created.')
    else:
        print('The group B natural questions all already exists')

    if not os.path.isfile(save_file_pathC):
        print('Pre processing group C natural questions...')
        pre_process_naturalquestions(10000, 11000, read_file_path, save_file_pathC)
        print('group C natural questions all created.')
    else:
        print('The group C natural questions all already exists')    
