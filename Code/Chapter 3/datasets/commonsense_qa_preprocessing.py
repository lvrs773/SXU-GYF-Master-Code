import os
import csv
import json
import re

# 数据集下载链接：https://huggingface.co/datasets/tau/commonsense_qa
# 下载 CommonsenseQA.tar.gz 数据集，然后解压获取对应的数据集文件 train_rand_split.jsonl
# 实验过程中在所选的三个数据集中共抽取了3000个问题进行测试，所抽取的每个数据集包含1000个问题，抽取的问题分为 A、B、C 组

def pre_process_commonsense_qa(startindex, endindex, read_file_path, save_file_path):    
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
                    
                    stem = data.get('question', {}).get('stem', '')
                    choices = data.get('question', {}).get('choices', [])
                    
                    new_data = {
                        'stem': stem,
                        'choices': choices
                    }
                    
                    outfile.write(json.dumps(new_data) + '\n')
                    
                except json.JSONDecodeError:
                    try:
                        stem_match = re.search(r'"stem":\s*"([^"]*)"', line)
                        choices_matches = re.findall(r'{"label":\s*"([A-Z])",\s*"text":\s*"([^"]*)"}', line)
                        
                        if stem_match:
                            stem = stem_match.group(1)
                            choices = [{"label": match[0], "text": match[1]} for match in choices_matches]
                            
                            new_data = {
                                'stem': stem,
                                'choices': choices
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
    read_file_path = './datasets/train_rand_split.jsonl'
    save_file_pathA = './datasets/commonsenseqa_pre_processingA.jsonl'
    save_file_pathB = './datasets/commonsenseqa_pre_processingB.jsonl'
    save_file_pathC = './datasets/commonsenseqa_pre_processingC.jsonl'
    if not os.path.isfile(save_file_pathA):
        print('Pre processing group A commonsense qa...')
        pre_process_commonsense_qa(1500, 2500, read_file_path, save_file_pathA)
        print('group A commonsense qa all created.')
    else:
        print('The group A commonsense qa all already exists')

    if not os.path.isfile(save_file_pathB):
        print('Pre processing group B commonsense qa...')
        pre_process_commonsense_qa(3300, 6300, read_file_path, save_file_pathB)
        print('group B commonsense qa all created.')
    else:
        print('The group B commonsense qa all already exists')

    if not os.path.isfile(save_file_pathC):
        print('Pre processing group C commonsense qa...')
        pre_process_commonsense_qa(7000, 8000, read_file_path, save_file_pathC)
        print('group C commonsense qa all created.')
    else:
        print('The group C commonsense qa all already exists')    
