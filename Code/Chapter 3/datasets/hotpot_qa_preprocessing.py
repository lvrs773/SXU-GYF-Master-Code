import os
import csv
import json
import re

# 数据集下载链接：https://huggingface.co/datasets/hotpotqa/hotpot_qa
# 下载 hotpot_dev_fullwiki_v1.json 数据集
# 实验过程中在所选的三个数据集中共抽取了3000个问题进行测试，所抽取的每个数据集包含1000个问题，抽取的问题分为 A、B、C 组

def pre_process_hotpot_qa(startindex, endindex, read_file_path, save_file_path):    
    try:
        with open(read_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if startindex < 0 or endindex >= len(data) or startindex > endindex:
            print(f"Error: The index range is invalid. The file contains {len(data)} pieces of data. Please use an index between 0 and {len(data)-1}.")
            return
        
        with open(save_file_path, 'w', encoding='utf-8') as outfile:
            for i in range(startindex, endindex + 1):
                item = data[i]
                
                question = item.get('question', '')
                
                context_list = item.get('context', [])
                context_text = ""
                
                for context_item in context_list:
                    if isinstance(context_item, list) and len(context_item) == 2:
                        title = context_item[0]
                        sentences = context_item[1]
                        
                        context_text += f"{title}: {' '.join(sentences)}\n\n"
                
                output_data = {
                    'question': question,
                    'content': context_text.strip()
                }
                
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        
        print(f"Successfully processed the data from index {startindex} to {endindex}, and saved the result to {save_file_path}.")
        print(f"A total of {endindex - startindex + 1} pieces of data were processed.")

    except FileNotFoundError:
        print(f"Error: Input file cannot be found {read_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse the JSON file {read_file_path}")
    except Exception as e:
        print(f"Error: An exception occurred while processing the file - {str(e)}")

if __name__ == '__main__':
    read_file_path = './datasets/hotpot_dev_fullwiki_v1.json'
    save_file_pathA = './datasets/hotpotqa_pre_processingA.jsonl'
    save_file_pathB = './datasets/hotpotqa_pre_processingB.jsonl'
    save_file_pathC = './datasets/hotpotqa_pre_processingC.jsonl'
    if not os.path.isfile(save_file_pathA):
        print('Pre processing group A hotpot qa...')
        pre_process_hotpot_qa(1500, 2500, read_file_path, save_file_pathA)
        print('group A hotpot qa all created.')
    else:
        print('The group A hotpot qa all already exists')

    if not os.path.isfile(save_file_pathB):
        print('Pre processing group B hotpot qa...')
        pre_process_hotpot_qa(3300, 6300, read_file_path, save_file_pathB)
        print('group B hotpot qa all created.')
    else:
        print('The group B hotpot qa all already exists')

    if not os.path.isfile(save_file_pathC):
        print('Pre processing group C hotpot qa...')
        pre_process_hotpot_qa(7000, 8000, read_file_path, save_file_pathC)
        print('group C hotpot qa all created.')
    else:
        print('The group C hotpot qa all already exists')    
