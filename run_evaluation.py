import argparse
import os
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

def open_and_parse_text_info(path:str):
    with open(path, 'r') as f:
        text = f.read().split('\n')
    full_text = text[0].split('full: ')[1]
    target = text[1].split('target: ')[1]
    target = [el.strip() for el in target.split('|')]
    separators = text[2].split('separators: ')[1].split(',')
    separators = [el.strip() for el in separators]
    return {
        'full_text': full_text,
        'target': target,
        'separators': separators
    }

def main(pipeline, args:dict):
    print('Running evaluation')

    files = sorted(os.listdir("./data/"), key=lambda x: int(x))
    correct_without_repeat = 0
    correct_in_clear_repeat = 0
    correct_in_corrupt_repeat = 0
    for file in files:
        path_with_repeat = os.path.join("./data/", file, 'sentence_with_repeat.wav')
        path_without_repeat = os.path.join("./data/", file, 'sentence_without_repeat.wav')
        
        out_with_repeat = normalize(pipeline(path_with_repeat))
        out_without_repeat = normalize(pipeline(path_without_repeat))
     

        text_info = open_and_parse_text_info(os.path.join("./data/", file, 'text.txt'))

        for target in text_info['target']:
            if target in out_without_repeat:
                correct_without_repeat += 1
                print(f'Target {target} found without repeat: {out_without_repeat}')
                break
        
        for i in range(len(text_info['separators'])):
            separator = text_info['separators'][i]
            if separator in out_with_repeat:
                split = out_with_repeat.split(separator)
                before, after = split[0], split[1]
                break
            elif i == len(text_info['separators']) - 1:
                print(f'Separator {separator} not found in output - {out_with_repeat}')
                before, after = None, None
        
        if before is not None:
            for target in text_info['target']:
                if target in before:
                    correct_in_corrupt_repeat += 1
                    print(f'Target {target} found in corrupt repeat: {before}')
                    break
        
        if after is not None:
            for target in text_info['target']:
                if target in after:
                    correct_in_clear_repeat += 1 
                    break 
                else:
                    print(f'Target {target} not found in clear repeat: {after}')    
            

        print(file,'----\n')
        
    correct_in_clear_repeat = (correct_in_clear_repeat / len(files)) * 100
    correct_in_corrupt_repeat = (correct_in_corrupt_repeat / len(files)) * 100
    correct_without_repeat = (correct_without_repeat / len(files)) * 100
    print(f'Correct in clear repeat: {correct_in_clear_repeat}\nCorrect in corrupt repeat: {correct_in_corrupt_repeat}\nCorrect without repeat: {correct_without_repeat}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lcasr', 'whisper', 'canary'], default='lcasr')
    main_args,_ = parser.parse_known_args()

    if main_args.model == 'lcasr':
        from models.lcasr.load_model import get_args, load
        model_args = get_args()
        pipeline = load(model_args)
    elif main_args.model == 'whisper':
        from models.whisper.load_model import get_args, load
        model_args = get_args()
        pipeline = load(model_args)
    elif main_args.model == 'canary':
        from models.canary.load_model import get_args, load
        model_args = get_args()
        pipeline = load(model_args)
    
    all_args = {**vars(main_args), **vars(model_args)}

    main(pipeline, all_args)