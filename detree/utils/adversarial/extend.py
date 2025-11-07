import argparse
import hashlib
import math
import os
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import random
model_alias_mapping = {
    'chatgpt': 'chatgpt',
    'ChatGPT': 'chatgpt',
    'chatGPT': 'chatgpt',
    'gpt-3.5-trubo': 'gpt-3.5-trubo',
    'GPT4': 'gpt4',
    'gpt4': 'gpt4',
    'text-davinci-002': 'text-davinci-002',
    'text-davinci-003': 'text-davinci-003',
    'davinci': 'text-davinci',
    'gpt1': 'gpt1',
    'gpt2_pytorch': 'gpt2-pytorch',
    'gpt2_large': 'gpt2-large',
    'gpt2_small': 'gpt2-small',
    'gpt2_medium': 'gpt2-medium',
    'gpt2-xl': 'gpt2-xl',
    'GPT2-XL': 'gpt2-xl',
    'gpt2_xl': 'gpt2-xl',
    'gpt2': 'gpt2-xl',
    'gpt3': 'gpt3',
    'GROVER_base': 'grover_base',
    'grover_base': 'grover_base',
    'grover_large': 'grover_large',
    'grover_mega': 'grover_mega',
    'llama2-fine-tuned': 'llama2',
    'opt_125m': 'opt_125m',
    'opt_1.3b': 'opt_1.3b',
    'opt_2.7b': 'opt_2.7b',
    'opt_6.7b': 'opt_6.7b',
    'opt_13b': 'opt_13b',
    'opt_30b': 'opt_30b',
    'opt_350m': 'opt_350m',
    'opt_iml_max_1.3b': 'opt_iml_max_1.3b',
    'opt_iml_30b': 'opt_iml_30b',
    'flan_t5_small': 'flan_t5_small',
    'flan_t5_base': 'flan_t5_base',
    'flan_t5_large': 'flan_t5_large',
    'flan_t5_xl': 'flan_t5_xl',
    'flan_t5_xxl': 'flan_t5_xxl',
    'flan_t5': 'flan_t5_xxl',
    'dolly': 'dolly',
    'GLM130B': 'GLM130B',
    'bloom_7b': 'bloom_7b',
    'bloomz': 'bloomz',
    't0_3b': 't0_3b',
    't0_11b': 't0_11b',
    'gpt_neox': 'gpt_neox',
    'xlm': 'xlm',
    'xlnet_large': 'xlnet_large',
    'xlnet_base': 'xlnet_base',
    'cohere': 'cohere',
    'ctrl': 'ctrl',
    'pplm_gpt2': 'pplm_gpt2',
    'pplm_distil': 'pplm_distil',
    'fair_wmt19': 'fair_wmt19',
    'fair_wmt20': 'fair_wmt20',
    'glm130b': 'GLM130B',
    'jais-30b': 'jais',
    'transfo_xl': 'transfo_xl',
    '7B': '7B',
    '13B': '13B',
    '65B': '65B',
    '30B': '30B',
    'gpt_j': 'gpt_j',
    'mpt': 'mpt',
    'mpt-chat': 'mpt-chat',
    'llama-chat': 'llama-chat',
    'mistral': 'mistral',
    'mistral-chat': 'mistral-chat',
    'cohere-chat': 'cohere-chat',
    'human': 'human',
}


def load_jsonl(file_path):
    out = []
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            now = json.loads(line)
            now['src'] = model_alias_mapping[now['src']]
            out.append(now)
    random.seed(1)
    random.shuffle(out)
    return out

def stable_long_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = (int_hash & ((1 << 63) - 1))
    return long_long_hash

#train data gen templates
# templates = ['Here is a piece of text. Please continue writing from where it ends, maintaining the same tone, style, and context while making the continuation coherent and engaging.Input Text:\n{}',
#            'Please expand on the following text, continuing its ideas and maintaining a consistent tone and style. Ensure the expansion is coherent, logical, and enhances the original content.Input Text:\n{}',
#            'I have an incomplete text that I need to complete. Please expand it into a complete text that includes the original text I provided. The original text must keep its formatting (such as capitalization and punctuation) intact.Input Text:\n{}']

#test data gen  templates
templates = ["Please continue the following text, expanding on its ideas in a way that maintains a consistent tone and style. The expansion should be coherent, logically structured, and serve to enrich the original content. Avoid using transitional phrases such as 'firstly,' 'secondly,' or 'then.' Instead, opt for smoother transitions that flow naturally from one thought to the next. Use punctuation carefully, particularly minimizing the overuse of commas. Input Text:\n{}",
             "Please continue the following text, ensuring the expansion flows naturally and coherently. Build upon the original ideas, introducing new insights that are logically derived from the premises already established. Use smooth transitions between thoughts, avoiding rigid or formulaic structures. The writing should maintain a refined balance of clarity and elegance, with careful attention to punctuation—favoring periods and semicolons over excessive commas. Make sure the expansion complements and enhances the original tone, with a focus on preserving its spirit while adding depth.Input Text:\n{}",]


def truncate_text(text,tokenizer_ ,max_length=128):
    
    tokens = tokenizer_.encode(text)
    if len(tokens)//2 > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens[:len(tokens)//2]
    truncated_text = tokenizer_.decode(tokens, skip_special_tokens=True)
    return truncated_text

def gen_extend(data):
    tokenizer_ = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', trust_remote_code=True, max_length=1024, truncation=True)
    prompts = []
    for item in data:
        now_prompt = random.choice(templates)
        text = truncate_text(now_prompt.format(item['text']),tokenizer_)+'\nPlease include the content of the original text and continue to output it together directly.And do not have additional explanatory words, either at the beginning or at the end.\nOutput Text:\n'
        prompts.append(text)
    output_text = []
    outputs = llm.generate(prompts, sampling_params)
    for i,output in enumerate(outputs):
        now_item = data[i]
        generated_text = output.outputs[0].text
        id = stable_long_hash(generated_text)
        source_id = now_item['id'] if now_item.get('adv_source_id','')=='' else now_item['adv_source_id']
        output_text.append({'text':generated_text,'label':now_item['label'],'src':now_item['src']+'_extend_'+call_name,'id':id,'adv_source_id':source_id})
    
    return output_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=4)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--dataset", type=str, default='Deepfake')
    parser.add_argument(
        "--data-root",
        type=str,
        default="/path/to/RealBench",
        help="Root directory containing the RealBench-style dataset splits.",
    )
    parser.add_argument("--model_name", type=str, default='databricks/dolly-v2-12b')
    #internlm/internlm2_5-7b-chat
    #THUDM/glm-4-9b-chat
    #mistralai/Ministral-8B-Instruct-2410
    #meta-llama/Llama-3.1-8B-Instruct
    #allenai/OLMo-2-1124-7B-Instruct
    #google/gemma-2-9b
    parser.add_argument("--call_name", type=str, default='dollyv2')
    #internlm2_5_7b
    #glm4_9b
    #Ministral_8b
    #Llama_3.1_8B
    #olmo2_7b
    #gemma2_9b
    args = parser.parse_args()
    mode=args.mode
    dataset=args.dataset
    call_name = args.call_name
    device_num = args.device_num
    data_root = os.path.abspath(args.data_root)
    source_path = os.path.join(data_root, dataset, "no_attack", f"{mode}.jsonl")
    data = load_jsonl(source_path)
    print('loading ', source_path)

    each_len = math.ceil(len(data)/device_num)
    st = each_len * args.device
    ed = min(st + each_len, len(data))
    data = data[st:ed]
    # data=data[:5]
    print(f"device {args.device} start from {st} to {ed}",len(data))

    model_name = args.model_name
    # model_name = "mistralai/Ministral-8B-Instruct-2410"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_length=1024, truncation=True)
    if 'glm' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
            # enforce_eager=True,
            # Enable the following options if GLM-4-9B-Chat-1M runs out of memory
            # enable_chunked_prefill=True,
            # max_num_batched_tokens=8192
        )
        stop_token_ids = [151329, 151336, 151338]
        sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)
    elif 'Ministral' in model_name:
        llm = LLM(
            model=model_name,
            tokenizer_mode="mistral", 
            config_format="mistral", 
            load_format="mistral",
            max_model_len=2048,
        )
        sampling_params = SamplingParams(max_tokens=1024)
    elif 'Llama' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=0.6,top_p=0.9)
    elif 'OLMo' in model_name or 'internlm' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=0.9,top_k=50)
    elif 'Qwen' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=0.7,top_k=20,top_p=0.8,repetition_penalty=1.05)
    elif 'gemma' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=1.0)
    elif 'dolly' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=2
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=0.7)

    output_text = gen_extend(data)
    extend_dir = os.path.join(data_root, dataset, "extend")
    if not os.path.exists(extend_dir):
        os.makedirs(extend_dir)
    target_path = os.path.join(extend_dir, f"{mode}.jsonl")
    with open(target_path, mode='a+', encoding='utf-8') as jsonl_file:
        for item in output_text:
            jsonl_file.write(json.dumps(item,ensure_ascii=False)+'\n')

    


