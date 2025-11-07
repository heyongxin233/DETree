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
    'outfox': 'outfox',
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

# templates = ['Please refine the following paragraph to improve its flow and clarity. Ensure that the original meaning and structure are preserved, while enhancing sentence construction and expression for better readability:\n{}',
#           'Please refine the sentences in the following paragraph to improve their fluency and clarity. Ensure that the overall content and structure remain unchanged. The focus should be on enhancing sentence construction and expression, ensuring the text flows smoothly and conveys information clearly and accurately:Input Text:\n{}',
#           'Kindly optimize the sentences in the following paragraph to improve readability and coherence. Do not make any changes to the main content or structure of the text. Concentrate on refining the sentence construction and expression to ensure the ideas are presented clearly and logically:Input Text:\n{}',
#           'Please enhance the fluency and clarity of the sentences in the paragraph below. Keep the overall content and structure intact, but focus on optimizing the construction and expression of the sentences to ensure that the text reads smoothly and conveys the intended information accurately:Input Text:\n{}',
#           'Examine the following text and make adjustments to improve the fluency and clarity of the sentences. Do not alter the structure or content of the paragraph. The goal is to improve the expression and flow of the sentences, ensuring the ideas are conveyed clearly and effectively:Input Text:\n{}',
#           'Please analyze the following text for spelling and grammatical inaccuracies, ensuring that any repetitive or improperly chosen words are replaced. Do not make any changes to the sentence order or structure. The goal is to enhance the precision and clarity of the language, maintaining the original sentence framework:Input Text:\n{}',
#           'Review the paragraph below and identify any spelling and grammatical errors. Replace any words that seem redundant, unclear, or incorrectly chosen. The sentence structure must remain intact, with changes being limited only to word choices to improve readability and appropriateness:Input Text:\n{}',
#           'Please optimize the sentences in the following paragraph to enhance fluency and clarity. Do not alter the overall content or structure of the paragraph. Focus on the construction and expression of the sentences, ensuring that the text is coherent and the information is accurate:Input Text:\n{}',
#           'Please polish the following text to make the language more fluent and cohesive, ensuring grammatical accuracy and enhancing the elegance and professionalism of expression:Input Text:\n{}',
#           'Optimize the following text to make sentence structures more varied, enrich vocabulary, and improve readability and appeal:Input Text:\n{}',
#           'Enhance the following text to elevate its expressive quality, add a literary touch, and retain its original meaning:Input Text:\n{}',
#           'Polish the following text to add emotional depth and vivid imagery, with the hallmark of creative writing:Input Text:\n{}',
#           'Reorganize the structure of the following text to make its logic clearer and its flow more coherent:Input Text:\n{}',]

templates = ['Please revise the following paragraph to enhance its fluency and coherence. Focus on improving the transitions between sentences, reinforcing the core argument, and eliminating any redundant or unnecessary content. The goal is to refine the expression and sentence structure, ensuring clarity and precision while maintaining the original meaning and overall structure. The revision should make the text more concise, logical, and engaging.Input Text:\n{}',
            'Please adjust the language style of the following paragraph to make it more informal. Maintain the core meaning and structure while ensuring that the tone aligns with a more casual audience.Input Text:\n{}']

def truncate_text(text,tokenizer_ ,max_length=1024):
    
    tokens = tokenizer_.encode(text, truncation=True, max_length=max_length)
    truncated_text = tokenizer_.decode(tokens, skip_special_tokens=True)
    return truncated_text

def gen_polish(data):
    tokenizer_ = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', trust_remote_code=True, max_length=1024, truncation=True)
    prompts = []
    for item in data:
        now_prompt = random.choice(templates)
        text = truncate_text(now_prompt.format(item['text']),tokenizer_)+'\nPlease output the polished text directly.\nOutput Text:\n'
        prompts.append(text)
    output_text = []
    outputs = llm.generate(prompts, sampling_params)
    for i,output in enumerate(outputs):
        now_item = data[i]
        generated_text = output.outputs[0].text
        id = stable_long_hash(generated_text)
        source_id = now_item['id'] if now_item.get('adv_source_id','')=='' else now_item['adv_source_id']
        output_text.append({'text':generated_text,'label':now_item['label'],'src':now_item['src']+'_polish_'+call_name,'id':id,'adv_source_id':source_id})
    
    return output_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--dataset", type=str, default='OUTFOX')
    parser.add_argument("--model_name", type=str, default='deepseek-ai/DeepSeek-V2-Lite')
    parser.add_argument("--call_name", type=str, default='deepseekv2')
    args = parser.parse_args()
    mode=args.mode
    dataset=args.dataset
    call_name = args.call_name
    device_num = args.device_num
    data=load_jsonl(f'/path/to/RealBench/{dataset}/no_attack/{mode}.jsonl')
    print('loading ',f'/path/to/RealBench/{dataset}/no_attack/{mode}.jsonl')

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
    elif 'DeepSeek' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=2
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=0.3,top_p=0.95)
    elif 'gemma' in model_name:
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=1024,temperature=1.0)

    output_text = gen_polish(data)
    if os.path.exists(f"/path/to/RealBench/{dataset}/polish")==False:
        os.makedirs(f"/path/to/RealBench/{dataset}/polish")
    target_path = f"/path/to/RealBench/{dataset}/polish/{mode}.jsonl"
    with open(target_path, mode='a+', encoding='utf-8') as jsonl_file:
        for item in output_text:
            jsonl_file.write(json.dumps(item,ensure_ascii=False)+'\n')

    


