"""Inference for FastChat models."""
import abc
from typing import Optional
import warnings
import os,json,csv
import torch

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        LlamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        LLamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )

from model.fastchat.conversation import (
    conv_templates,
    get_default_conv_template,
    compute_skip_echo_len,
    SeparatorStyle,
)
from model.fastchat.serve.compression import compress_module
from model.fastchat.serve.monkey_patch_non_inplace import (
    replace_llama_attn_with_non_inplace_operations,
)
from model.fastchat.serve.serve_chatglm import chatglm_generate_stream


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower():
        try:
            is_vicuna = isinstance(model, LlamaForCausalLM)
        except Exception:
            is_vicuna = isinstance(model, LLamaForCausalLM)
        if is_vicuna and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fschat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(
    model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, debug=False
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        ).cuda()
    elif "google/flan-t5" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        raise_warning_for_old_weights(model_path, model)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 32))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)
    # print("token len:", len(input_ids)) ## TODO
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: str,
    max_gpu_memory: str,
    load_8bit: bool,
    conv_template,
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_template.copy() 
    else:
        conv = get_default_conv_template(model_path).copy()

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset :]
            generate_stream_func = chatglm_generate_stream
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        
        skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)
        stop_str = (
            conv.sep
            if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE]
            else None
        )

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

def question_loop(
    model_path: str,
    device: str,
    num_gpus: str,
    max_gpu_memory: str,
    load_8bit: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
    prompt_caption: dict = None,
    prompt_caption_path: str = None,
    output_path: str = None,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()
        
    # Question
    if prompt_caption:
        questions = prompt_caption
    elif not prompt_caption and prompt_caption_path:
        with open(prompt_caption_path, 'r') as f:
            questions = json.load(f)
    else:
        raise ValueError("prompt_caption or prompt_caption_path must be provided")
    
    

    captions = {}
    for id,question in questions.items():
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset :]
            generate_stream_func = chatglm_generate_stream
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)
        stop_str = (
            conv.sep
            if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE]
            else None
        )

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        captions[id] = outputs
        # clear conv for next question
        del conv
        conv = get_default_conv_template(model_path).copy()
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    with open(output_path, 'w') as f:
        json.dump(captions, f)
    print(captions)
    return captions

def get_test(file_path):
    data_info = dict()
    # if data_info exists, load it
    if os.path.exists('data_info.json'):
        print("data info exists, loading...")
        with open('data_info.json', 'r') as fp:
            data_info = json.load(fp)
        return data_info
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip the first row
        next(reader)
        for row in reader: 
            # num,key,question,answer,vid_id,gif_name,description
            if row[3] == '' or row[3] not in ['yes', 'no']:
                continue
            video = row[4]
            try:
                data_info[video]['questions'][row[1]] = row[2]
                data_info[video]['answers'][row[1]] = row[3]
            except:
                data_info[video] = dict()
                data_info[video]['questions'] = dict()
                data_info[video]['answers'] = dict()
                data_info[video]['infer'] = dict() ### empty dict for inference results
                data_info[video]['questions'][row[1]] = row[2]
                data_info[video]['answers'][row[1]] = row[3]
    with open('data_info.json', 'w') as fp:
        json.dump(data_info, fp)
    return data_info

def answer_loop(
    model_path: str,
    device: str,
    num_gpus: str,
    max_gpu_memory: str,
    load_8bit: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
    prompt_caption: dict = None,
    prompt_caption_path: str = None,
    output_path: str = None,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()
        
    # Question
    if os.path.exists(answer_path):
        with open(answer_path, 'r') as f:
            import json
            print("answer file"+ str(answer_path) + "exists, loading...")
            data = json.load(f)
    else:
        print("loading origin data info...")
        data = get_test(data_info_path)
    
    if question_path and caption_path:
        import json
        with open(question_path, 'r') as f:
            questions = json.load(f)
    
    

    for id,prompted_cap in questions.items():
        # single loop for one video
        captions = {}
        qid_list = []
        question_list = []
        global_counter = 0
        counter = 0
        question_batch_size = 10
        for qid, question in data[id]['questions'].items():
            global_counter += 1
            counter += 1
            qid_list.append(qid)
            question_list.append(question)
            prompted_questions = ''
            # if it's the last step of the loop, set the batch size to the counter
            if global_counter == len(data[id]['questions']):
                question_batch_size = counter
                
            if counter == question_batch_size:
                for i in range(len(qid_list)):
                    prompted_questions += 'Question ' + str(i) + '. ' + question_list[i] + '\n' 
                print(prompted_cap+prompted_questions)
                conv.append_message(conv.roles[0], prompted_cap+prompted_questions)
                conv.append_message(conv.roles[1], None)

                if is_chatglm:
                    prompt = conv.messages[conv.offset :]
                    generate_stream_func = chatglm_generate_stream
                else:
                    generate_stream_func = generate_stream
                    prompt = conv.get_prompt()

                skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)
                stop_str = (
                    conv.sep
                    if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE]
                    else None
                )

                params = {
                    "model": model_path,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "stop": stop_str,
                }

                chatio.prompt_for_output(conv.roles[1])
                output_stream = generate_stream_func(model, tokenizer, params, device)
                outputs = chatio.stream_output(output_stream, skip_echo_len)
                if question_batch_size == 1:
                    data[id]['infer'][qid_list[0]] = outputs
                else:
                    output = outputs.split('\n')
                    print(output)
                    for i in range(len(qid_list)):  
                        try:
                            data[id]['infer'][qid_list[i]] = output[i][3:] # remove the index
                            print(output[i][3:])
                        except Exception as e:
                            # save to file of current video name and exception question id
                            print("error")
                            with open("error_info.txt", 'a') as f:
                                f.write(id + ':'+'\n')
                                f.write(str(e))
                                f.write('\n')
                            raise Exception("error")
                captions[id] = outputs
                # clear conv for next question
                del conv
                counter = 0
                qid_list = []
                question_list = []
                conv = get_default_conv_template(model_path).copy()
                if debug:
                    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        with open(caption_path, 'w') as f:
            json.dump(captions, f)
        with open(answer_path, 'w') as f:
            json.dump(data, f)