from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, regex
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import copy
import wandb  # Import wandb for logging
from copy import deepcopy
from DeepSeekMath.evaluation.data_processing.answer_extraction import (
    extract_math_answer,
)
from DeepSeekMath.evaluation.eval.eval_script import (
    is_correct,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/data2/Qwen/Qwen2.5-1.5B-Instruct"
gen_device = 4    # GPU device for generation; don't include in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 10000
Q_batch_size = 2
num_pre_Q = 2
train_batch_size = 2
gen_update_steps = 16
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"
aux_reward_weight = 1  # Weight for the auxiliary reward component
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': 
            return None
    except:
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5:
        data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = []  # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch, frozen_model):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    # Extract meets_criteria flags for each rollout
    if "meets_criteria" in batch:
        # Convert to tensor of shape [batch_size]
        meets_criteria = torch.tensor(batch["meets_criteria"], dtype=torch.float32, device=engine.device)
        # Reshape to [batch_size, 1] for proper broadcasting
        meets_criteria = meets_criteria.unsqueeze(1)
    else:
        meets_criteria = torch.ones_like(advantages)
    
    # Get original unnormalized rewards for logging
    original_rewards = batch.get("original_rewards", None)
    
    # Simple debug print at beginning
    if dist.get_rank() == 0:
        print(f"\nGRPO_step: batch size={inputs.shape[0]}, rollouts meeting criteria={meets_criteria.sum().item()}/{meets_criteria.numel()}")
        if original_rewards:
            print(f"Original rewards: {original_rewards}")
            print(f"Criteria flags: {batch.get('meets_criteria', [])}")
    
    # Get logits from current model
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps_completion = per_token_logps[:,prompt_length-1:]
    
    # Get frozen model logits for aux reward
    with torch.no_grad():
        frozen_logits = frozen_model(inputs).logits[:, :-1, :]
        frozen_per_token_logps = get_per_token_logps(frozen_logits, input_ids)
        frozen_per_token_logps_completion = frozen_per_token_logps[:,prompt_length-1:]
    
    # Calculate auxiliary reward
    token_aux_rewards = -frozen_per_token_logps_completion
    print(f"frozen model loss: {frozen_per_token_logps_completion.mean().item():.4f}")
    # Add clipping to limit auxiliary reward magnitude
    token_aux_rewards = torch.clamp(token_aux_rewards, min=-3.0, max=3.0)
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # Calculate base model loss in nats
    base_model_loss = -(per_token_logps_completion * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    base_model_loss = base_model_loss.mean().item()  # Convert to scalar
    
    # Apply aux reward on a per-rollout basis using the meets_criteria flag
    # Expand to match token dimension
    meets_criteria_expanded = meets_criteria.expand_as(token_aux_rewards)
    
    # Only apply aux reward to tokens from rollouts that meet criteria
    token_aux_rewards = token_aux_rewards * meets_criteria_expanded
    
    # Rest of the GRPO logic (similar to original)
    ref_per_token_logps = batch['refs'].to(per_token_logps_completion.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps_completion) - (ref_per_token_logps - per_token_logps_completion) - 1
    
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps_completion - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        token_advantages = advantages.expand_as(per_token_logps_completion)
        token_advantages = token_advantages + (aux_reward_weight * token_aux_rewards)
        per_token_loss = torch.min(ratio * token_advantages, clipped_ratio * token_advantages)
    else: 
        token_advantages = advantages.expand_as(per_token_logps_completion)
        token_advantages = token_advantages + (aux_reward_weight * token_aux_rewards)
        per_token_loss = torch.exp(per_token_logps_completion - per_token_logps_completion.detach()) * token_advantages
    
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Calculate reward statistics for logging
    masked_aux = token_aux_rewards * completion_mask
    affected = (masked_aux > 0).sum().item()
    total = completion_mask.sum().item()
    avg_aux_reward = masked_aux.sum().item() / max(total, 1)
    
    # Log detailed reward information including per-rollout stats
    if dist.get_rank() == 0:
        print(f">>> Aux Reward: {affected}/{total} tokens affected. Avg={avg_aux_reward:.4f}")
        
        # Calculate per-rollout metrics
        for i in range(inputs.shape[0]):
            rollout_mask = completion_mask[i]
            rollout_aux = masked_aux[i]
            rollout_meets_criteria = meets_criteria[i].item()
            
            tokens_in_rollout = rollout_mask.sum().item()
            affected_tokens = (rollout_aux > 0).sum().item()
            
            if tokens_in_rollout > 0:
                rollout_avg_aux = rollout_aux.sum().item() / tokens_in_rollout
                print(f"  Rollout {i}: meets_criteria={rollout_meets_criteria}, "
                     f"affected_tokens={affected_tokens}/{tokens_in_rollout}, "
                     f"avg_aux={rollout_avg_aux:.4f}, "
                     f"reward={advantages[i].item():.4f}")
        
        # Log to wandb
        if wandb.run is not None:
            # Log metrics to wandb
            wandb_metrics = {
                "loss": loss.item(),
                "base_model_loss_nats": base_model_loss,  # Add base model loss in nats
                "avg_aux_reward": avg_aux_reward,
                "tokens_with_aux_reward": affected / max(total, 1),
                "rollouts_meeting_criteria": meets_criteria.sum().item() / meets_criteria.numel()
            }
            
            # Log original rewards if available
            if original_rewards is not None:
                # Map criteria flags to rewards for separate tracking
                criteria_flags = batch.get('meets_criteria', [])
                good_rewards = [r for r, f in zip(original_rewards, criteria_flags) if f > 0.5]
                bad_rewards = [r for r, f in zip(original_rewards, criteria_flags) if f <= 0.5]
                
                avg_original_reward = sum(original_rewards) / len(original_rewards)
                avg_good_reward = sum(good_rewards) / max(len(good_rewards), 1)
                avg_bad_reward = sum(bad_rewards) / max(len(bad_rewards), 1)
                
                wandb_metrics.update({
                    "avg_original_reward": avg_original_reward,
                    "avg_reward_good_rollouts": avg_good_reward,
                    "avg_reward_bad_rollouts": avg_bad_reward,
                    "good_rollouts_pct": len(good_rewards) / len(original_rewards)
                })
                
                print(f">>> Average original reward: {avg_original_reward:.4f}")
                print(f">>> Good rollouts: {len(good_rewards)}/{len(original_rewards)}, avg reward: {avg_good_reward:.4f}")
                print(f">>> Bad rollouts: {len(bad_rewards)}/{len(original_rewards)}, avg reward: {avg_bad_reward:.4f}")
            
            wandb.log(wandb_metrics)
    
    # Print per-token stats (as before)
    # Calculate masked values for better visibility
    masked_loss = per_token_loss * completion_mask
    masked_advantages = token_advantages * completion_mask
    
    # Print statistics for each element in batch (limit to first example to avoid too much output)
    for i in range(min(1, inputs.shape[0])):
        nonzero_mask = completion_mask[i].bool()
        if nonzero_mask.sum() > 0:
            print(f"\n--- Per-token statistics for batch item {i} ---")
            # Get values only for non-padding tokens
            nonzero_loss = masked_loss[i][nonzero_mask]
            nonzero_advantages = masked_advantages[i][nonzero_mask]
            nonzero_aux_rewards = masked_aux[i][nonzero_mask]
            
            # Print basic statistics
            print(f"Token loss: min={nonzero_loss.min().item():.4f}, max={nonzero_loss.max().item():.4f}, mean={nonzero_loss.mean().item():.4f}")
            print(f"Token advantages: min={nonzero_advantages.min().item():.4f}, max={nonzero_advantages.max().item():.4f}, mean={nonzero_advantages.mean().item():.4f}")
            print(f"Aux rewards: min={nonzero_aux_rewards.min().item():.4f}, max={nonzero_aux_rewards.max().item():.4f}, mean={nonzero_aux_rewards.mean().item():.4f}")
            
            # Print a sample of token values (first 5 tokens)
            sample_size = min(5, nonzero_mask.sum().item())
            print(f"\nSample of first {sample_size} tokens:")
            for j in range(sample_size):
                token_idx = torch.nonzero(nonzero_mask)[j].item()
                print(f"  Token {j}: loss={masked_loss[i][token_idx].item():.4f}, advantage={masked_advantages[i][token_idx].item():.4f}, aux_reward={masked_aux[i][token_idx].item():.4f}")

    return loss

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5, dtype="float16")
    ref_server_ver = 'tensor'  # auto switch based on first upload

    sampling_params = SamplingParams(
        n=num_pre_Q,
        temperature=0.9,
        max_tokens=1024,  # Increased from 700
        top_p=0.95
    )
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    from datasets import load_dataset
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    dataset = dataset.filter(lambda x: x['source'] in ['orca_math'])
    QAs = [{'Q': x['problem'], 'A': extract_math_answer(x['problem'], x['solution'], "math")} for x in dataset]
    
    system_prompt = (
        "You are a helpful mathematics assistant. A conversation between User and Assistant. "
        "The user presents a mathematics problem, and you solve it step by step."
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
        "tags, respectively."
    )
    
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = []
        ans_token_ids = []
        for v in voutputs:
            for z in v.outputs:
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)

        with open("generation_log.txt", "a") as f:
            for i, ans in enumerate(answers[:3]):
                f.write(f"[Answer {i}]:\n{ans}\n{'-'*40}\n")
        return answers, ans_token_ids

    def reward_correct(item, answer):
        # Extract the answer from the model's response
        extracted_answers = extract_math_answer(item["Q"], answer, "math")
        
        # If no answer was extracted, return negative reward
        if not extracted_answers:
            return -1.0
        
        # Prepare item for evaluation
        eval_item = {
            'prediction': extracted_answers,
            'answer': item["A"]
        }
        
        # Use official evaluation logic
        try:
            is_answer_correct = is_correct(eval_item, pred_key='prediction', prec=1e-3)
            return 1.0 if is_answer_correct else -1.0
        except:
            return -1.0

    def reward_format(item, answer):
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else -1

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        rewards = []
        is_correct_list = []  # Track correctness
        is_formatted_list = []  # Track formatting
        meets_criteria_list = []  # Track if both criteria are met

        print(f"\n===== Generated Samples (step {it}) =====")
        for i, ans in enumerate(answers[:3]):
            print(f"[Answer {i}]")
            print(ans)
            print("-" * 40)
        
        for i, inp in enumerate(inputs):
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                # Evaluate correctness
                correctness = reward_correct(inp, a)
                is_correct = 1.0 if correctness > 0 else 0.0
                is_correct_list.append(is_correct)
                
                # Evaluate formatting
                format_reward = reward_format(inp, a)
                is_formatted = 1.0 if format_reward > 0 else 0.0
                is_formatted_list.append(is_formatted)
                
                # Check if both criteria are met (per rollout)
                meets_criteria = 1.0 if (is_correct > 0.5 and is_formatted > 0.5) else 0.0
                meets_criteria_list.append(meets_criteria)
                
                # Total reward is still the sum
                rewards.append(correctness + format_reward)
        
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids, meets_criteria_list

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            return

    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0:
            try_update_model()
        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        print("about to generate samples")
        prompt_inputs, rewards, answers, ans_token_ids, meets_criteria_list = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
        if it % 5 == 0:
            print('answers:', answers[0])
        for i, pp in enumerate(prompt_inputs):
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_is_correct = meets_criteria_list[i*num_pre_Q:(i+1)*num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4:
                continue
            if ref_server_ver == 'tensor':
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                
                # Keep track of which samples meet criteria (per rollout)
                curr_meets_criteria = meets_criteria_list[i*num_pre_Q:(i+1)*num_pre_Q]
                
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    
                    # Get criteria flags for this batch slice
                    sub_meets_criteria = curr_meets_criteria[ii:ii+train_batch_size]
                    
                    # Build the configuration JSON with prompt length and per-rollout criteria flags
                    config_json = json.dumps({
                        "plen": plen, 
                        "meets_criteria": sub_meets_criteria,
                        "original_rewards": sub_rewards.cpu().tolist()
                    }).encode()
                    
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    
                    # Package data with correctness flags in JSON
                    if compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data = [
                            config_json,
                            tensor_to_bytes(merged_ids), 
                            tensor_to_bytes(sub_rewards),
                            tensor_to_bytes(gen_logps)
                        ]
                    else:
                        data = [
                            config_json,
                            tensor_to_bytes(merged_ids), 
                            tensor_to_bytes(sub_rewards)
                        ]

                    # Rest of the code same as before
                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([
                    json.dumps({"Q": pp[0], "As": curr_answers}).encode(),
                    tensor_to_bytes(curr_rewards),
                    tensor_to_bytes(curr_is_correct)
                ])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor':
                    ref_server_ver = 'tensor'

tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    print('\nSTART vLLM generation...\n')
    print(f'Using auxiliary reward weight: {aux_reward_weight}')
    print(f'This training run WILL apply token-wise auxiliary rewards based on frozen model loss')
    print(f'Auxiliary rewards are applied on a per-rollout basis to answers that are both CORRECT AND PROPERLY FORMATTED')
    
    # Initialize wandb if rank is 0
    if dist.get_rank() == 0:
        wandb.init(
            project="grpo_training", 
            name=f"grpo_aux_reward_{aux_reward_weight}",
            config={
                "model_path": model_path,
                "beta": beta,
                "aux_reward_weight": aux_reward_weight,
                "clip_param": clip_param,
                "all_steps": all_steps,
                "per_rollout_gating": True
            }
        )
        print("Initialized wandb for tracking metrics")
    
    mp.set_start_method('spawn')
    Q = mp.Queue()
    p = mp.Process(target=gen_worker, args=(Q, gen_device))
    p.start()

    # Initialize the main model for training
    model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float16, _attn_implementation="sdpa")
    
    # Initialize a frozen copy of the original model for auxiliary reward calculation
    frozen_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float16, _attn_implementation="sdpa")
    for param in frozen_model.parameters():
        param.requires_grad = False
    frozen_model = frozen_model.eval().to(torch.device(f"cuda:{dist.get_rank()}"))

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model,
                                                   model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0:
        progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss = GRPO_step(batch, frozen_model)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            
            # Log step info to wandb
            if wandb.run is not None:
                wandb.log({"step": step, "training_loss": loss.item()})
        
        # Add periodic step summary
        if step % 50 == 0 and dist.get_rank() == 0:
            print(f"\n===== Step {step} Summary =====")
            print(f"Current loss: {loss.item():.6f}")
            print(f"Auxiliary reward weight: {aux_reward_weight}")
            print(f"Progress: {step}/{all_steps} steps ({100*step/all_steps:.1f}%)")
            print("=============================\n")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()

