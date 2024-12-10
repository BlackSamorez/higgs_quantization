import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, HiggsConfig
from transformers.integrations.higgs import HiggsLinear
from transformers.quantizers.quantizer_higgs import get_num_sms_from_device

import wandb
from tqdm import tqdm, trange

import flute.utils

from .gptq.gptq import apply_higgs_gptq, get_accumulate_input_fn

DEV = torch.device('cuda')

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def replace_submodule(module, submodule_path, new_submodule):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names[:-1]:
        module = getattr(module, submodule)
    setattr(module, submodule_names[-1], new_submodule)


@torch.no_grad()
def llama_higgs_gptq(
    model: nn.Module, nsamples: int, dataloader, dev: torch.device,
    p: int, bits: int, hadamard_size: int, group_size: int,
    modules_to_not_convert: list[str] = None,
):
    flute_workspaces = {}
    
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    outs = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            outs.append(torch.zeros_like(inp))
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    layer_counter = 0
    for i in trange(len(layers), desc="Quantizing with GPTQ..."):
        layer = layers[i].to(dev)
        linear_layers = find_layers(layer)

        hessians = {name: None for name in linear_layers}
        num_samples = {name: 0 for name in linear_layers}
        handles = [
            linear_layers[name].register_forward_hook(
                get_accumulate_input_fn(name, hessians, num_samples)
            ) for name in linear_layers
        ]
        for j in trange(nsamples, leave=False, desc="Before pass..."):
            outs[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        for name, linear in linear_layers.items():
            if any(name.endswith(to_skip_name) for to_skip_name in modules_to_not_convert):
                print("Skipping", name)
                continue
            
            layer_counter += 1
            
            flute_dict = apply_higgs_gptq(
                linear.weight.data, 2 * hessians[name] / num_samples[name],
                p=p, bits=bits, hadamard_size=hadamard_size, group_size=group_size,
            )
                
            quantized_linear = HiggsLinear(
                linear.in_features, linear.out_features, bias=linear.bias is not None,
                num_bits=bits, dtype=dtype, device=dev,
                group_size=group_size, hadamard_size=hadamard_size,
            )
            
            for key, value in flute_dict.items():
                if key in quantized_linear._parameters:
                    quantized_linear._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
                elif key in quantized_linear._buffers:
                    quantized_linear._buffers[key] = torch.nn.Buffer(value)
                else:
                    raise ValueError(f"Unexpected key {key} in module {quantized_linear}")

            quantized_linear.num_sms_packed = torch.nn.Parameter(
                torch.tensor(get_num_sms_from_device(dev), device=dev, dtype=torch.int32),
                requires_grad=False,
            )
            
            if quantized_linear.weight.device not in flute_workspaces:
                flute_workspaces[quantized_linear.weight.device] = flute.utils.make_workspace_streamk(
                    device=quantized_linear.weight.device
                )
            quantized_linear.workspace = flute_workspaces[quantized_linear.weight.device]
            
            replace_submodule(layer, name, quantized_linear)

        mse = 0
        norm = 0
        for j in trange(nsamples, leave=False, desc="After pass..."):
            out = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
            mse += torch.nn.functional.mse_loss(outs[j][0], out[0]).item()
            norm += outs[j][0].float().pow(2).mean().item()
            inps[j] = out
        wandb.log({"block_mse": mse, "block_rmse": mse / norm, "block_id": i})

        if any([inp.isnan().any() for inp in inps]):
            raise Exception("NaNs!")
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        
    assert layer_counter == 7 * 32

    model.config.use_cache = use_cache
    return model
        

@torch.no_grad()
def llama_eval(model, dataloader, dev):
    print('Evaluating ...')

    nsamples = len(dataloader) 

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    for i in trange(len(layers), desc=f"Evaluating layer-by-layer..."):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    
    return ppl.item()


def get_zero_shots(model, task_list = ('arc_easy',), num_fewshots=1):
    import lm_eval

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
    )

    tasks = lm_eval.tasks.get_task_dict(task_list)
    if num_fewshots != 1:
        # TODO: make fewshots properly
        for task_name in tasks:
            task = tasks[task_name]
            if isinstance(task, tuple):
                task = task[1]
            if task is None:
                continue
            task.config.num_fewshot = num_fewshots

    results = lm_eval.evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=tasks,
    )

    result_dict = {task_name: task_result['acc,none'] for task_name, task_result in results['results'].items()}
    result_err_dict = {f'{task_name}_err': task_result['acc_stderr,none'] for task_name, task_result in
                       results['results'].items()}
    result_dict = dict(list(result_dict.items()) + list(result_err_dict.items()))

    if num_fewshots != 1:
        result_dict = {f'{task_name}@{num_fewshots}': acc for task_name, acc in result_dict.items()}

    return result_dict


if __name__ == '__main__':
    import argparse
    from .utils.datautils import get_loaders

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--p', type=int, default=2,
        help='HIGGS grid dimension'
    )
    parser.add_argument(
        '--bits', type=int, default=4,
        help='HIGGS bits'
    )
    parser.add_argument(
        '--hadamard_size', type=int, default=512, choices=[64, 128, 256, 512, 1024, 2048, 4096],
        help='Groupsize to use for hadamard; default is 512.'
    )
    parser.add_argument(
        '--group_size', type=int, default=256, choices=[64, 128, 256],
        help='Groupsize to use for scaling; default is 256.'
    )
    parser.add_argument(
        '--modules_to_not_convert', 
        type=str, 
        nargs='+', 
        default=["lm_head"], 
        help="List of linear layers that should not be quantized."
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--seqlen',
        type=int, default=8192, help='Seq len for PPL evals.'
    )
    parser.add_argument(
        '--method', type=str, choices=["gptq"], default="gptq", help="Method to quantize with",
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--push_to_hub', type=str, default=None,
        help='Push the model to the hub.'
    )
    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="higgs-gptq-lib",
        
        # track hyperparameters and run metadata
        config=args,
        name=f"{args.model}-HIGGS-{args.method.upper()}-{args.p}d-{args.bits}bit",
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()
    
    match args.method:
        case "gptq":
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            model = llama_higgs_gptq(
                model, args.nsamples, dataloader, DEV,
                args.p, args.bits, args.hadamard_size, args.group_size,
                modules_to_not_convert = args.modules_to_not_convert,
            )
        case _:
            raise Exception("AAA")
        
    # Add quantization metadata
    model.config.quantization_config = HiggsConfig(
        bits=args.bits, p=args.p, hadamard_size=args.hadamard_size, group_size=args.group_size,
        modules_to_not_convert=args.modules_to_not_convert,
    )
    model.is_quantized = True

    # Eval PPL to test the model
    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        ppl = llama_eval(model, testloader, DEV)
        wandb.log({f"{dataset}_PPL": ppl})

    # Maybe push to hub
    if args.push_to_hub is not None:
        model.push_to_hub(args.push_to_hub)

    # Heavier evals
    model = model.to(DEV)
    model = model.dequantize()
    wandb.log(get_zero_shots(model, task_list = ['mmlu'], num_fewshots=5))
    wandb.log(get_zero_shots(model, task_list = ('winogrande','piqa','arc_easy','arc_challenge'), num_fewshots=1))
