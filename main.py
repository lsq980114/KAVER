import os
import gc
import torch
import logging
import argparse
import json
import psutil
from argparse import Namespace

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, BertTokenizer
from KG_CoT_Model.produce import find_path
from utils.dialogue_state import DialogueState

# Import local modules
from data.dataset import load_data, load_val_data, convert_example_to_graph_inputs
from utils.kg_utils import build_global_kg, get_global_entities, detect_start_entity_spacy, load_custom_entities
from utils.text_utils import create_improved_prompt, analyze_task_type, infer_date_from_now
from KG_CoT_Model.model import GraphReasoningModel
from run_training import train_kgcot_fixed as train_kgcot
from generation.enhanced_generator import EnhancedResponseGenerator


from training.adapter_finetuner import DomainAdapterFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_optimal_device():

    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s) available")
    
    if num_gpus == 1:
        device = torch.device("cuda:0")
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        return device

    best_gpu = 0
    max_memory = 0
    
    for i in range(num_gpus):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        gpu_name = torch.cuda.get_device_name(i)
        current_memory = torch.cuda.memory_allocated(i)
        free_memory = gpu_memory - current_memory
        
        logger.info(f"GPU {i}: {gpu_name}, Total: {gpu_memory/1e9:.1f}GB, "
                   f"Used: {current_memory/1e9:.1f}GB, Free: {free_memory/1e9:.1f}GB")
        
        if free_memory > max_memory:
            max_memory = free_memory
            best_gpu = i
    
    device = torch.device(f"cuda:{best_gpu}")
    logger.info(f"Selected GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)} "
               f"with {max_memory/1e9:.1f}GB free memory")
    
    return device

def check_device_compatibility(device):

    if device.type == "cuda":
        try:
            test_tensor = torch.randn(100, 100).to(device)
            _ = test_tensor @ test_tensor.T
            del test_tensor
            torch.cuda.empty_cache()
            
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            logger.info(f"Device {device} is compatible")
            logger.info(f"GPU Memory - Total: {total_memory/1e9:.1f}GB, "
                       f"Allocated: {allocated_memory/1e9:.1f}GB, "
                       f"Free: {free_memory/1e9:.1f}GB")
            
            if free_memory < 4e9:
                logger.warning(f"Low GPU memory: {free_memory/1e9:.1f}GB free. "
                             "Consider using CPU or clearing memory.")
            
            return True
        except Exception as e:
            logger.error(f"Device {device} compatibility test failed: {e}")
            return False
    else:
        logger.info("Using CPU device")
        return True

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def analyze_training_data_quality(training_data, dataset_name):

    logger.info(f"=== ANALYZING {dataset_name.upper()} TRAINING DATA QUALITY ===")
    
    task_stats = {}
    total_utterances = 0
    generic_responses = 0
    
    for example_id, example_data in training_data.items():
        task = example_data.get("task", "unknown")
        utterances = example_data.get("utterances", [])
        
        if task not in task_stats:
            task_stats[task] = {"total": 0, "generic": 0, "high_quality": 0}
        
        for utterance in utterances:

            response = (utterance.get("response", "") or 
                       utterance.get("system_response", "") or
                       utterance.get("target", ""))
            
            total_utterances += 1
            task_stats[task]["total"] += 1
            

            generic_patterns = [
                "need more specific information",
                "can you be more specific", 
                "what else would you like to know",
                "anything else i can help",
                "is there anything else"
            ]
            
            if any(pattern in response.lower() for pattern in generic_patterns):
                task_stats[task]["generic"] += 1
                generic_responses += 1
            elif len(response.split()) >= 3 and response.lower().strip() not in ["you're welcome", "thank you"]:
                task_stats[task]["high_quality"] += 1
    

    if total_utterances == 0:
        logger.warning(f"No utterances found in {dataset_name} training data!")
        return {}
    

    logger.info(f"Total utterances: {total_utterances}")
    logger.info(f"Generic responses: {generic_responses} ({generic_responses/total_utterances*100:.1f}%)")
    
    for task, stats in task_stats.items():
        if stats['total'] > 0:
            logger.info(f"Task {task}: {stats['total']} total, {stats['generic']} generic, {stats['high_quality']} high-quality")
    

    if total_utterances > 0 and generic_responses > total_utterances * 0.3:
        logger.warning(f"⚠️  HIGH GENERIC RESPONSE RATE (>30%) in {dataset_name} - training will filter these")
    else:
        logger.info(f"✓ Generic response rate acceptable for {dataset_name}")
    
    return task_stats

def build_global_kg_woz21(dataset, entity_filepath):

    combined_entities = load_combined_entities()
    
    ent2id = {}
    for i, entity in enumerate(combined_entities):
        ent2id[entity] = i
    
    rel2id = {}
    triple_list = []

    for sample in dataset:
        kg = sample.get("knowledge_text") or sample.get("kg") or []
        if kg:
            for triple in kg:
                if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                    continue
                subj, rel, obj = triple[:3]
                
                if subj not in ent2id:
                    ent2id[subj] = len(ent2id)
                if obj not in ent2id:
                    ent2id[obj] = len(ent2id)
                if rel not in rel2id:
                    rel2id[rel] = len(rel2id)
                
                triple_list.append([ent2id[subj], rel2id[rel], ent2id[obj]])
    
    if "unknown" not in ent2id:
        ent2id["unknown"] = len(ent2id)
        
    logger.info(f"Built WOZ2_1 global KG: {len(ent2id)} entities, {len(rel2id)} relations, {len(triple_list)} triples")
    
    return ent2id, rel2id, triple_list

def load_combined_entities():

    entities_file = "/home/Experiments/KAVER_528/data/woz2_1/entities.json"
    woz_entities_file = "/home/Experiments/KAVER_528/data/woz2_1/woz_entities.json"
    
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            specific_entities = json.load(f)

        with open(woz_entities_file, 'r', encoding='utf-8') as f:
            categorical_entities = json.load(f)

        combined_entities = list(specific_entities)
        for category, entity_list in categorical_entities.items():
            for entity in entity_list:
                if entity not in combined_entities:
                    combined_entities.append(entity)
        
        return combined_entities
        
    except Exception as e:
        logger.error(f"Error loading entity files: {e}")
        return ["unknown"]
def convert_dataset_to_adapter_format(dataset_examples, dataset_name):

    logger.info(f"Converting {dataset_name} dataset to adapter training format...")
    
    adapter_data = {}
    
    for i, example in enumerate(dataset_examples):
        dialog_id = f"{dataset_name}_dialog_{i}"
        
        task = example.get("task", "unknown")
        current_question = example.get("current_question", "")
        response_text = example.get("response_text", "")
        history = example.get("history", [])
        
        utterances = []
        
        if history:
            if dataset_name.lower() in ["camrest", "camrest2_1"]:
                for j, hist_turn in enumerate(history[:-1]):  
                    if isinstance(hist_turn, list):
                        try:
                            from transformers import AutoTokenizer
                            hist_text = f"History turn {j}"
                        except:
                            hist_text = f"History turn {j}"
                    else:
                        hist_text = str(hist_turn)
                    
                    utterances.append({
                        "turn_id": j,
                        "user_utterance": hist_text,
                        "response": ""
                    })
            else:
                for j, hist_turn in enumerate(history):
                    if isinstance(hist_turn, str):
                        utterances.append({
                            "turn_id": j,
                            "user_utterance": hist_turn,
                            "response": ""
                        })
        
        if current_question and response_text:
            utterances.append({
                "turn_id": len(utterances),
                "user_utterance": current_question,
                "response": response_text
            })
        
        adapter_data[dialog_id] = {
            "task": task,
            "utterances": utterances,
            "dialog_id": dialog_id
        }
    
    logger.info(f"Converted {len(adapter_data)} {dataset_name} dialogues to adapter format")
    return adapter_data

def get_dataset_config(dataset_name):

    configs = {
        "camrest": {
            "dataroot": "data/camrest",
            "checkpoint_path": "/home/Experiments/KAVER_528/ckpt_camrest_fixed/kgcot_best_epoch9_loss13.780798.pt",
            "adapters_dir": "/home/Experiments/KAVER_528/adapters_camrest",
            "kg_cache_path": "/home/Experiments/KAVER_528/ckpt_camrest/global_kg_fixed.pkl",
            "evaluation_function": evaluate_camrest_integrated,
            "dataset_module": "scripts.dataset_camrest"
        },
        "woz2_1": {
            "dataroot": "data/woz2_1", 
            "checkpoint_path": "/home/Experiments/KAVER_528/ckpt_woz2_1_fixed/kgcot_best_epoch12_loss14.051238.pt",
            "adapters_dir": "/home/Experiments/KAVER_528/adapters_woz21",
            "kg_cache_path": "/home/Experiments/KAVER_528/ckpt_woz2_1/global_kg_fixed.pkl", 
            "evaluation_function": evaluate_woz21_integrated,
            "dataset_module": "scripts.dataset_woz2_1"
        },
        "incar": {
            "dataroot": "data/incar",
            "checkpoint_path": "/home/Experiments/KAVER_528/ckpt_incar/kgcot_best_epoch10_loss2.368533.pt",
            "adapters_dir": "/home/Experiments/KAVER_528/adapters_incar",
            "kg_cache_path": "/home/Experiments/KAVER_528/ckpt_incar/global_kg_fixed.pkl",
            "evaluation_function": None,  # 需要实现
            "dataset_module": "scripts.dataset_incar"
        }
    }
    
    return configs.get(dataset_name.lower(), configs["woz2_1"])
def convert_woz_to_adapter_format(woz_dataset):

    logger.info("Converting camrest dataset to adapter training format...")
    
    adapter_data = {}
    
    for i, example in enumerate(woz_dataset):
        dialog_id = f"dialog_{i}"
        
        task = example.get("task", "unknown")
        current_question = example.get("current_question", "")
        response_text = example.get("response_text", "")
        history = example.get("history", [])
        
        utterances = []
        
        for j, hist_turn in enumerate(history):
            if isinstance(hist_turn, str):
                utterances.append({
                    "turn_id": j,
                    "user_utterance": hist_turn,
                    "response": "" 
                })
        
        if current_question and response_text:
            utterances.append({
                "turn_id": len(utterances),
                "user_utterance": current_question,
                "response": response_text
            })
        
        adapter_data[dialog_id] = {
            "task": task,
            "utterances": utterances,
            "dialog_id": dialog_id
        }
    
    logger.info(f"Converted {len(adapter_data)} dialogues to adapter format")
    return adapter_data

def load_camrest_kg_simple(dataset):

    import pickle
    import os

    kg_cache_path = "/home/Experiments/KAVER_528/ckpt_camrest/global_kg_fixed.pkl"
    
    if os.path.exists(kg_cache_path):
        try:
            with open(kg_cache_path, "rb") as f:
                kg_data = pickle.load(f)
            logger.info("Successfully loaded CamRest KG from cache")
            return kg_data["global_ent_dict"], kg_data["rel2id"], kg_data["triple_list"]
        except Exception as e:
            logger.warning(f"Failed to load CamRest KG cache: {e}")
    
    logger.info("Building simple CamRest KG...")
    
    ent2id = {}
    entities_file = "/home/Experiments/KAVER_528/data/camrest/entities.json"
    
    if os.path.exists(entities_file):
        try:
            import json
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_list = json.load(f)
                for entity in entities_list:
                    if entity and entity not in ent2id:
                        ent2id[entity] = len(ent2id)
            logger.info(f"Loaded {len(entities_list)} CamRest entities")
        except Exception as e:
            logger.warning(f"Failed to load CamRest entities: {e}")

            basic_entities = ["unknown", "restaurant", "centre", "north", "south", "east", "west", 
                             "cheap", "moderate", "expensive", "chinese", "italian", "british"]
            for entity in basic_entities:
                ent2id[entity] = len(ent2id)
    else:
        logger.warning(f"CamRest entities file not found: {entities_file}")

        basic_entities = ["unknown", "restaurant", "centre", "north", "south", "east", "west", 
                         "cheap", "moderate", "expensive", "chinese", "italian", "british"]
        for entity in basic_entities:
            ent2id[entity] = len(ent2id)
    
    rel2id = {}
    triple_list = []
    
    basic_relations = ["food", "area", "pricerange", "name", "address", "phone", "postcode"]
    for rel in basic_relations:
        rel2id[rel] = len(rel2id)
    
    if hasattr(dataset, 'examples'):
        dataset_examples = dataset.examples
    elif hasattr(dataset, '__iter__'):
        dataset_examples = dataset
    else:
        dataset_examples = []
    
    for sample in dataset_examples:
        kg = sample.get("knowledge_text") or sample.get("kg") or []
        for triple in kg:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            subj, rel, obj = triple[:3]
            
            if subj not in ent2id:
                ent2id[subj] = len(ent2id)
            if obj not in ent2id:
                ent2id[obj] = len(ent2id)

            if rel not in rel2id:
                rel2id[rel] = len(rel2id)
            
            triple_list.append([ent2id[subj], rel2id[rel], ent2id[obj]])
    
    if "unknown" not in ent2id:
        ent2id["unknown"] = len(ent2id)

    try:
        os.makedirs(os.path.dirname(kg_cache_path), exist_ok=True)
        with open(kg_cache_path, "wb") as f:
            pickle.dump({
                "global_ent_dict": ent2id,
                "rel2id": rel2id,
                "triple_list": triple_list
            }, f)
        logger.info(f"Saved CamRest KG to cache: {kg_cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save CamRest KG cache: {e}")
    
    logger.info(f"Built CamRest KG: {len(ent2id)} entities, {len(rel2id)} relations, {len(triple_list)} triples")
    return ent2id, rel2id, triple_list

def load_val_data_universal(decode_tokenizer, dataset_name):

    from argparse import Namespace
    
    config = get_dataset_config(dataset_name)
    
    import importlib
    dataset_module = importlib.import_module(config["dataset_module"])
    EvalDataset = getattr(dataset_module, "EvalDataset")
    
    args_data = Namespace(
        dataroot=config["dataroot"],
        history_max_tokens=256,
        knowledge_max_tokens=256,
        history_max_utterances=10,
        top_weights=7
    )
    
    eval_dataset = EvalDataset(args_data, decode_tokenizer, name=dataset_name, split_type="val")
    
    return eval_dataset, args_data

def convert_camrest_to_adapter_format(camrest_dataset):

    logger.info("Converting camrest dataset to adapter training format...")
    
    adapter_data = {}
    
    for i, example in enumerate(camrest_dataset):
        dialog_id = f"camrest_dialog_{i}"

        task = example.get("task", "dontcare") 
        current_question = example.get("current_question", "")
        response_text = example.get("response_text", "")
        history = example.get("history", [])
        

        utterances = []
        
        for j, hist_turn in enumerate(history):
            if isinstance(hist_turn, str):
                utterances.append({
                    "turn_id": j,
                    "user": hist_turn, 
                    "response": ""
                })
        

        if current_question and response_text:
            utterances.append({
                "turn_id": len(utterances),
                "user": current_question,  
                "response": response_text
            })
        

        adapter_data[dialog_id] = {
            "task": task,
            "utterances": utterances,
            "dialog_id": dialog_id
        }
    
    logger.info(f"Converted {len(adapter_data)} camrest dialogues to adapter format")
    return adapter_data
def main_universal(dataset_name="camrest"):

    config = get_dataset_config(dataset_name)
    logger.info(f"=== INITIALIZING {dataset_name.upper()} EVALUATION SYSTEM ===")
    
    logger.info("=== DEVICE SELECTION ===")
    device = get_optimal_device()
    
    if not check_device_compatibility(device):
        logger.error("Device compatibility check failed, falling back to CPU")
        device = torch.device("cpu")
    
    if device.type == "cuda":
        torch.cuda.set_device(device)
        logger.info(f"Set default CUDA device to {device}")
    
    with open("config/gpt2/params.json", "r") as fp:
        global_params = json.load(fp)
    with open("config/gpt2/generation_params.json", "r") as fp:
        gen_params = json.load(fp)
    if "max_new_tokens" not in gen_params:
        gen_params["max_new_tokens"] = 50
        
    set_seed(global_params.get("seed", 42))
    
    logger.info("Loading tokenizers...")
    decode_tokenizer = AutoTokenizer.from_pretrained(global_params["model_name_or_path"])
    decode_tokenizer.add_special_tokens({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "[PAD]",
        "additional_special_tokens": ["[SYS]", "[USR]", "[KG]", "[SUB]", "[PRED]", "[OBJ]", "[TRIPLE]", "[SEP]", "[Q]"],
    })
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    logger.info(f"Loading {dataset_name} training data...")
    
    import importlib
    dataset_module = importlib.import_module(config["dataset_module"])
    EvalDataset = getattr(dataset_module, "EvalDataset")
    
    args_data = Namespace(
        dataroot=config["dataroot"],
        history_max_tokens=256,
        knowledge_max_tokens=256,
        history_max_utterances=10,
        top_weights=7
    )
    dataset = EvalDataset(args_data, decode_tokenizer, name=dataset_name, split_type="train")

    logger.info(f"Loading/building {dataset_name} knowledge graph...")
    if dataset_name.lower() in ["camrest", "camrest2_1"]:
        global_ent_dict, rel2id, triple_list = load_camrest_kg_simple(dataset)
    else:
        global_ent_dict, rel2id, triple_list = load_or_build_kg_universal(dataset, dataset_name)

    if not global_ent_dict:
        global_ent_dict = {"unknown": 0}
    if triple_list:
        triples_tensor = torch.tensor(triple_list, dtype=torch.long)
    else:
        triples_tensor = torch.zeros((0, 3), dtype=torch.long)
    logger.info(f"{dataset_name} Global KG stats: entities=%d, relations=%d, triples=%d", 
               len(global_ent_dict), len(rel2id), triples_tensor.size(0))
    
    logger.info("Initializing KG-CoT model...")
    args_model = Namespace(bert_name="bert-base-uncased")
    kgcot_model = GraphReasoningModel(args_model, global_ent_dict, rel2id, triples_tensor)
    
    ckpt_path = config["checkpoint_path"]
    if os.path.exists(ckpt_path):
        logger.info(f"Loading pre-trained KG-CoT model checkpoint: {ckpt_path}")
        
        if device.type == "cuda":
            checkpoint = torch.load(ckpt_path, map_location=device)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
        kgcot_model.load_state_dict(checkpoint)
        kgcot_model = kgcot_model.to(device)
        logger.info(f"Model loaded and moved to {device}")
        
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.warning(f"Pre-trained checkpoint not found at {ckpt_path}")
        logger.info("You may need to train the model first.")
        return None
    
    print(f"\n=== {dataset_name.upper()} ADAPTER TRAINING OPTIONS ===")
    print("1. Train new adapters with FIXED training pipeline")
    print("2. Use existing adapters (if available)")
    print("3. Skip adapter training")
    choice = input("Choose option (1/2/3): ").strip()
    
    adapters_dir = config["adapters_dir"]
    
    if choice == "1":
        print(f"Starting FIXED {dataset_name} domain adapter training...")
        
        try:
            logger.info(f"Converting {dataset_name} dataset for adapter training...")
            training_data = convert_dataset_to_adapter_format(dataset.examples, dataset_name)
            print(f"Converted {dataset_name} training data with {len(training_data)} dialogues")
            
            analyze_training_data_quality(training_data, dataset_name)
            
        except Exception as e:
            print(f"Error loading {dataset_name} training data: {e}")
            training_data = {}
        
        os.makedirs(adapters_dir, exist_ok=True)
        
        adapter_finetuner = DomainAdapterFineTuner(
            base_model_path="/home/Experiments/KAVER-main/runs/Qwen1.5-7B",
            output_dir=adapters_dir,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05
        )
        
        print(f"Training {dataset_name} adapters with FIXED pipeline...")
        successful_adapters = adapter_finetuner.train_all_domain_adapters(training_data)
        print(f"FIXED {dataset_name} adapter training completed: {successful_adapters}")
        
        if not successful_adapters:
            print(f"⚠️  No {dataset_name} adapters were successfully trained. Check the logs.")
            adapters_dir = None
        else:
            print(f"✓ Successfully trained {dataset_name} adapters: {list(successful_adapters.keys())}")
            
    elif choice == "2":
        print(f"Using existing {dataset_name} adapters from {adapters_dir}")
        if not os.path.exists(adapters_dir):
            print(f"⚠️  {dataset_name} adapters directory {adapters_dir} does not exist")
            adapters_dir = None
    else:
        print(f"Skipping {dataset_name} adapter training")
        adapters_dir = None
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        logger.info(f"Memory allocated before evaluation: {allocated/1e9:.1f}GB")
    
    logger.info(f"Initializing {dataset_name} enhanced response generator...")
    if adapters_dir and os.path.exists(adapters_dir):
        print(f"Initializing response generator with {dataset_name} adapters from {adapters_dir}")
        enhanced_generator = EnhancedResponseGenerator(
            base_model_path="/home/Experiments/KAVER-main/runs/Qwen1.5-7B",
            adapters_dir=adapters_dir
        )
    else:
        print(f"Initializing response generator without {dataset_name} adapters (base model only)")
        enhanced_generator = EnhancedResponseGenerator(
            base_model_path="/home/Experiments/KAVER-main/runs/Qwen1.5-7B",
            adapters_dir=None
        )

    logger.info(f"Loading {dataset_name} validation dataset...")
    eval_dataset, args_data = load_val_data_universal(decode_tokenizer, dataset_name)
    logger.info(f"Loaded {len(eval_dataset)} {dataset_name} validation examples")
    
    logger.info(f"Starting {dataset_name} enhanced integrated evaluation...")
    
    evaluation_function = config["evaluation_function"]
    if evaluation_function is None:
        logger.error(f"No evaluation function defined for {dataset_name}")
        return None
    
    results = evaluation_function(
        kgcot_model, 
        eval_dataset,
        decode_tokenizer, 
        bert_tokenizer, 
        len(global_ent_dict), 
        device, 
        rel2id,
        enhanced_generator,
        global_ent_dict=global_ent_dict
    )
    
    print(f"\n=== {dataset_name.upper()} EVALUATION COMPLETE ===")
    print(f"Results saved and logged. Check the evaluation output for detailed metrics.")
    
    if device.type == "cuda":
        final_allocated = torch.cuda.memory_allocated(device)
        logger.info(f"Final memory allocated: {final_allocated/1e9:.1f}GB")
        torch.cuda.empty_cache()
        logger.info("Final GPU memory cleanup completed")
    
    return results

def main_camrest():
    return main_universal("camrest")

def main_woz21():
    return main_universal("woz2_1")

def main_incar():
    return main_universal("incar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAVER Universal Evaluation System")
    parser.add_argument("--dataset", type=str, default="camrest", 
                       choices=["camrest", "woz2_1", "incar"],
                       help="Dataset to evaluate (default: camrest)")
    args = parser.parse_args()
    
    try:
        results = main_universal(args.dataset)
        if results:
            print(f"✅ {args.dataset.upper()} evaluation completed successfully!")
        else:
            print(f"❌ {args.dataset.upper()} evaluation failed!")
    except KeyboardInterrupt:
        print(f"\n⚠️  {args.dataset.upper()} evaluation interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Resources cleaned up")
    except Exception as e:
        print(f"\n❌ Error occurred during {args.dataset.upper()} evaluation: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Resources cleaned up")
