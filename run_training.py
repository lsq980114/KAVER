import os
import sys
import torch
import json
import pickle
import logging
import argparse
from pathlib import Path
from argparse import Namespace

camrest_discovered_entities = {}

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer, BertTokenizer
from KG_CoT_Model.model import GraphReasoningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_camrest_dataset_comprehensive(dataset, dataset_name):

    if dataset_name.lower() not in ["camrest", "camrest2_1"]:
        return {}, {}

    print(f"\n=== 开始全面分析CamRest数据集 ===")

    task_stats = {}
    original_task_stats = {}
    cuisine_from_task = set()
    cuisine_from_data = set()
    area_from_data = set()
    price_from_data = set()

    cuisine_from_kg = set()
    area_from_kg = set()
    price_from_kg = set()

    cuisine_from_questions = set()
    area_from_questions = set()
    price_from_questions = set()

    area_keywords = ['centre', 'center', 'north', 'south', 'east', 'west']
    price_keywords = ['cheap', 'moderate', 'expensive']

    print(f"分析 {len(dataset.examples)} 个样本...")

    for i, example in enumerate(dataset.examples):
        task = example.get('task', 'unknown')
        task_stats[task] = task_stats.get(task, 0) + 1

        original_task = example.get('original_task', task)
        original_task_stats[original_task] = original_task_stats.get(original_task, 0) + 1

        non_cuisine_keywords = ['restaurant', 'inform', 'request', 'recommend', 'book', 'unknown']
        if original_task not in non_cuisine_keywords:
            cuisine_from_task.add(original_task)

        ref_entities = example.get('reference_entities', [])
        for entity in ref_entities:
            entity_str = str(entity).lower().strip()
            if entity_str in area_keywords:
                area_from_data.add(entity_str)
            elif entity_str in price_keywords:
                price_from_data.add(entity_str)
            elif (len(entity_str) < 20 and
                  not entity_str.replace('_', '').replace(' ', '').isdigit() and
                  'road' not in entity_str and 'street' not in entity_str and
                  'phone' not in entity_str and len(entity_str.split('_')) < 4):
                cuisine_from_data.add(entity_str)

        kg = example.get('knowledge_text', [])
        for triple in kg:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                subj, rel, obj = str(triple[0]).lower(), str(triple[1]).lower(), str(triple[2]).lower()

                if rel in ['food', 'cuisine']:
                    cuisine_from_kg.add(obj)
                elif rel in ['area', 'location']:
                    area_from_kg.add(obj)
                elif rel in ['pricerange', 'price']:
                    price_from_kg.add(obj)

        current_question = example.get('current_question', '').lower()

        for cuisine in cuisine_from_task:
            if cuisine in current_question:
                cuisine_from_questions.add(cuisine)

        for area in area_keywords:
            if area in current_question:
                area_from_questions.add(area)

        for price in price_keywords:
            if price in current_question:
                price_from_questions.add(price)

    all_cuisines = cuisine_from_task | cuisine_from_kg | cuisine_from_questions | cuisine_from_data
    all_areas = area_from_data | area_from_kg | area_from_questions
    all_prices = price_from_data | price_from_kg | price_from_questions

    print(f"\n=== 原始任务类型统计 ===")
    for task, count in sorted(original_task_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task}: {count} 个样本 ({count / len(dataset.examples) * 100:.1f}%)")

    print(f"\n=== 映射后任务类型统计 ===")
    for task, count in sorted(task_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task}: {count} 个样本 ({count / len(dataset.examples) * 100:.1f}%)")

    print(f"\n=== Cuisine类型检测 ===")
    print(f"从task字段检测到: {sorted(cuisine_from_task)}")
    print(f"从knowledge_graph检测到: {sorted(cuisine_from_kg)}")
    print(f"从问题文本检测到: {sorted(cuisine_from_questions)}")
    print(f"从reference_entities检测到: {sorted(cuisine_from_data)}")
    print(f"总计cuisine类型: {sorted(all_cuisines)} (共{len(all_cuisines)}种)")

    print(f"\n=== 区域类型检测 ===")
    print(f"检测到的区域: {sorted(all_areas)}")

    print(f"\n=== 价格范围检测 ===")
    print(f"检测到的价格范围: {sorted(all_prices)}")

    entity_counter = {}
    for example in dataset.examples:
        for entity in example.get('reference_entities', []):
            entity_str = str(entity).lower().strip()
            entity_counter[entity_str] = entity_counter.get(entity_str, 0) + 1

    print(f"\n=== 最常见的实体 (Top 20) ===")
    top_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    for entity, count in top_entities:
        print(f"  {entity}: {count} 次")

    print(f"\n=== 分析完成 ===")

    return {
        'task_stats': task_stats,
        'original_task_stats': original_task_stats,
        'cuisines': sorted(all_cuisines),
        'areas': sorted(all_areas),
        'prices': sorted(all_prices),
        'top_entities': top_entities
    }


def load_woz_entities_correctly(dataset_name, discovered_cuisines=None, discovered_areas=None, discovered_prices=None):

    if dataset_name.lower() in ["woz2_1", "woz"]:
        entity_mapping = {}

        entities_file1 = "data/woz2_1/entities.json"
        if os.path.exists(entities_file1):
            with open(entities_file1, 'r', encoding='utf-8') as f:
                entities_list = json.load(f)
                for i, entity in enumerate(entities_list):
                    if entity and entity not in entity_mapping:
                        entity_mapping[entity] = len(entity_mapping)
            print(f"从 {entities_file1} 加载了 {len(entities_list)} 个实体")

        entities_file2 = "data/woz2_1/woz_entities.json"
        if os.path.exists(entities_file2):
            with open(entities_file2, 'r', encoding='utf-8') as f:
                entities_dict = json.load(f)
                for category, entity_list in entities_dict.items():
                    for entity in entity_list:
                        if entity and entity not in entity_mapping:
                            entity_mapping[entity] = len(entity_mapping)
            print(f"从 {entities_file2} 额外加载了实体")

        common_entities = [
            "centre", "north", "south", "east", "west",
            "13", "hotel", "restaurant", "cheap", "moderate", "expensive",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]

        for entity in common_entities:
            if entity not in entity_mapping:
                entity_mapping[entity] = len(entity_mapping)

        print(f"最终实体映射包含 {len(entity_mapping)} 个实体")
        print(f"前10个实体: {list(entity_mapping.keys())[:10]}")

        test_entities = ["centre", "north", "13"]
        for entity in test_entities:
            if entity in entity_mapping:
                print(f"✓ 实体 '{entity}' 映射到索引 {entity_mapping[entity]}")
            else:
                print(f"✗ 实体 '{entity}' 未找到")

        return entity_mapping

    elif dataset_name.lower() in ["camrest", "camrest2_1"]:
        entity_mapping = {}

        entities_file = "data/camrest/entities.json"
        if os.path.exists(entities_file):
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_list = json.load(f)
                for i, entity in enumerate(entities_list):
                    if entity and entity not in entity_mapping:
                        entity_mapping[entity] = len(entity_mapping)
            print(f"从 {entities_file} 加载了 {len(entities_list)} 个实体")

        if discovered_cuisines:
            print(f"动态添加发现的cuisine类型: {discovered_cuisines}")
            for cuisine in discovered_cuisines:
                if cuisine not in entity_mapping:
                    entity_mapping[cuisine] = len(entity_mapping)

        if discovered_areas:
            print(f"动态添加发现的区域类型: {discovered_areas}")
            for area in discovered_areas:
                if area not in entity_mapping:
                    entity_mapping[area] = len(entity_mapping)

        if discovered_prices:
            print(f"动态添加发现的价格类型: {discovered_prices}")
            for price in discovered_prices:
                if price not in entity_mapping:
                    entity_mapping[price] = len(entity_mapping)

        camrest_common_entities = [
            "centre", "north", "south", "east", "west",
            "restaurant", "cheap", "moderate", "expensive",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]

        for entity in camrest_common_entities:
            if entity not in entity_mapping:
                entity_mapping[entity] = len(entity_mapping)

        print(f"最终CamRest实体映射包含 {len(entity_mapping)} 个实体")
        print(f"前10个实体: {list(entity_mapping.keys())[:10]}")

        test_entities = ["centre"]
        if discovered_cuisines:
            test_entities.extend(list(discovered_cuisines)[:2])  # 检查前2个发现的cuisine

        for entity in test_entities:
            if entity in entity_mapping:
                print(f"✓ 实体 '{entity}' 映射到索引 {entity_mapping[entity]}")
            else:
                print(f"✗ 实体 '{entity}' 未找到")

        return entity_mapping

    else:
        entity_file = "data/incar/kvr_entities_incar.txt"
        entity_mapping = {}
        if os.path.exists(entity_file):
            with open(entity_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    entity = line.strip()
                    if entity:
                        entity_mapping[entity] = i
        return entity_mapping


def load_dataset_and_kg(dataset_name, decode_tokenizer):

    if dataset_name.lower() == "incar":
        from scripts.dataset_incar import EvalDataset
        base_args = {
            "dataroot": "data/incar",
            "history_max_tokens": 128,
            "knowledge_max_tokens": 256,
            "top_weights": 7,
            "history_max_utterances": 20
        }
        args = Namespace(**base_args)
        dataset = EvalDataset(args, decode_tokenizer, name="incar", split_type="train")
        entity_mapping = load_woz_entities_correctly("incar")

    elif dataset_name.lower() == "woz2_1" or dataset_name.lower() == "woz":
        from scripts.dataset_woz2_1 import EvalDataset
        base_args = {
            "dataroot": "data/woz2_1",
            "history_max_tokens": 128,
            "knowledge_max_tokens": 256,
            "top_weights": 7,
            "history_max_utterances": 20
        }
        args = Namespace(**base_args)
        dataset = EvalDataset(args, decode_tokenizer, name="woz2_1", split_type="train")
        entity_mapping = load_woz_entities_correctly("woz2_1")

    elif dataset_name.lower() in ["camrest", "camrest2_1"]:
        from scripts.dataset_camrest import EvalDataset
        base_args = {
            "dataroot": "data/camrest",
            "history_max_tokens": 128,
            "knowledge_max_tokens": 256,
            "top_weights": 7,
            "history_max_utterances": 20
        }
        args = Namespace(**base_args)

        print("=== 第一阶段：加载数据集并分析实体类型 ===")
        dataset = EvalDataset(args, decode_tokenizer, name="camrest", split_type="train")

        print("=== 第二阶段：分析数据集 ===")
        analysis_results = analyze_camrest_dataset_comprehensive(dataset, dataset_name)

        discovered_cuisines = analysis_results.get('cuisines', [])
        discovered_areas = analysis_results.get('areas', [])
        discovered_prices = analysis_results.get('prices', [])

        print("=== 第三阶段：构建增强的实体映射 ===")
        entity_mapping = load_woz_entities_correctly(
            dataset_name,
            discovered_cuisines=discovered_cuisines,
            discovered_areas=discovered_areas,
            discovered_prices=discovered_prices
        )

        global camrest_discovered_entities
        camrest_discovered_entities = {
            'cuisines': discovered_cuisines,
            'areas': discovered_areas,
            'prices': discovered_prices
        }

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    logger.info(f"加载{dataset_name}数据集，共 {len(dataset)} 个样本")

    kg_cache_path = f"ckpt_{dataset_name}/global_kg_fixed.pkl"

    if os.path.exists(kg_cache_path):
        logger.info("加载已缓存的知识图谱...")
        with open(kg_cache_path, "rb") as f:
            kg_data = pickle.load(f)
        ent2id = kg_data["global_ent_dict"]
        rel2id = kg_data["rel2id"]
        triple_list = kg_data["triple_list"]
    else:
        logger.info("构建新的知识图谱...")
        ent2id, rel2id, triple_list = build_global_kg(dataset, entity_mapping, dataset_name)

        os.makedirs(f"ckpt_{dataset_name}", exist_ok=True)
        with open(kg_cache_path, "wb") as f:
            pickle.dump({
                "global_ent_dict": ent2id,
                "rel2id": rel2id,
                "triple_list": triple_list
            }, f)

    return dataset, ent2id, rel2id, triple_list


def build_global_kg(dataset, entity_file_or_mapping, dataset_name):
    if isinstance(entity_file_or_mapping, dict):
        ent2id = entity_file_or_mapping.copy()
    else:
        ent2id = {}
        entity_file = entity_file_or_mapping
        if os.path.exists(entity_file):
            if entity_file.endswith('.txt'):
                with open(entity_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entity = line.strip()
                        if entity and entity not in ent2id:
                            ent2id[entity] = len(ent2id)
            elif entity_file.endswith('.json'):
                with open(entity_file, 'r', encoding='utf-8') as f:
                    entities = json.load(f)
                    for entity in entities:
                        if entity and entity not in ent2id:
                            ent2id[entity] = len(ent2id)

    rel2id = {}
    triple_list = []

    if dataset_name.lower() in ["camrest", "camrest2_1"]:
        analysis_results = analyze_camrest_dataset_comprehensive(dataset, dataset_name)

        detected_cuisines = analysis_results.get('cuisines', [])
        detected_areas = analysis_results.get('areas', [])
        detected_prices = analysis_results.get('prices', [])

        camrest_relations = [
            "cuisine", "area", "pricerange", "name", "address", "phone",
            "postcode", "food", "location", "type", "rating", "book",
            "has_cuisine", "in_area", "price_range", "serves"
        ]
        for rel in camrest_relations:
            if rel not in rel2id:
                rel2id[rel] = len(rel2id)
        for cuisine in detected_cuisines:
            if cuisine not in ent2id:
                ent2id[cuisine] = len(ent2id)
        for area in detected_areas:
            if area not in ent2id:
                ent2id[area] = len(ent2id)
        for price in detected_prices:
            if price not in ent2id:
                ent2id[price] = len(ent2id)
    for sample in dataset.examples:
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

    if dataset_name.lower() in ["camrest", "camrest2_1"]:
        logger.info(f"CamRest数据集完整分析结果:")
        logger.info(f"  实体数={len(ent2id)}, 关系数={len(rel2id)}, 三元组数={len(triple_list)}")

        analysis_results = analyze_camrest_dataset_comprehensive(dataset, dataset_name)
        detected_cuisines = analysis_results.get('cuisines', [])
        detected_areas = analysis_results.get('areas', [])
        detected_prices = analysis_results.get('prices', [])

        logger.info(f"  检测到的完整cuisine类型: {detected_cuisines}")
        logger.info(f"  检测到的区域类型: {detected_areas}")
        logger.info(f"  检测到的价格类型: {detected_prices}")

        task_stats = analysis_results.get('task_stats', {})
        logger.info(f"  任务类型分布: {task_stats}")

    if "unknown" not in ent2id:
        ent2id["unknown"] = len(ent2id)

    return ent2id, rel2id, triple_list


def convert_example_to_graph_inputs(example, decode_tokenizer, bert_tokenizer, global_ent_size,
                                    entity_mapping=None, dataset_name="woz", use_reference_entities=True,
                                    discovered_cuisines=None, discovered_areas=None, discovered_prices=None):
    total_ent_size = global_ent_size
    heads = torch.zeros((1, total_ent_size), dtype=torch.float)

    if dataset_name.lower() == "incar":
        current_question = example.get("current_question", "")

    elif dataset_name.lower() in ["woz2_1", "woz"]:
        history_data = example.get("history", [])
        if history_data:
            current_question_tokens = history_data[-1] if isinstance(history_data[-1], list) else []

            if current_question_tokens:
                current_question = decode_tokenizer.decode(current_question_tokens, skip_special_tokens=True)
            else:
                current_question = "empty question"
        else:
            current_question = "empty question"

    elif dataset_name.lower() in ["camrest", "camrest2_1"]:
        history_data = example.get("history", [])
        if history_data:
            current_question_tokens = history_data[-1] if isinstance(history_data[-1], list) else []

            if current_question_tokens:
                current_question = decode_tokenizer.decode(current_question_tokens, skip_special_tokens=True)
            else:
                current_question = "empty question"
        else:
            current_question = "empty question"
    else:
        current_question = "empty question"

    ref_entities = example.get("reference_entities", [])

    if not hasattr(convert_example_to_graph_inputs, '_debug_count'):
        convert_example_to_graph_inputs._debug_count = 0

    if convert_example_to_graph_inputs._debug_count < 5:
        logger.info(f"=== 样本 {convert_example_to_graph_inputs._debug_count} 实体处理 ===")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"example keys: {list(example.keys())}")
        logger.info(f"reference_entities: {ref_entities}")
        logger.info(f"current_question: '{current_question}'")

        convert_example_to_graph_inputs._debug_count += 1

    activated_entities = 0
    if ref_entities and entity_mapping and use_reference_entities:
        for entity in ref_entities:
            entity_str = str(entity).lower().strip()

            if entity_str in entity_mapping:
                idx = entity_mapping[entity_str]
                if 0 <= idx < total_ent_size:
                    heads[0, idx] = 1.0
                    activated_entities += 1
            else:
                variants = [
                    entity_str,
                    entity_str.replace("_", " "),
                    entity_str.replace(" ", "_"),
                    entity_str.replace("-", "_"),
                    entity_str.replace("'", "")
                ]

                for variant in variants:
                    if variant in entity_mapping:
                        idx = entity_mapping[variant]
                        if 0 <= idx < total_ent_size:
                            heads[0, idx] = 1.0
                            activated_entities += 1
                        break

    if dataset_name.lower() in ["camrest", "camrest2_1"]:
        question_lower = current_question.lower()

        if discovered_cuisines:
            cuisine_keywords = list(discovered_cuisines)
        else:
            cuisine_keywords = ["chinese", "korean", "british", "indian", "italian", "french", "thai", "vietnamese",
                                "japanese"]

        for cuisine in cuisine_keywords:
            if cuisine in question_lower and cuisine in entity_mapping:
                idx = entity_mapping[cuisine]
                if 0 <= idx < total_ent_size:
                    heads[0, idx] = 1.0
                    activated_entities += 1

        if discovered_areas:
            area_keywords = list(discovered_areas)
        else:
            area_keywords = ["centre", "center", "north", "south", "east", "west"]

        for area in area_keywords:
            if area in question_lower and area in entity_mapping:
                idx = entity_mapping[area]
                if 0 <= idx < total_ent_size:
                    heads[0, idx] = 1.0
                    activated_entities += 1

        if discovered_prices:
            price_keywords = list(discovered_prices)
        else:
            price_keywords = ["cheap", "moderate", "expensive"]

        for price in price_keywords:
            if price in question_lower and price in entity_mapping:
                idx = entity_mapping[price]
                if 0 <= idx < total_ent_size:
                    heads[0, idx] = 1.0
                    activated_entities += 1

        task = example.get("task", "")
        original_task = example.get("original_task", task)

        if original_task and original_task in entity_mapping:
            idx = entity_mapping[original_task]
            if 0 <= idx < total_ent_size:
                heads[0, idx] = 1.0
                activated_entities += 1

    if activated_entities == 0:
        heads[0, :min(50, total_ent_size)] = 1.0
        activated_entities = min(50, total_ent_size)

    history_text = ""
    history_data = example.get("history", [])
    if history_data:
        try:
            history_items = []
            if dataset_name.lower() in ["woz2_1", "woz", "camrest", "camrest2_1"]:
                history_to_use = history_data[:-1]
            else:
                history_to_use = history_data[-2:]

            for item in history_to_use:
                if isinstance(item, list):
                    decoded_text = decode_tokenizer.decode(item, skip_special_tokens=True)
                    history_items.append(decoded_text)
                elif isinstance(item, str):
                    history_items.append(item)
            history_text = " ".join(history_items)
        except Exception as e:
            logger.warning(f"处理历史对话时出错: {e}")

    question_text = current_question
    if history_text:
        question_text = f"{history_text} {current_question}".strip()

    questions = bert_tokenizer(
        question_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    if "token_type_ids" not in questions:
        questions["token_type_ids"] = torch.zeros_like(questions["input_ids"])

    answer = torch.zeros((1, total_ent_size), dtype=torch.float)

    if ref_entities and entity_mapping:
        for entity in ref_entities:
            entity_str = str(entity).lower().strip()
            if entity_str in entity_mapping:
                idx = entity_mapping[entity_str]
                if 0 <= idx < total_ent_size:
                    answer[0, idx] = 1.0

    entity_range = heads.clone()

    return heads, questions, answer, entity_range, current_question, history_text


def extract_current_question_for_training(example, decode_tokenizer, dataset_name):
    current_question = ""

    if dataset_name.lower() == "incar":
        current_question = example.get("current_question", "")

    elif dataset_name.lower() in ["woz2_1", "woz"]:
        history_data = example.get("history", [])
        if history_data:
            current_question_tokens = history_data[-1] if isinstance(history_data[-1], list) else []
            if current_question_tokens:
                current_question = decode_tokenizer.decode(current_question_tokens, skip_special_tokens=True)

    elif dataset_name.lower() in ["camrest", "camrest2_1"]:
        history_data = example.get("history", [])
        if history_data:
            current_question_tokens = history_data[-1] if isinstance(history_data[-1], list) else []
            if current_question_tokens:
                current_question = decode_tokenizer.decode(current_question_tokens, skip_special_tokens=True)

        if not current_question:
            current_question = example.get("current_question", "")

    return current_question


def train_kgcot_fixed(model, dataset, decode_tokenizer, bert_tokenizer, global_ent_size, device,
                      dataset_name, num_epochs=30, batch_size=4, learning_rate=2e-4, adam_epsilon=1e-8,
                      max_grad_norm=0.7):
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, RandomSampler
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    import logging
    import os
    import gc

    model.to(device)
    model.train()

    logger = logging.getLogger(__name__)
    checkpoint_dir = f"ckpt_{dataset_name}_fixed"
    os.makedirs(checkpoint_dir, exist_ok=True)
    bert_param = [(n, p) for n, p in model.named_parameters() if n.startswith('bert_encoder')]
    entity_param = [(n, p) for n, p in model.named_parameters() if 'entity_' in n or 'relation_' in n]
    other_param = [(n, p) for n, p in model.named_parameters()
                   if not n.startswith('bert_encoder') and 'entity_' not in n and 'relation_' not in n]

    no_decay = ['bias', 'LayerNorm.weight']
    bert_lr = 5e-5

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,  'lr': bert_lr},
        {'params': [p for n, p in entity_param if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.03, 'lr': learning_rate * 1.5},
        {'params': [p for n, p in entity_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,  'lr': learning_rate * 1.5},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.015, 'lr': learning_rate},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,   'lr': learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset) * 2)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            collate_fn=lambda batch: batch, num_workers=2)

    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(num_training_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_loss = float("inf")
    best_epoch = -1
    patience = 3
    patience_counter = 0
    improvement_threshold = 1e-3
    min_epochs = 8

    gradient_accumulation_steps = 2

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        batch_sizes = []
        all_batch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_examples = 0

            for example in batch:
                current_question = extract_current_question_for_training(example, decode_tokenizer, dataset_name)

                if "thank" in current_question.lower() or "thanks" in current_question.lower():
                    continue

                if "reference_entities" in example and example["reference_entities"]:
                    entity_ids = []
                    for entity in example["reference_entities"]:
                        entity_formats = [entity, entity.replace(" ", "_"), entity.replace("_", " ")]
                        for entity_format in entity_formats:
                            if entity_format in global_ent_dict:
                                entity_ids.append(global_ent_dict[entity_format])
                                break

                    if entity_ids:
                        example["entity_ids"] = entity_ids

                try:
                    # heads, questions, answers, entity_range, origin_question, history_text = convert_example_to_graph_inputs(
                    #     example, decode_tokenizer, bert_tokenizer, global_ent_size,
                    #     entity_mapping=global_ent_dict, dataset_name=dataset_name, use_reference_entities=True
                    # )

                    discovered_info = {}
                    if dataset_name.lower() in ["camrest", "camrest2_1"] and 'camrest_discovered_entities' in globals():
                        discovered_info = camrest_discovered_entities

                    heads, questions, answers, entity_range, origin_question, history_text = convert_example_to_graph_inputs(
                        example, decode_tokenizer, bert_tokenizer, global_ent_size,
                        entity_mapping=global_ent_dict, dataset_name=dataset_name, use_reference_entities=True,
                        discovered_cuisines=discovered_info.get('cuisines'),
                        discovered_areas=discovered_info.get('areas'),
                        discovered_prices=discovered_info.get('prices')
                    )

                    heads = heads.to(device).float()
                    answers = answers.to(device).float()
                    entity_range = entity_range.to(device).float()
                    questions = {k: v.to(device) for k, v in questions.items()}

                    outputs = model(heads, questions, answers, entity_range,
                                    question_text=origin_question, history_text=history_text)

                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']

                        normalized_loss = loss / (len(batch) * gradient_accumulation_steps)

                        normalized_loss.backward()

                        batch_loss += loss.item()
                        valid_examples += 1

                except Exception as e:
                    logger.warning(f"处理样本时出错: {e}")
                    continue

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                if valid_examples > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()

                    avg_loss = batch_loss / valid_examples if valid_examples > 0 else 0

                    all_batch_losses.append(avg_loss)
                    batch_sizes.append(valid_examples)
                    batch_count += 1
                    epoch_loss += batch_loss

                    pbar.set_postfix_str(
                        f"loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, valid={valid_examples}")

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        total_examples = sum(batch_sizes)
        weighted_epoch_loss = epoch_loss / total_examples if total_examples > 0 else float("inf")

        if all_batch_losses:
            min_loss = min(all_batch_losses)
            max_loss = max(all_batch_losses)
            median_loss = sorted(all_batch_losses)[len(all_batch_losses) // 2]

            logger.info(f"Epoch {epoch + 1}: average loss = {weighted_epoch_loss:.6f}, "
                        f"min_loss = {min_loss:.6f}, max_loss = {max_loss:.6f}, "
                        f"median_loss = {median_loss:.6f}, learning rate = {scheduler.get_last_lr()[0]:.7f}")
        else:
            logger.info(f"Epoch {epoch + 1}: No valid examples processed")

        if epoch >= min_epochs - 1:
            loss_change = best_loss - weighted_epoch_loss
            if loss_change > improvement_threshold:
                best_loss = weighted_epoch_loss
                best_epoch = epoch + 1
                patience_counter = 0

                save_path = os.path.join(checkpoint_dir,
                                         f"kgcot_best_epoch{epoch + 1}_loss{weighted_epoch_loss:.6f}.pt")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model to: {save_path} (improvement: {loss_change:.6f})")

            else:
                patience_counter += 1
                logger.info(f"No improvement (change: {loss_change:.6f}), patience: {patience_counter}/{patience}")

                if patience_counter > 0 and patience_counter % 2 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8
                    logger.info(f"Applied LR decay, new LR: {optimizer.param_groups[0]['lr']:.7f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.6f} at epoch {best_epoch}")
                break

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(checkpoint_dir, f"kgcot_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), save_path)

    logger.info(f"Training completed, best_loss={best_loss:.6f} at epoch {best_epoch}")
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="修复版KG-CoT训练脚本")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["incar", "woz2_1", "woz", "camrest"],
                        help="数据集名称")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="预训练检查点路径")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    model_path = "/home/Experiment/KAVER-main/runs/Qwen1.5-7B"
    decode_tokenizer = AutoTokenizer.from_pretrained(model_path)
    decode_tokenizer.add_special_tokens({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "[PAD]",
        "additional_special_tokens": ["[SYS]", "[USR]", "[KG]", "[SUB]", "[PRED]", "[OBJ]", "[TRIPLE]", "[SEP]", "[Q]"],
    })
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset, ent2id, rel2id, triple_list = load_dataset_and_kg(args.dataset, decode_tokenizer)

    if triple_list:
        triples_tensor = torch.tensor(triple_list, dtype=torch.long)
    else:
        triples_tensor = torch.zeros((0, 3), dtype=torch.long)

    args_model = Namespace(bert_name="bert-base-uncased")
    model = GraphReasoningModel(args_model, ent2id, rel2id, triples_tensor)

    logger.info(f"模型创建完成：实体数={len(ent2id)}, 关系数={len(rel2id)}, 三元组数={triples_tensor.size(0)}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"加载检查点: {args.checkpoint}")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            logger.info("检查点加载成功")
        except Exception as e:
            logger.warning(f"检查点加载失败: {e}")

    global global_ent_dict
    global_ent_dict = ent2id

    logger.info("开始训练...")
    best_loss = train_kgcot_fixed(
        model, dataset, decode_tokenizer, bert_tokenizer,
        len(ent2id), device, args.dataset
    )
    logger.info(f"训练完成！最佳损失: {best_loss:.6f}")


if __name__ == "__main__":

    main()
