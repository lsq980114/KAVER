import os
import torch
import json
import re
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device

def reconstruct_reasoning_paths(model_outputs, knowledge, id2ent, id2rel,
                                top_Kr=5, top_Ke=10, τ_r=0.05, τ_e=0.01):

    rel_probs_all = model_outputs.get('rel_probs', [])  # list of [num_relations]
    ent_probs_all = model_outputs.get('ent_probs', [])  # list of [num_entities]

    num_steps = len(rel_probs_all)
    if num_steps == 0 or len(ent_probs_all) == 0:
        return []

    top_rel_sets = []   # top_rel_sets[t] = set of top-Kr relation indices
    top_ent_sets = []   # top_ent_sets[t] = set of top-Ke entity indices at step t

    for t in range(num_steps):
        rel_probs = rel_probs_all[t]
        k_r = min(top_Kr, rel_probs.size(0))
        top_rel_sets.append(set(torch.topk(rel_probs, k_r).indices.tolist()))

    for t in range(len(ent_probs_all)):
        ent_probs = ent_probs_all[t]
        k_e = min(top_Ke, ent_probs.size(0))
        top_ent_sets.append(set(torch.topk(ent_probs, k_e).indices.tolist()))

    step_candidates = []  # step_candidates[t] = list of (h, r, t_ent, rel_prob)

    for t in range(num_steps):
        rel_probs = rel_probs_all[t]

        # Eq.13 中 TopK(e_t) 和 TopK(e_{t+1})
        ent_set_cur  = top_ent_sets[t]     if t < len(top_ent_sets)     else set()
        ent_set_next = top_ent_sets[t + 1] if (t + 1) < len(top_ent_sets) else set()

        candidates = []
        for triple in knowledge:
            h = triple[0].item() if isinstance(triple[0], torch.Tensor) else int(triple[0])
            r = triple[1].item() if isinstance(triple[1], torch.Tensor) else int(triple[1])
            t_ent = triple[2].item() if isinstance(triple[2], torch.Tensor) else int(triple[2])

            if r >= rel_probs.size(0):
                continue
            rel_prob_val = rel_probs[r].item()

            if r not in top_rel_sets[t]:
                continue
            if h not in ent_set_cur and t_ent not in ent_set_next:
                continue

            if rel_prob_val <= τ_r:
                continue

            ent_probs_next = ent_probs_all[t + 1] if (t + 1) < len(ent_probs_all) else None
            if ent_probs_next is not None:
                if t_ent < ent_probs_next.size(0):
                    ent_prob_val = ent_probs_next[t_ent].item()
                else:
                    ent_prob_val = 0.0
                if ent_prob_val <= τ_e:
                    continue

            candidates.append((h, r, t_ent, rel_prob_val))

        step_candidates.append(candidates)


    complete_paths = []

    def enumerate_paths(current_path, step_idx):
        if step_idx >= len(step_candidates):

            path_score = sum(p[3] for p in current_path)
            complete_paths.append((list(current_path), path_score))
            return
        for triple in step_candidates[step_idx]:

            if not current_path or triple[0] == current_path[-1][2]:
                enumerate_paths(current_path + [triple], step_idx + 1)

    enumerate_paths([], 0)

    complete_paths.sort(key=lambda x: x[1], reverse=True)

    formatted_paths = []
    for path, score in complete_paths[:5]:
        chain = " → ".join(
            f"{id2ent.get(p[0], f'ent_{p[0]}')} "
            f"--[{id2rel.get(p[1], f'rel_{p[1]}')}]--> "
            f"{id2ent.get(p[2], f'ent_{p[2]}')}"
            for p in path
        )
        formatted_paths.append({
            'chain': chain,
            'score': score,
            'path': chain,
            'triples': path
        })

    return formatted_paths



def extract_question_entities(question, id2name, id2ent):
    question_lower = question.lower()
    relevant_entities = []

    for ent_id, ent_name in id2name.items():
        if ent_name != '-' and ent_name.lower() in question_lower:
            relevant_entities.append((ent_id, ent_name, 'exact_match'))
    
    for ent_id, ent_name in id2ent.items():
        if ent_name.lower() in question_lower:
            relevant_entities.append((ent_id, ent_name, 'fuzzy_match'))

    relevant_entities.sort(key=lambda x: (x[2] == 'exact_match', len(x[1])), reverse=True)

    seen_ids = set()
    unique_entities = []
    for ent_id, ent_name, match_type in relevant_entities:
        if ent_id not in seen_ids:
            unique_entities.append(ent_id)
            seen_ids.add(ent_id)
    
    return unique_entities[:10]



def find_path(args, model, data, device, mapping):

    model.eval()
    predicted_paths_list = []
    
    with torch.no_grad():
        for batch in data:
            if len(batch) == 9:
                heads, questions, answers, used_knowledge, entity_range, origin_question, id2name, id2ent, domain = batch
                task_type = domain
            else:
                heads, questions, answers, used_knowledge, entity_range, origin_question, id2name, id2ent = batch
                task_type = analyze_task_type(origin_question)
            
            print(f"处理问题: {origin_question}")
            print(f"任务类型: {task_type}")
            print(f"知识三元组数量: {len(used_knowledge) if hasattr(used_knowledge, '__len__') else 'N/A'}")

            if len(used_knowledge) == 0 or not isinstance(used_knowledge, torch.Tensor):
                            print(f"[SKIP] No KG triples for question: '{origin_question}', skipping.")
                            continue

            outputs = model(heads, questions, answers, entity_range, question_text=origin_question)

            for i in range(answers.shape[0]):

                if 'e_score' in outputs:
                    e_score = outputs['e_score'][i]
                    top_entities = e_score.topk(min(20, len(e_score))).indices.tolist()
                elif 'ent_probs' in outputs and len(outputs['ent_probs']) > 0:
                    e_score = outputs['ent_probs'][-1][i]
                    top_entities = e_score.topk(min(20, len(e_score))).indices.tolist()
                else:
                    print("警告：无法获取实体分布")
                    continue

                if 'rel_probs' in outputs:
                    rel_probs_list = outputs['rel_probs']
                    if isinstance(rel_probs_list, list) and len(rel_probs_list) > 0:
                        if len(rel_probs_list) > i:
                            rel_probs = rel_probs_list[i]
                        else:
                            rel_probs = rel_probs_list[0]
                    else:
                        rel_probs = torch.ones(len(mapping.id2rel), device=device) / len(mapping.id2rel)
                else:
                    rel_probs = torch.ones(len(mapping.id2rel), device=device) / len(mapping.id2rel)

                question_entities = extract_question_entities(origin_question, id2name, id2ent)

                all_paths = reconstruct_reasoning_paths(
                    model_outputs=outputs,
                    knowledge=used_knowledge,
                    id2ent=id2ent,
                    id2rel=mapping.id2rel,
                    top_Kr=5, top_Ke=10, τ_r=0.05, τ_e=0.01
                )

                unique_paths = []
                seen_paths = set()
                for path in all_paths:
                    path_str = path['path']
                    if path_str not in seen_paths:
                        unique_paths.append(path)
                        seen_paths.add(path_str)
                    if len(unique_paths) >= 5:
                        break

                if unique_paths:
                    if len(unique_paths) == 1:
                        reasoning_chain = unique_paths[0]['path']
                    else:
                        reasoning_chain = " | ".join(p['path'] for p in unique_paths)
                else:
                    reasoning_chain = ""
                
                result = {
                    'question': origin_question,
                    'task_type': task_type,
                    'reasoning_chain': reasoning_chain,
                    'paths': unique_paths,
                    'question_entities': question_entities,
                    'is_solved': len(unique_paths) > 0,
                    'path_count': len(unique_paths),
                    'reasoning_type': 'question_answer_guided' if unique_paths else 'no_path'
                }
                
                predicted_paths_list.append(result)
    
    success_rate = sum(1 for result in predicted_paths_list if result['is_solved']) / len(predicted_paths_list) if predicted_paths_list else 0
    
    return success_rate, predicted_paths_list

