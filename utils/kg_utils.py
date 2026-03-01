#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
utils/kg_utils.py

"""
import os
import re
import json
import torch
import logging
import spacy
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Failed to load spaCy model: {e}")
    # Create a simple fallback
    import spacy.blank
    nlp = spacy.blank("en")

def load_global_entities(filepath):

    ent2id = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    if entity not in ent2id:
                        ent2id[entity] = len(ent2id)
        logger.info(f"Loaded {len(ent2id)} entities from {filepath}")
    except Exception as e:
        logger.error(f"Error loading entities from {filepath}: {e}")
        ent2id = {"unknown": 0}
        
    return ent2id

def build_global_kg(dataset, entity_filepath):

    ent2id = load_global_entities(entity_filepath)
    rel2id = {}
    triple_list = []

    for sample in dataset.examples:
        kg = sample.get("knowledge_text") or sample.get("kg") or sample.get("knowledge")
        
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
                    
                # Add triple to list
                triple_list.append([ent2id[subj], rel2id[rel], ent2id[obj]])

    if "unknown" not in ent2id:
        ent2id["unknown"] = len(ent2id)
        
    logger.info(f"Built global KG: {len(ent2id)} entities, {len(rel2id)} relations, {len(triple_list)} triples")
    return ent2id, rel2id, triple_list

def load_entity_mapping(filepath):

    entity_mapping = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                entity = line.strip()
                if entity:
                    entity_mapping[entity] = i  # Map entity name to unique ID
        logger.info(f"Loaded entity mapping with {len(entity_mapping)} entities")
    except Exception as e:
        logger.error(f"Error loading entity mapping from {filepath}: {e}")
        entity_mapping = {}
        
    return entity_mapping

def load_custom_entities(filepath):

    custom_entities = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    custom_entities.append(entity)
        logger.info(f"Loaded {len(custom_entities)} custom entities from {filepath}")
    except Exception as e:
        logger.error(f"Error loading custom entities from {filepath}: {e}")
        custom_entities = []
        
    return custom_entities

def get_global_entities(dataset="incar"):

    try:
        if dataset == "incar":
            with open('data/incar/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                            
                global_entity_list = list(set(global_entity_list))
                return global_entity_list
                
        elif dataset == "camrest":
            return json.load(open(f"data/{dataset}/entities.json"))
            
        elif dataset == "woz2_1" or dataset == "WOZ2_1":

            possible_paths = [
                "/home/Experiments/KAVER_528/data/woz2_1/entities.json"
            ]
            
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            entities = json.load(f)
                        logger.info(f"Successfully loaded WOZ2.1 entities from {path}")
                        return entities
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
                    continue
            
            logger.warning("Could not find WOZ2.1 entities file in any expected location")
            return []
            
        else:
   
            logger.warning(f"Unknown dataset: {dataset}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading global entities for {dataset}: {e}")
        return []  


def detect_start_entity_spacy(user_question, history_text=""):

    if not isinstance(user_question, str):
        user_question = str(user_question)
    if not isinstance(history_text, str):
        history_text = str(history_text)

    full_context = history_text + " " + user_question if history_text else user_question
    processed = full_context.replace("_", " ").strip()
    
    try:
        doc = nlp(processed)

        question_doc = nlp(user_question.replace("_", " ").strip())
        question_entities = [ent.text.strip() for ent in question_doc.ents]
        
        if question_entities:
            return question_entities[0]

        if doc.ents:
            return doc.ents[0].text.strip()
    except Exception as e:
        logger.warning(f"Entity detection error: {e}")
        return "unknown"

    location_patterns = ["in", "at", "near", "for"]
    for pattern in location_patterns:
        if pattern in user_question:
            parts = user_question.split(pattern, 1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()

    match = re.search(r'\b([a-zA-Z]+_[a-zA-Z]+)\b', user_question)
    if match:
        return match.group(1)
    
    return "unknown"

def detect_anchor_entities_spacy(history, window_size=10):

    if isinstance(history, list):
        windowed = history[-window_size:]
        if len(windowed) % 2 == 1:
            current_question = windowed[-1]
        else:
            current_question = windowed[-2] if len(windowed) >= 2 else (windowed[-1] if windowed else "")
        history_text = " ".join([str(t) for t in windowed])
    else:
        current_question = str(history)
        history_text = current_question

    if not isinstance(current_question, str):
        current_question = str(current_question)
    if not isinstance(history_text, str):
        history_text = str(history_text)

    anchor_entities = set()

    try:
        question_doc = nlp(current_question.replace("_", " ").strip())
        for ent in question_doc.ents:
            anchor_entities.add(ent.text.strip().lower().replace(" ", "_"))

        if not anchor_entities:
            context_doc = nlp(history_text.replace("_", " ").strip())
            for ent in context_doc.ents:
                anchor_entities.add(ent.text.strip().lower().replace(" ", "_"))
    except Exception as e:
        logger.warning(f"spaCy NER error in detect_anchor_entities_spacy: {e}")

    location_patterns = ["in", "at", "near", "for"]
    for pattern in location_patterns:
        if f" {pattern} " in f" {current_question} ":
            parts = current_question.split(pattern, 1)
            if len(parts) > 1:
                candidate = parts[1].strip().split()[0] if parts[1].strip() else ""
                if candidate:
                    anchor_entities.add(candidate.lower().replace(" ", "_"))

    for token in re.findall(r'\b([a-zA-Z]+_[a-zA-Z]+)\b', current_question):
        anchor_entities.add(token.lower())

    cleaned = set()
    entity_list = sorted(anchor_entities)
    for i, e1 in enumerate(entity_list):
        dominated = False
        for j, e2 in enumerate(entity_list):
            if i != j and SequenceMatcher(None, e1, e2).ratio() > 0.9 and len(e2) > len(e1):
                dominated = True
                break
        if not dominated:
            cleaned.add(e1)

    anchor_entities = cleaned if cleaned else {"unknown"}
    logger.debug(f"Anchor entities detected: {anchor_entities}")
    return anchor_entities


def build_query_subgraph_khop(triples, anchor_entities, k=2):

    if not triples:
        return list(triples) if triples is not None else []

    norm_anchors = {str(e).lower().replace(" ", "_") for e in anchor_entities}

    if not norm_anchors or norm_anchors == {"unknown"}:
        logger.debug("build_query_subgraph_khop: no valid anchors, returning all triples")
        return list(triples)

    adjacency = {}
    triple_list = list(triples)
    for triple in triple_list:
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        h = str(triple[0]).lower().replace(" ", "_")
        t = str(triple[2]).lower().replace(" ", "_")
        adjacency.setdefault(h, set()).add(t)
        adjacency.setdefault(t, set()).add(h)

    visited = set(norm_anchors)
    frontier = set(norm_anchors)
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for neighbour in adjacency.get(node, set()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
        frontier = next_frontier
        if not frontier:
            break

    subgraph = [
        triple for triple in triple_list
        if isinstance(triple, (list, tuple)) and len(triple) >= 3
        and (
            str(triple[0]).lower().replace(" ", "_") in visited
            or str(triple[2]).lower().replace(" ", "_") in visited
        )
    ]

    if not subgraph:
        logger.debug("build_query_subgraph_khop: empty subgraph after BFS, falling back to all triples")
        return triple_list

    logger.debug(f"build_query_subgraph_khop: {len(triple_list)} -> {len(subgraph)} triples (k={k})")
    return subgraph

