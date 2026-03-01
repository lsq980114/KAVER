import re
import torch
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


CUISINE_TO_RESTAURANT = {
    "italian", "chinese", "indian", "british", "french", "spanish",
    "thai", "vietnamese", "turkish", "portuguese", "korean", "japanese",
    "mediterranean", "modern_european", "gastropub", "african",
    "asian_oriental", "international", "lebanese", "dontcare"
}


def extract_answer_entity(response, task_type):
    response = response.strip()

 
    if task_type in CUISINE_TO_RESTAURANT:
        task_type = "restaurant"

    if task_type in ["navigate", "navigation"]:

        addr_pattern = r'\b\d+_[a-z_]+(?:ave|st|rd|dr|blvd|boulevard|way)\b'
        matches = re.findall(addr_pattern, response.lower())
        if matches:
            return matches[0]
        addr_pattern2 = r'\b\d+\s+[A-Za-z\s]+(?:Ave|Avenue|St|Street|Rd|Road|Dr|Drive|Blvd)\b'
        matches = re.findall(addr_pattern2, response, re.IGNORECASE)
        if matches:
            return matches[0].strip()
        quoted = re.findall(r'"([^"]+)"', response)
        if quoted:
            return quoted[0]

    elif task_type in ["weather"]:
        temp_pattern = r'(-?\d+)°?\s*[FC]'
        matches = re.findall(temp_pattern, response)
        if matches:
            condition_words = ['sunny', 'cloudy', 'rainy', 'snowy',
                               'clear', 'overcast', 'warm', 'cold', 'windy']
            condition = next(
                (w for w in condition_words if w in response.lower()), ""
            )
            return f"{matches[0]}° {condition}".strip()

    elif task_type in ["schedule", "calendar"]:
        datetime_pattern = (
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
            r'\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)'
        )
        matches = re.findall(datetime_pattern, response.lower())
        if matches:
            return matches[0]
        time_pattern = r'\d{1,2}(?::\d{2})?\s*(?:am|pm)'
        matches = re.findall(time_pattern, response.lower())
        if matches:
            return matches[0]

    elif task_type in ["restaurant", "hotel", "attraction"]:
        underscore = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', response.lower())
        if underscore:
            return underscore[0]
        quoted = re.findall(r'"([^"]+)"', response)
        if quoted:
            return quoted[0]
        words = response.split()
        if words and words[0][0].isupper():
            entity_words = [words[0]]
            for word in words[1:]:
                if word and word[0].isupper() and word not in ['The', 'A', 'An', 'I']:
                    entity_words.append(word)
                else:
                    break
            if entity_words:
                return ' '.join(entity_words)

    sentences = response.split('.')
    return sentences[0].strip() if sentences[0].strip() else response


def calculate_entity_similarity(answer1, answer2):
    if not answer1 or not answer2:
        return 0.0
    if answer1.lower() == answer2.lower():
        return 1.0
    if answer1.lower() in answer2.lower() or answer2.lower() in answer1.lower():
        return 0.9
    return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()


def self_consistency_aggregation(responses, task_type, similarity_threshold=0.8,
                                  response_log_probs=None):

    if not responses:
        return ""
    if len(responses) == 1:
        return responses[0]

    extracted_answers = []
    for response in responses:
        answer = extract_answer_entity(response, task_type)
        extracted_answers.append(answer)

    logger.debug(f"[SC] task={task_type}, extracted={extracted_answers}")

    clusters = []
    for i, answer in enumerate(extracted_answers):
        added_to_cluster = False
        for cluster in clusters:
            sim = calculate_entity_similarity(answer, cluster['representative'])
            if sim >= similarity_threshold:
                cluster['members'].append(i)
                cluster['count'] += 1
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append({
                'representative': answer,
                'members': [i],
                'count': 1
            })

    best_cluster = max(clusters, key=lambda x: x['count'])

    logger.debug(
        f"[SC] {len(clusters)} clusters, best='{best_cluster['representative']}' "
        f"freq={best_cluster['count']}/{len(responses)}"
    )

    if best_cluster['count'] == 1:
        logger.warning("[SC] No majority consensus, using highest log_prob response")

    if response_log_probs is not None and len(response_log_probs) == len(responses):
        best_response_idx = max(
            best_cluster['members'],
            key=lambda idx: response_log_probs[idx]
        )
    else:
        best_response_idx = best_cluster['members'][0]

    logger.debug(
        f"[SC] Selected idx={best_response_idx}, "
        f"log_prob={response_log_probs[best_response_idx] if response_log_probs else 'N/A'}"
    )

    return responses[best_response_idx]
