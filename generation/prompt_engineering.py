#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
generation/prompt_engineering.py

"""
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPromptEngineering:
    def __init__(self):
        """Initialize prompt engineering module"""

        self.system_instructions = {
            "restaurant": (
                "You are a restaurant recommendation assistant for Cambridge. "
                "Provide specific restaurant information including name, food type, "
                "price range (cheap/moderate/expensive), area (centre/north/south/east/west), "
                "address, and phone number when available. Be precise and helpful."
            ),
            "hotel": (
                "You are a hotel booking assistant for Cambridge. "
                "Provide specific accommodation information including name, type (hotel/guesthouse), "
                "star rating, price range, area, address, phone number, and available facilities "
                "(internet/parking/wifi) when available. Help with bookings when requested."
            ),
            "attraction": (
                "You are a tourist information assistant for Cambridge. "
                "Provide specific attraction information including name, type (museum/college/park/church), "
                "area, address, phone number, entrance fee information, and opening details "
                "when available. Be informative and helpful for tourists."
            ),
            "default": (
                "You are a helpful assistant for Cambridge information. "
                "Answer questions based on the provided information. "
                "Ask for clarification when needed."
            )
        }


    def create_improved_prompt(self, question, history_text, domain, location, chain_text, knowledge_summary, task,
                               missing_info=None, entity_library=None):

        intent = self.detect_question_intent(question)
        question_type = self.analyze_question_type(question, task)

        context_parts = []

        if history_text and history_text.strip():
            context_parts.append(f"Previous conversation: {history_text}")

        if domain and domain.strip() and domain != "unknown":
            context_parts.append(f"Domain: {domain}")

        if location and location.strip() and location != "unknown":
            context_parts.append(f"Location: {location}")

        if knowledge_summary and knowledge_summary.strip():
            IMPORTANT_RELS = {
                # SMD (InCar)
                "address", "poi_type", "traffic_info", "temperature",
                "weather_attribute", "distance", "road_condition",
                "high_temperature", "low_temperature", "date", "time",
                # CamRest / MultiWOZ
                "food", "pricerange", "stars", "area", "phone",
                "type", "entrance", "fee", "parking", "internet",
                # MultiWOZ train/taxi
                "departure", "destination", "arriveBy", "leaveAt",
                "trainID", "duration", "price",
            }
            knowledge_items = []
            for item in knowledge_summary.split(";"):
                item = item.strip()
                if item:
                    tag = " [IMPORTANT]" if any(r in item for r in IMPORTANT_RELS) else ""
                    knowledge_items.append(f"{item}{tag}")
            if knowledge_items:
                context_parts.append(f"Available information: {'; '.join(knowledge_items)}")

        if chain_text and chain_text.strip() and "无效思维链" not in chain_text and "无可用知识" not in chain_text:
            context_parts.append(f"Reasoning: {chain_text}")

        if missing_info:
            missing_info_text = f"Missing information: {', '.join(missing_info)}"
            context_parts.append(missing_info_text)

        if context_parts:
            context = " | ".join(context_parts)
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        return prompt

    def detect_question_intent(self, question):
        question_lower = question.lower()

        booking_keywords = ["book", "booking", "reserve", "reservation", "table"]
        if any(keyword in question_lower for keyword in booking_keywords):
            if any(word in question_lower for word in ["table", "restaurant", "food", "eat"]) or \
                    any(word in question_lower for word in ["people", "person", "party"]):
                return "restaurant_booking"
            elif any(word in question_lower for word in ["room", "hotel", "guesthouse", "stay", "night"]):
                return "hotel_booking"

        location_keywords = ["where", "address", "located", "location", "phone", "number"]
        if any(keyword in question_lower for keyword in location_keywords):
            if any(name in question_lower for name in ["pizza", "restaurant", "_hut", "_kitchen", "cafe"]) or \
                    "restaurant" in question_lower or "food" in question_lower:
                return "restaurant_info"
            elif any(name in question_lower for name in ["hotel", "guesthouse", "_house", "_lodge"]) or \
                    any(word in question_lower for word in ["hotel", "guesthouse", "accommodation"]):
                return "hotel_info"
            elif any(name in question_lower for name in ["museum", "_college", "_park", "_church"]) or \
                    any(word in question_lower for word in ["museum", "attraction", "college", "park"]):
                return "attraction_info"

        cost_keywords = ["cost", "price", "fee", "entrance", "ticket", "much", "expensive", "cheap"]
        if any(keyword in question_lower for keyword in cost_keywords):
            if any(word in question_lower for word in ["enter", "entrance", "museum", "attraction", "visit"]):
                return "attraction_pricing"
            elif any(word in question_lower for word in ["restaurant", "food", "meal"]):
                return "restaurant_general"
            elif any(word in question_lower for word in ["hotel", "room", "stay"]):
                return "hotel_general"

        if any(word in question_lower for word in ["restaurant", "food", "eat", "meal", "cuisine"]):
            recommendation_words = ["recommend", "suggest", "want", "looking for", "need", "find", "show me"]
            if any(word in question_lower for word in recommendation_words):
                return "restaurant_recommendation"
            else:
                return "restaurant_general"

        elif any(word in question_lower for word in ["hotel", "guesthouse", "lodge", "accommodation", "stay", "room"]):
            recommendation_words = ["recommend", "suggest", "need", "looking for", "want", "find", "show me"]
            if any(word in question_lower for word in recommendation_words):
                return "hotel_recommendation"
            else:
                return "hotel_general"

        elif any(word in question_lower for word in
                 ["attraction", "museum", "college", "park", "church", "entertainment", "visit", "see", "tour"]):
            recommendation_words = ["recommend", "suggest", "show", "want", "looking for", "find", "show me"]
            if any(word in question_lower for word in recommendation_words):
                return "attraction_recommendation"
            else:
                return "attraction_general"

        elif question_lower.startswith(("thank", "thanks")):
            return "gratitude"
        elif question_lower.strip() in ["yes", "no", "okay", "ok", "sure"]:
            return "confirmation"
        elif any(word in question_lower for word in ["bye", "goodbye", "exit", "quit"]):
            return "farewell"

        return "general_query"

    def analyze_question_type(self, question, task):
        question_lower = question.lower()

        booking_patterns = [
            r'\bbook\b.*\bfor\s+\d+\s+people\b',  # "book for 2 people"
            r'\breservation\s+for\s+\d+\b',  # "reservation for 3"
            r'\btable\s+for\s+\d+\b',  # "table for 4"
            r'\b\d+\s+people\b',  # "3 people"
            r'\broom\s+for\s+\d+\s+nights?\b'  # "room for 2 nights"
        ]

        if any(re.search(pattern, question_lower) for pattern in booking_patterns):
            return "booking_request"
        info_patterns = [
            r'\bwhat\s+is\s+the\s+(address|phone|number)\b',
            r'\bwhere\s+is\b',
            r'\bhow\s+much\b',
            r'\bwhat\s+type\b',
            r'\bwhat\s+kind\b'
        ]

        if any(re.search(pattern, question_lower) for pattern in info_patterns):
            return "information_request"

        recommendation_patterns = [
            r'\brecommend\b',
            r'\bsuggest\b',
            r'\bshow\s+me\b',
            r'\bfind\s+me\b',
            r'\bi\s+(want|need|am\s+looking\s+for)\b'
        ]

        if any(re.search(pattern, question_lower) for pattern in recommendation_patterns):
            if task == "restaurant" and any(word in question_lower for word in ["cheap", "expensive", "moderate"]):
                return "price_inquiry"
            else:
                return "recommendation_request"
        if question_lower.strip() in ["yes", "yes please", "no", "okay", "ok", "sure", "that sounds good", "perfect"]:
            return "confirmation"

        if task == "restaurant":
            if any(word in question_lower for word in ["cheap", "expensive", "moderate"]) and \
                    not any(word in question_lower for word in ["recommend", "suggest", "show"]):
                return "price_inquiry"
            elif any(word in question_lower for word in ["italian", "chinese", "indian", "british", "french"]):
                return "cuisine_preference"
        elif task == "hotel":
            if any(word in question_lower for word in ["star", "stars", "rating"]):
                return "rating_inquiry"
            elif any(word in question_lower for word in ["parking", "internet", "wifi"]):
                return "facilities_inquiry"
        elif task == "attraction":
            if any(word in question_lower for word in ["free", "cost", "entrance", "fee"]):
                return "pricing_inquiry"

        return "general"

    def _validate_restaurant_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        has_restaurant_name = re.search(r'\b\w+_\w+\b', answer_lower)

        has_food_type = any(
            food in answer_lower for food in ["italian", "chinese", "indian", "british", "french", "thai"])
        has_price_info = any(price in answer_lower for price in ["cheap", "moderate", "expensive"])
        has_area_info = any(area in answer_lower for area in ["centre", "north", "south", "east", "west"])

        if any(word in question.lower() for word in ["recommend", "suggest", "want"]):
            if not (has_restaurant_name or has_food_type):
                return False, "missing restaurant recommendation details"

        if any(word in question.lower() for word in ["address", "where", "located"]):
            has_address = re.search(r'\b\d+\s+\w+\s+(street|road|avenue)\b', answer_lower)
            if not has_address and "address" not in answer_lower:
                return False, "missing address information"

        if any(word in question.lower() for word in ["book", "table", "reservation"]):
            booking_terms = ["book", "table", "reservation", "reference", "confirmed"]
            if not any(term in answer_lower for term in booking_terms):
                return False, "missing booking information"

        return True, "restaurant answer validated"

    def _validate_hotel_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        has_hotel_name = re.search(r'\b\w+_\w+\b', answer_lower)

        has_hotel_type = any(htype in answer_lower for htype in ["hotel", "guesthouse", "lodge"])
        has_star_info = re.search(r'\b\d+[_\s]*star\b', answer_lower)
        has_area_info = any(area in answer_lower for area in ["centre", "north", "south", "east", "west"])

        if any(word in question.lower() for word in ["recommend", "suggest", "need"]):
            if not (has_hotel_name or has_hotel_type):
                return False, "missing hotel recommendation details"

        if any(word in question.lower() for word in ["book", "booking", "reservation"]):
            booking_terms = ["book", "booking", "reservation", "reference", "confirmed", "room"]
            if not any(term in answer_lower for term in booking_terms):
                return False, "missing booking information"

        if "star" in question.lower():
            if not has_star_info:
                return False, "missing star rating information"

        return True, "hotel answer validated"

    def _validate_attraction_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        has_attraction_name = re.search(r'\b\w+_\w+\b', answer_lower)

        has_attraction_type = any(
            atype in answer_lower for atype in ["museum", "college", "park", "church", "entertainment"])
        has_area_info = any(area in answer_lower for area in ["centre", "north", "south", "east", "west"])

        if any(word in question.lower() for word in ["recommend", "suggest", "show"]):
            if not (has_attraction_name or has_attraction_type):
                return False, "missing attraction recommendation details"

        if any(word in question.lower() for word in ["fee", "cost", "price", "entrance"]):
            fee_terms = ["free", "entrance", "fee", "cost", "price", "ticket"]
            if not any(term in answer_lower for term in fee_terms):
                return False, "missing fee information"

        if any(word in question.lower() for word in ["address", "where", "located"]):
            if not any(term in answer_lower for term in ["address", "street", "road", "located"]) and not has_area_info:
                return False, "missing location information"

        return True, "attraction answer validated"

    def detect_requires_history_context(self, question):
        question_lower = question.lower()

        if len(question.split()) <= 3:
            return True

        if any(word in question_lower for word in ["it", "that", "this", "them", "they"]):
            return True

        time_patterns = [
            r'^\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+\d{1,2}(am|pm)\s*$',
            r'^\s*\d{1,2}:\d{2}\s*(am|pm)\s*$',
            r'^\s*(today|tomorrow)\s+at\s+\d{1,2}(am|pm)\s*$'
        ]

        if any(re.match(pattern, question_lower) for pattern in time_patterns):
            return True

        return False

    def validate_answer(self, answer, task, knowledge_summary, question):
        if not answer or len(answer.strip()) < 3:
            return False, "answer is null or too short"

        if self.detect_question_intent(question) == "gratitude":
            return True, "gratitude response"

        if "need more specific information" in answer.lower():
            return False, "generic fallback response"

        if task == "restaurant":
            return self._validate_restaurant_answer(answer, question, knowledge_summary)
        elif task == "hotel":
            return self._validate_hotel_answer(answer, question, knowledge_summary)
        elif task == "attraction":
            return self._validate_attraction_answer(answer, question, knowledge_summary)

        return True, "general answer accepted"

    def _validate_navigation_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        address_pattern = r'\b\d+\s+[a-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|ln|ct|way|blvd|place|pl)\b'
        has_address = re.search(address_pattern, answer_lower, re.IGNORECASE)

        if any(word in question.lower() for word in ["where", "address", "location"]) and not has_address:
            clarification_phrases = ["which", "what", "where", "specify", "clarify", "need more"]
            asks_clarification = any(phrase in answer_lower for phrase in clarification_phrases)

            if not asks_clarification and not any(
                    entity in answer_lower for entity in ["miles", "restaurant", "garage", "station"]):
                return False, "missing address information"

        return True, "navigation answer validated"

    def _validate_weather_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        has_temp = re.search(r'\b\d+\s*°?f\b|\b\d+\s*degrees\b', answer_lower)

        weather_conditions = ["sunny", "cloudy", "rainy", "clear", "stormy", "foggy", "snowy", "windy"]
        has_condition = any(cond in answer_lower for cond in weather_conditions)

        if "weather" in question.lower():
            if not has_temp and not has_condition:
                location_questions = ["which city", "what location", "where"]
                asks_location = any(phrase in answer_lower for phrase in location_questions)

                if not asks_location:
                    return False, "missing weather information"

        return True, "weather answer validated"

    def _validate_schedule_answer(self, answer, question, knowledge_summary):
        answer_lower = answer.lower()

        confirmation_phrases = ["scheduled", "set", "confirmed", "reminder", "appointment"]
        has_confirmation = any(phrase in answer_lower for phrase in confirmation_phrases)

        has_time = re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', answer_lower)

        date_words = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow"]
        has_date = any(word in answer_lower for word in date_words)

        if any(word in question.lower() for word in ["schedule", "appointment", "meeting", "remind"]):
            if has_confirmation and not (has_time or has_date):
                info_questions = ["what time", "what date", "when", "which day"]
                asks_info = any(phrase in answer_lower for phrase in info_questions)

                if not asks_info:
                    return False, "missing scheduling details"

        return True, "schedule answer validated"

    def enhance_answer_with_entities(self, answer, knowledge_summary, task):
        if not knowledge_summary or not answer:
            return answer

        entities_by_relation = {}
        for triple in knowledge_summary.split(';'):
            parts = triple.strip().split()
            if len(parts) >= 3:
                subj, rel, obj = parts[0], parts[1], parts[2]
                if rel not in entities_by_relation:
                    entities_by_relation[rel] = []
                entities_by_relation[rel].append((subj, obj))

        if task == "navigate":
            return self._enhance_navigation_answer(answer, entities_by_relation)
        elif task == "weather":
            return self._enhance_weather_answer(answer, entities_by_relation)
        elif task == "schedule":
            return self._enhance_schedule_answer(answer, entities_by_relation)

        return answer

    def _enhance_navigation_answer(self, answer, entities):
        answer_lower = answer.lower()

        if "address" in entities and not re.search(r'\b\d+\s+[a-z\s]+(?:street|st|avenue|ave)', answer_lower):
            for subj, addr in entities["address"]:
                if subj.lower() in answer_lower:
                    # Format address properly (remove underscores)
                    formatted_addr = addr.replace('_', ' ')
                    return f"{answer} The address is {formatted_addr}."

        if "distance" in entities and "miles" not in answer_lower:
            for subj, dist in entities["distance"]:
                if subj.lower() in answer_lower:
                    return f"{answer} It's {dist} away."

        if "traffic_info" in entities and "traffic" not in answer_lower:
            for subj, traffic in entities["traffic_info"]:
                if subj.lower() in answer_lower:
                    return f"{answer} Traffic condition: {traffic}."

        return answer

    def _enhance_weather_answer(self, answer, entities):
        answer_lower = answer.lower()

        if "temperature" in entities and not re.search(r'\b\d+\s*°?f\b', answer_lower):
            for subj, temp in entities["temperature"]:
                return f"{answer} Temperature: {temp}."

        if "weather_attribute" in entities:
            weather_words = ["sunny", "cloudy", "rainy", "clear", "stormy"]
            if not any(word in answer_lower for word in weather_words):
                for subj, condition in entities["weather_attribute"]:
                    return f"{answer} Conditions: {condition}."

        return answer

    def _enhance_schedule_answer(self, answer, entities):
        answer_lower = answer.lower()

        if "time" in entities and not re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', answer_lower):
            for subj, time_val in entities["time"]:
                return f"{answer} Time: {time_val}."

        if "date" in entities:
            date_words = ["monday", "tuesday", "wednesday", "thursday", "friday"]
            if not any(word in answer_lower for word in date_words):
                for subj, date_val in entities["date"]:
                    return f"{answer} Date: {date_val}."

        return answer
