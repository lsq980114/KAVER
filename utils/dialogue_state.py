#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
utils/dialogue_state.py

DialogueState class for managing multi-turn dialogue context and knowledge graph integration.
"""
import re
import logging
from .kg_utils import detect_start_entity_spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueState:
    """
    Class for tracking dialogue state across multiple turns, including
    conversation history, knowledge graph, and domain information.
    """
    def __init__(self, max_kg_size=30, tokenizer=None):
        """
        Initialize a new dialogue state.
        
        Args:
            max_kg_size: Maximum number of KG triples to store
            tokenizer: Tokenizer for processing text
        """
        self.history = []
        self.last_question = ""
        self.accumulated_kg = []
        self.topic_entities = set()  # Track important entities mentioned in dialogue
        self.focused_entities = set()  # Currently focused entities
        self.max_kg_size = max_kg_size
        self.tokenizer = tokenizer
        self.domains = {  # Track confidence in each domain
            "weather": 0,
            "traffic": 0,
            "schedule": 0,
            "restaurant": 0,
            "navigate": 0
        }
        self.current_domain = None
        self.locations = {}  # Locations with last mention time
        self.current_location = None
        self.dates = {}  # Dates with last mention time
        self.current_date = None
        self.turn_count = 0
        
    def detect_entities(self, text):
        """
        Detect entities mentioned in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Set of entity names
        """
        entities = set()
        
        # 1. Detect underscored words as entities
        underscored = re.findall(r'\b([a-zA-Z0-9]+_[a-zA-Z0-9_]+)\b', text)
        entities.update(underscored)
        
        # 2. Detect capitalized noun phrases
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.update(capitalized)
        
        # 3. Detect known locations
        known_locations = {'san_francisco', 'new_york', 'danville', 'brentwood', 'atherton', 
                         'menlo_park', 'oakland', 'manhattan', 'chicago', 'carson'}
        for loc in known_locations:
            if loc in text or loc.replace('_', ' ') in text:
                entities.add(loc)
        
        # 4. Detect dates and times
        date_patterns = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                      'saturday', 'sunday', 'today', 'tomorrow', 'next_week'}
        for date in date_patterns:
            if date in text or date.replace('_', ' ') in text:
                entities.add(date)
        
        return entities
    
    def detect_domain(self, text):
        """
        Detect domain of a question.
        
        Args:
            text: Question text
            
        Returns:
            Domain name (string)
        """
        text_lower = text.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "weather": ["weather", "temperature", "rain", "snow", "sunny", "cloudy", "forecast"],
            "traffic": ["traffic", "road", "congestion", "car", "drive", "route"],
            "schedule": ["schedule", "appointment", "meeting", "reserve", "book"],
            "restaurant": ["restaurant", "food", "eat", "dinner", "lunch", "café", "cafe", "coffee"],
            "navigate": ["directions", "locate", "find", "address", "where", "nearest", "closest"]
        }
        
        # Calculate domain scores
        scores = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[domain] += 1
                    
        # Update domain confidence (exponential decay)
        decay = 0.7
        for domain in self.domains:
            self.domains[domain] *= decay
            self.domains[domain] += scores.get(domain, 0)
        
        # Choose domain with highest confidence
        max_domain = max(self.domains.items(), key=lambda x: x[1])
        if max_domain[1] > 0:
            self.current_domain = max_domain[0]
        
        return self.current_domain
    
    def detect_location(self, text):
        """
        Detect location mentioned in text.
        
        Args:
            text: Input text
            
        Returns:
            Location name or None
        """
        entities = self.detect_entities(text)
        
        # Location candidates
        location_candidates = {
            'san_francisco', 'new_york', 'danville', 'brentwood', 'atherton', 
            'menlo_park', 'oakland', 'manhattan', 'chicago', 'carson', 'gas_station',
            'restaurant', 'mall', 'conference_room', 'valero'
        }
        
        # Check if any entity is a location
        for entity in entities:
            if entity in location_candidates or any(loc in entity for loc in ['street', 'ave', 'road', 'center']):
                self.locations[entity] = self.turn_count
                self.current_location = entity
                return entity
        
        # If no new location found, use most recently mentioned
        if self.locations:
            recent_location = max(self.locations.items(), key=lambda x: x[1])
            self.current_location = recent_location[0]
            return recent_location[0]
        
        return None
    
    def detect_date(self, text):
        """
        Detect date mentioned in text.
        
        Args:
            text: Input text
            
        Returns:
            Date string or 'today'
        """
        text_lower = text.lower()
        
        # Date patterns
        date_patterns = {
            'monday': 'monday', 'tuesday': 'tuesday', 'wednesday': 'wednesday',
            'thursday': 'thursday', 'friday': 'friday', 'saturday': 'saturday', 
            'sunday': 'sunday', 'today': 'today', 'tomorrow': 'tomorrow'
        }
        
        # Check for date patterns
        for pattern, date in date_patterns.items():
            if pattern in text_lower:
                self.dates[date] = self.turn_count
                self.current_date = date
                return date
        
        # If no new date found, use most recently mentioned
        if self.dates:
            recent_date = max(self.dates.items(), key=lambda x: x[1])
            self.current_date = recent_date[0]
            return recent_date[0]
        
        return 'today'  # Default to today
    
    def update(self, new_example):
        """
        Update dialogue state with new example.
        
        Args:
            new_example: New dialogue example
            
        Returns:
            None
        """
        self.turn_count += 1
        
        # Update history
        if "history" in new_example:
            history_items = []
            for item in new_example["history"]:
                if isinstance(item, list):
                    if self.tokenizer:
                        decoded_text = self.tokenizer.decode(item, skip_special_tokens=True)
                        history_items.append(decoded_text)
                elif isinstance(item, str):
                    history_items.append(item)
            self.history.extend(history_items)
        
        # Get current question
        question = new_example.get("current_question", "")
        self.last_question = question
        
        # Analyze question
        self.detect_domain(question)
        self.detect_location(question)
        self.detect_date(question)
        
        # Extract entities from question
        new_entities = self.detect_entities(question)
        self.topic_entities.update(new_entities)
        self.focused_entities = new_entities
        
        # Update knowledge graph
        new_kg = new_example.get("knowledge_text", []) or new_example.get("kg", [])
        
        # Select relevant triples
        relevant_kg = []
        for triple in new_kg:
            # Ensure triple is valid
            if triple and len(triple) >= 3:
                subj, rel, obj = triple[:3]
                subj_str = str(subj).lower()
                rel_str = str(rel).lower()
                obj_str = str(obj).lower()
                
                # Entity matching
                is_topic_match = any(entity.lower() in subj_str or entity.lower() in obj_str 
                                    for entity in self.focused_entities)
                
                # Domain matching
                is_domain_match = self.current_domain and self.current_domain.lower() in rel_str
                
                # Location matching
                is_location_match = self.current_location and (
                    self.current_location.lower() in subj_str or self.current_location.lower() in obj_str)
                
                # Date matching
                is_date_match = self.current_date and self.current_date.lower() in rel_str
                
                # Calculate triple relevance score
                score = 0
                if is_topic_match: score += 3
                if is_domain_match: score += 2
                if is_location_match: score += 2
                if is_date_match: score += 2
                
                if score > 0:
                    relevant_kg.append((triple, score))
        
        # Sort by relevance
        relevant_kg.sort(key=lambda x: x[1], reverse=True)
        filtered_kg = [t[0] for t in relevant_kg]
        
        # Add to accumulated KG
        self.accumulated_kg.extend(filtered_kg)
        
        # Ensure KG size doesn't exceed limit
        if len(self.accumulated_kg) > self.max_kg_size:
            # Keep most recent triples
            self.accumulated_kg = self.accumulated_kg[-self.max_kg_size:]
    
    def get_relevant_kg(self):
        """
        Get triples most relevant to current question.
        
        Returns:
            List of triple tuples
        """
        scored_triples = []
        
        for triple in self.accumulated_kg:
            if not triple or len(triple) < 3:
                continue
                
            subj, rel, obj = triple[:3]
            subj_str = str(subj).lower()
            rel_str = str(rel).lower()
            obj_str = str(obj).lower()
            
            # Score relevance
            score = 0
            
            # Entity matching
            for entity in self.focused_entities:
                entity_lower = entity.lower()
                if entity_lower in subj_str or entity_lower in obj_str:
                    score += 2
            
            # Domain matching
            if self.current_domain and self.current_domain.lower() in rel_str:
                score += 1.5
            
            # Location matching
            if self.current_location:
                loc_lower = self.current_location.lower()
                if loc_lower in subj_str or loc_lower in obj_str:
                    score += 1.5
            
            # Date matching
            if self.current_date and self.current_date.lower() in rel_str:
                score += 1.5
            
            if score > 0:
                scored_triples.append((triple, score))
        
        # Sort by relevance
        scored_triples.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N most relevant triples
        return [t[0] for t in scored_triples[:10]]
    
    def get_current_state(self):
        """
        Get current dialogue state.
        
        Returns:
            Dictionary with current state
        """
        return {
            "history": self.history,
            "last_question": self.last_question,
            "domain": self.current_domain,
            "location": self.current_location,
            "date": self.current_date,
            "topic_entities": list(self.topic_entities),
            "focused_entities": list(self.focused_entities),
            "relevant_kg": self.get_relevant_kg()
        }
        
    def get_relevant_history(self, question, max_turns=2, task=None):
        """
        Get relevant dialogue history based on task type.
        
        Args:
            question: Current question
            max_turns: Maximum number of turns to include
            task: Task type (optional)
            
        Returns:
            String with relevant history
        """
        # Use provided task, state task, or analyze
        current_task = task or getattr(self, 'current_task', None)
        
        if not current_task:
            # If no task type provided, analyze from question
            from utils.text_utils import analyze_task_type
            current_task = analyze_task_type(question)
        
        # Get most recent history
        recent_history = self.history[-max_turns*2:] if len(self.history) >= max_turns*2 else self.history
        
        # For small history or unknown task, return all recent
        if not current_task or current_task == "unknown" or len(recent_history) <= max_turns:
            return " ".join(recent_history[-max_turns:])
        
        # Filter by task relevance
        task_related = []
        for turn in recent_history:
            # Check if turn is related to current task
            if current_task.lower() in turn.lower():
                task_related.append(turn)
        
        # If enough task-related history found, use it
        if len(task_related) >= 2:
            return " ".join(task_related[-max_turns:])
        
        # Otherwise return most recent
        return " ".join(recent_history[-max_turns:])
    
    def extract_current_context(self, question, task):
        """
        Extract context for current question.
        
        Args:
            question: Current question
            task: Task type
            
        Returns:
            Dictionary with context information
        """
        domain = self.current_domain or task
        location = self.current_location or detect_start_entity_spacy(question, " ".join(self.history))
        
        # Infer date from question
        from utils.text_utils import infer_date_from_now
        date = self.current_date or infer_date_from_now(question)
        
        context = {
            "domain": domain,
            "location": location,
            "date": date,
            "recent_history": self.history[-3:] if len(self.history) >= 3 else self.history
        }
        return context