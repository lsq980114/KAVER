#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
generation/enhanced_generator.py

"""
import os
import gc
import re
import logging
import torch
import psutil
from generation.prompt_engineering import EnhancedPromptEngineering


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedResponseGenerator:
    def __init__(self,
                 base_model_path="/home/Experiments/KAVER-main/runs/Qwen1.5-7B",
                 adapters_dir="/home/Experiments/KAVER-main/adapters"):

        self.adapter_finetuner = None
        self.base_model_path = base_model_path
        self.adapters_dir = adapters_dir

        # Initialize tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Initialized tokenizer from {base_model_path}")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            self.tokenizer = None

        # Track loaded adapters
        self.loaded_adapters = {}
        self.current_model = None
        self.current_domain = None

        # Initialize device tracking
        self.primary_input_device = None

        # Create prompt engineering module
        self.prompt_engineer = EnhancedPromptEngineering()

    def get_domain_for_task(self, task):
        task_to_adapter_mapping = {

            "korean": "asian_oriental",
            "japanese": "asian_oriental",

            "modern_european": "modern_european",

            "mediterranean": "mediterranean",

            "gastropub": "gastropub",

            "lebanese": "international",
            "african": "african",

            "chinese": "chinese",
            "italian": "italian",
            "british": "british",
            "indian": "indian",
            "french": "french",
            "spanish": "spanish",
            "thai": "thai",
            "vietnamese": "vietnamese",
            "turkish": "turkish",
            "portuguese": "portuguese",
            "dontcare": "dontcare",

            "": "dontcare",
            "unknown": "dontcare"
        }

        return task_to_adapter_mapping.get(task, task)

    def generate_missing_info_query(self, prompt, task, missing_types, temperature=0.7, max_new_tokens=30):

        model = self._get_model_for_task(task)

        if model is None:
            if "location" in missing_types:
                return "What city do you want the weather for?"
            return "Could you provide more specific information?"

        input_device = getattr(self, 'primary_input_device', None)
        if input_device is None:
            try:
                if hasattr(model, 'get_input_embeddings'):
                    input_device = model.get_input_embeddings().weight.device
                else:
                    for param in model.parameters():
                        input_device = param.device
                        break
            except Exception:
                input_device = torch.device('cuda:0')

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)

            outputs_cpu = outputs.cpu()
            full_text = self.tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)

            answer = self._extract_answer_from_generation(full_text, prompt)
            answer = self._clean_and_validate_answer(answer, "", task)

            if answer and not answer.strip().endswith('?'):
                if "city" in answer.lower() or "location" in answer.lower():
                    answer = answer.strip().rstrip('.') + '?'

            del outputs, outputs_cpu, inputs
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            logger.error(f"Error generating missing info query: {e}")
            import traceback
            logger.error(traceback.format_exc())

            if "location" in missing_types:
                return "What city do you want the weather for?"
            return "Could you provide more specific information?"


    def _build_full_context_prompt(self, reasoning_prompt, history_text, location, knowledge_summary):
        context_parts = []

        if history_text and history_text.strip():
            context_parts.append(f"[CONVERSATION HISTORY]\n{history_text}")

        if location and location.strip():
            context_parts.append(f"[LOCATION CONTEXT]\n{location}")

        if knowledge_summary and knowledge_summary.strip():
            context_parts.append(f"[KNOWLEDGE SUMMARY]\n{knowledge_summary}")

        context_parts.append(reasoning_prompt)

        return "\n\n".join(context_parts)

    def generate_answer_with_reasoning(self, question, history_text, domain, location,
                                       chain_text, knowledge_summary, reasoning_data,
                                       task, temperature=0.7, max_new_tokens=50,
                                       num_samples=3, debug_mode=False):
        try:
            enhanced_prompt_data = integrate_reasoning_with_llm_prompt([reasoning_data])[0]

            enhanced_chain_text = chain_text
            if 'reasoning_chain' in reasoning_data:
                if enhanced_chain_text:
                    enhanced_chain_text = f"{enhanced_chain_text} ||| Enhanced reasoning: {reasoning_data['reasoning_chain']}"
                else:
                    enhanced_chain_text = f"Enhanced reasoning: {reasoning_data['reasoning_chain']}"

            if debug_mode:
                logger.info(f"DEBUG - Using reasoning-enhanced generation with enhanced chain: {enhanced_chain_text}")

            return self.generate_answer(
                question=question,
                history_text=history_text,
                domain=domain,
                location=location,
                chain_text=enhanced_chain_text,
                knowledge_summary=knowledge_summary,
                task=task,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_samples=num_samples,
                debug_mode=debug_mode
            )

        except Exception as e:
            logger.warning(f"Reasoning-enhanced generation failed: {e}, falling back to standard generation")
            return self.generate_answer(
                question, history_text, domain, location,
                chain_text, knowledge_summary, task,
                temperature, max_new_tokens, num_samples, debug_mode
            )

    def _get_model_for_task(self, task):
        trained_camrest_adapters = [
            "dontcare", "italian", "chinese", "indian", "european", "asian_oriental",
            "british", "turkish", "spanish", "vietnamese", "international",
            "mediterranean", "thai", "french", "portuguese", "modern_european",
            "gastropub", "african"
        ]

        if task in trained_camrest_adapters:
            actual_task = task
            logger.info(f"Using trained CamRest {task} adapter")
        elif task in ["korean", "japanese"]:
            actual_task = "asian_oriental"
            logger.info(f"Mapping {task} to asian_oriental adapter")
        elif task in ["lebanese", "mexican", "north_american", "steakhouse", "seafood"]:
            actual_task = "international"
            logger.info(f"Mapping {task} to international adapter")
        elif task in ["irish"]:
            actual_task = "british"
            logger.info(f"Mapping {task} to british adapter")
        elif task == "":
            actual_task = "dontcare"
            logger.info(f"Mapping empty task to dontcare adapter")
        else:
            woz_domains = ["restaurant", "hotel", "attraction"]
            if task in woz_domains:
                actual_task = task
            else:
                logger.warning(f"Unknown task {task}, using base model")
                actual_task = "base"

        if self.current_model is not None and self.current_domain == actual_task:
            return self.current_model

        if actual_task in trained_camrest_adapters or actual_task in ["restaurant", "hotel", "attraction"]:
            adapter_path = f"{self.adapters_dir}/{actual_task}"
        else:
            adapter_path = None

        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Loading {actual_task} domain adapter for task '{task}'...")

            try:
                if self.current_model is not None:
                    del self.current_model
                    self.current_model = None
                    self.current_domain = None

                torch.cuda.empty_cache()
                gc.collect()

                from transformers import AutoModelForCausalLM
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cuda:0"}
                )

                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, adapter_path)

                if hasattr(model, 'get_input_embeddings'):
                    self.primary_input_device = model.get_input_embeddings().weight.device
                else:
                    for param in model.parameters():
                        self.primary_input_device = param.device
                        break

                logger.info(f"Detected primary input device: {self.primary_input_device}")

                self.current_model = model
                self.current_domain = actual_task
                logger.info(f"{actual_task} adapter loaded successfully for task '{task}'")
                return model

            except Exception as e:
                logger.error(f"Error loading {actual_task} adapter: {e}")
                import traceback
                logger.error(traceback.format_exc())

                try:
                    if self.current_model is None:
                        from transformers import AutoModelForCausalLM
                        self.current_model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_path,
                            torch_dtype=torch.float16,
                            device_map={"": "cuda:0"}  # Force single GPU
                        )

                        if hasattr(self.current_model, 'get_input_embeddings'):
                            self.primary_input_device = self.current_model.get_input_embeddings().weight.device
                        else:
                            for param in self.current_model.parameters():
                                self.primary_input_device = param.device
                                break

                        self.current_domain = "base"
                        return self.current_model
                except Exception as e2:
                    logger.error(f"Error loading base model: {e2}")
                    return None

        logger.info(f"No {actual_task} domain adapter found for task '{task}', using base model")
        try:
            if self.current_model is None:
                from transformers import AutoModelForCausalLM
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cuda:0"}  # Force single GPU
                )

                if hasattr(self.current_model, 'get_input_embeddings'):
                    self.primary_input_device = self.current_model.get_input_embeddings().weight.device
                else:
                    for param in self.current_model.parameters():
                        self.primary_input_device = param.device
                        break

                self.current_domain = "base"
            return self.current_model
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            return None

    def _create_training_compatible_prompt(self, question, history_text, domain, location,
                                           chain_text, knowledge_summary, task):
        context_parts = []

        if history_text and history_text.strip():
            context_parts.append(f"Previous conversation: {history_text}")

        if domain and domain.strip():
            context_parts.append(f"Domain: {domain}")

        if location and location.strip():
            context_parts.append(f"Location: {location}")

        if knowledge_summary and knowledge_summary.strip():
            if task == "restaurant":
                enhanced_knowledge = []
                for item in knowledge_summary.split(";"):
                    item = item.strip()
                    if item:

                        if any(rel in item for rel in ["food", "pricerange", "area", "address", "phone"]):
                            enhanced_knowledge.append(f"{item} [IMPORTANT]")
                        else:
                            enhanced_knowledge.append(item)
                context_parts.append(f"Available information: {'; '.join(enhanced_knowledge)}")

            elif task == "hotel":

                enhanced_knowledge = []
                for item in knowledge_summary.split(";"):
                    item = item.strip()
                    if item:

                        if any(rel in item for rel in ["type", "stars", "pricerange", "area", "address", "phone"]):
                            enhanced_knowledge.append(f"{item} [IMPORTANT]")
                        else:
                            enhanced_knowledge.append(item)
                context_parts.append(f"Available information: {'; '.join(enhanced_knowledge)}")

            elif task == "attraction":
                enhanced_knowledge = []
                for item in knowledge_summary.split(";"):
                    item = item.strip()
                    if item:

                        if any(rel in item for rel in ["type", "area", "address", "phone"]):
                            enhanced_knowledge.append(f"{item} [IMPORTANT]")
                        else:
                            enhanced_knowledge.append(item)
                context_parts.append(f"Available information: {'; '.join(enhanced_knowledge)}")
            else:
                context_parts.append(f"Available information: {knowledge_summary}")

        if chain_text and chain_text.strip():
            context_parts.append(f"Reasoning: {chain_text}")

        if task == "restaurant":
            question_lower = question.lower()
            if any(word in question_lower for word in ["recommend", "suggest", "want", "looking for"]):
                context_parts.append(
                    "Task guidance: Provide specific restaurant recommendation with food type, price, and area")
            elif any(word in question_lower for word in ["address", "where", "located"]):
                context_parts.append("Task guidance: Provide exact address and contact information")
            elif any(word in question_lower for word in ["book", "table", "reservation"]):
                context_parts.append("Task guidance: Help with booking process and ask for required details")

        elif task == "hotel":
            question_lower = question.lower()
            if any(word in question_lower for word in ["book", "reserve", "stay"]):
                context_parts.append("Task guidance: Help with hotel booking and ask for dates/people")
            elif any(word in question_lower for word in ["recommend", "suggest", "need"]):
                context_parts.append("Task guidance: Provide hotel recommendation with type, stars, and area")
            elif any(word in question_lower for word in ["address", "where", "located"]):
                context_parts.append("Task guidance: Provide exact address and contact information")

        elif task == "attraction":
            question_lower = question.lower()
            if any(word in question_lower for word in ["recommend", "suggest", "show", "visit"]):
                context_parts.append("Task guidance: Provide attraction recommendation with type and area")
            elif any(word in question_lower for word in ["address", "where", "located"]):
                context_parts.append("Task guidance: Provide exact address and contact information")
            elif any(word in question_lower for word in ["fee", "cost", "price", "entrance"]):
                context_parts.append("Task guidance: Provide entrance fee and opening information")

        context = " | ".join(context_parts) if context_parts else ""

        if context:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        return prompt

    def generate_answer(self, question, history_text, domain, location,
                        chain_text, knowledge_summary, task,
                        temperature=0.7,
                        max_new_tokens=50,        
                        num_samples=5,          
                        use_self_consistency=True,
                        debug_mode=False):

        prompt = self._create_training_compatible_prompt(
            question, history_text, domain,
            location, chain_text, knowledge_summary, task
        )

        model = self._get_model_for_task(task)
        if model is None:
            logger.error(f"No model available for task: {task}")
            return "I need more specific information to answer your question properly."

        device = self.primary_input_device if self.primary_input_device is not None \
                else torch.device("cuda:0")

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if use_self_consistency and num_samples > 1:
            candidate_responses = []
            candidate_log_probs = []

            generation_config = {
                "max_new_tokens": max_new_tokens,   
                "temperature": temperature,          
                "top_p": 0.9,                       
                "top_k": 50,                        
                "repetition_penalty": 1.1,         
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            for i in range(num_samples):
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_config)

                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True
                )
                avg_log_prob = transition_scores[0].mean().item()

                full_text = self.tokenizer.decode(
                    outputs.sequences[0], skip_special_tokens=True
                )

                response = self._extract_clean_response(full_text, prompt)

                candidate_responses.append(response)
                candidate_log_probs.append(avg_log_prob)

                del outputs
                torch.cuda.empty_cache()

                if debug_mode:
                    logger.info(
                        f"  Sample {i+1}/{num_samples} "
                        f"(log_prob={avg_log_prob:.3f}): {response[:80]}..."
                    )

            from self_consistency.self_consistency_integration import self_consistency_aggregation

            final_response = self_consistency_aggregation(
                responses=candidate_responses,
                task_type=task,
                similarity_threshold=0.8,
                response_log_probs=candidate_log_probs
            )

            if debug_mode:
                logger.info(f"  Final: {final_response[:100]}")

            return final_response

        else:
        
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.1),
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_clean_response(full_text, prompt)

    def _extract_clean_response(self, full_text: str, prompt: str) -> str:

        if prompt.rstrip("Answer:").strip() in full_text:
            response = full_text[len(prompt):].strip()

        elif "Answer:" in full_text:
            response = full_text.split("Answer:")[-1].strip()
        else:
            response = full_text.strip()


        stop_markers = ["\nQuestion:", "\nContext:", "\nAnswer:", "\nUser:", "\nAssistant:"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()

        if len(response.split()) > 60:
            sentences = response.split('.')
            if len(sentences) > 1 and len(sentences[0].strip()) > 5:
                response = sentences[0].strip() + '.'

        return response.strip()

    def _extract_answer_from_generation(self, full_text, prompt, debug_mode=False):

        if debug_mode:
            logger.debug(f"Full generation: {full_text}")
            logger.debug(f"Original prompt length: {len(prompt)}")

        if prompt in full_text:
            answer = full_text.replace(prompt, "", 1).strip()
            if debug_mode:
                logger.debug(f"Method 1 (prompt removal): {answer[:100]}...")
        else:

            if "Answer:" in full_text:
                parts = full_text.split("Answer:", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    if debug_mode:
                        logger.debug(f"Method 2 (Answer: split): {answer[:100]}...")
                else:
                    answer = full_text.strip()
            else:

                if "Question:" in full_text:
                    parts = full_text.split("Question:")
                    if len(parts) > 1:

                        last_part = parts[-1].strip()

                        if "\n" in last_part:
                            lines = last_part.split("\n")

                            for line in lines:
                                if line.strip() and not line.strip().startswith(("Context:", "Question:", "Answer:")):
                                    answer = line.strip()
                                    break
                            else:
                                answer = last_part.strip()
                        else:
                            answer = last_part.strip()
                        if debug_mode:
                            logger.debug(f"Method 3 (Question split): {answer[:100]}...")
                    else:
                        answer = full_text.strip()
                else:

                    sentences = full_text.split('.')
                    if len(sentences) > 1:
                        answer = sentences[-2].strip() + '.'
                    else:
                        answer = full_text.strip()
                    if debug_mode:
                        logger.debug(f"Method 4 (fallback): {answer[:100]}...")

        return answer

    def _clean_and_validate_answer(self, answer, question, task, debug_mode=False):

        if not answer or len(answer.strip()) < 2:
            return "I need more specific information to answer your question properly."

        answer = re.sub(r'(Context:|Question:|Answer:).*$', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = answer.strip()

        if len(answer.strip()) < 3:
            return "I need more specific information to answer your question properly."

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        if len(question_words.intersection(answer_words)) > len(question_words) * 0.8:
            if debug_mode:
                logger.debug("Answer seems to repeat question, using fallback")
            return "I need more specific information to answer your question properly."

        if re.search(r'[\u4e00-\u9fff]', answer):
            if debug_mode:
                logger.debug("Non-English text detected, using fallback")
            return "I need more specific information to answer your question properly."

        if task == "restaurant":

            restaurant_indicators = [
                "restaurant", "food", "italian", "indian", "british", "chinese",
                "expensive", "cheap", "moderate", "centre", "north", "south", "east", "west"
            ]
            has_restaurant_content = any(indicator in answer.lower() for indicator in restaurant_indicators)

            useful_patterns = [
                r'\b\w+_\w+\b',
                r'\b(cheap|moderate|expensive)\b',
                r'\b(centre|north|south|east|west)\b',
                r'\b\d+\s*\w+\s*(street|road|avenue)\b'
            ]
            has_useful_info = any(re.search(pattern, answer.lower()) for pattern in useful_patterns)

            if not (has_restaurant_content or has_useful_info):
                if debug_mode:
                    logger.debug("Restaurant answer lacks domain-specific content")
                return "I need more specific information about restaurants to help you."

        elif task == "hotel":

            hotel_indicators = [
                "hotel", "guesthouse", "lodge", "star", "stars",
                "booking", "room", "reservation", "stay"
            ]
            has_hotel_content = any(indicator in answer.lower() for indicator in hotel_indicators)

            useful_patterns = [
                r'\b\w+_\w+\b',
                r'\b\d+[_\s]*star\b',
                r'\b(hotel|guesthouse)\b',
                r'\breference\s*number\b'
            ]
            has_useful_info = any(re.search(pattern, answer.lower()) for pattern in useful_patterns)

            if not (has_hotel_content or has_useful_info):
                if debug_mode:
                    logger.debug("Hotel answer lacks domain-specific content")
                return "I need more specific information about hotels to help you."

        elif task == "attraction":

            attraction_indicators = [
                "museum", "college", "park", "church", "attraction",
                "visit", "see", "entrance", "fee"
            ]
            has_attraction_content = any(indicator in answer.lower() for indicator in attraction_indicators)

            useful_patterns = [
                r'\b\w+_\w+\b',
                r'\b(museum|college|park|church)\b',
                r'\bfree\b',
                r'\bphone\s*number\b'
            ]
            has_useful_info = any(re.search(pattern, answer.lower()) for pattern in useful_patterns)

            if not (has_attraction_content or has_useful_info):
                if debug_mode:
                    logger.debug("Attraction answer lacks domain-specific content")
                return "I need more specific information about attractions to help you."

        if answer and not answer.rstrip().endswith(('.', '!', '?')):

            sentences = answer.split('.')
            if len(sentences) > 1 and len(sentences[0].strip()) > 10:
                answer = sentences[0].strip() + '.'
            else:
                answer = answer.rstrip() + '.'

        return answer

    def _select_best_answer(self, candidates, task, question, knowledge_summary, chain_text=None):

        if len(candidates) <= 1:
            return candidates[0] if candidates else "I need more specific information to answer your question properly."

        non_generic = [c for c in candidates if
                       c != "I need more specific information to answer your question properly."]
        if not non_generic:
            return candidates[0]

        if len(non_generic) == 1:
            return non_generic[0]

        scored_candidates = []

        for answer in non_generic:
            score = 0

            word_count = len(answer.split())
            if 5 <= word_count <= 50:
                score += 2
            elif word_count > 50:
                score += 1

            if answer.strip().endswith(('.', '!', '?')):
                score += 1

            if task == "restaurant":

                restaurant_terms = [
                    "restaurant", "food", "italian", "indian", "british", "chinese", "french",
                    "cheap", "moderate", "expensive", "centre", "north", "south", "east", "west",
                    "address", "phone", "recommend"
                ]
                score += sum(2 for term in restaurant_terms if term in answer.lower())

                if re.search(r'\b\w+_\w+\b', answer):
                    score += 3
                if "would you like" in answer.lower():
                    score += 1

            elif task == "hotel":
                hotel_terms = [
                    "hotel", "guesthouse", "lodge", "star", "stars", "booking", "room",
                    "cheap", "moderate", "expensive", "north", "south", "east", "west", "centre",
                    "address", "phone", "reservation", "reference"
                ]
                score += sum(2 for term in hotel_terms if term in answer.lower())

                if re.search(r'\b\d+[_\s]*star\b', answer.lower()):
                    score += 3
                if "booking" in answer.lower() or "reservation" in answer.lower():
                    score += 2

            elif task == "attraction":
                attraction_terms = [
                    "museum", "college", "park", "church", "attraction", "visit", "see",
                    "centre", "north", "south", "east", "west", "address", "phone",
                    "free", "entrance", "fee"
                ]
                score += sum(2 for term in attraction_terms if term in answer.lower())

                if "free" in answer.lower():
                    score += 2

            if knowledge_summary:
                knowledge_words = set(knowledge_summary.lower().replace('_', ' ').split())
                answer_words = set(answer.lower().replace('_', ' ').split())
                overlap = len(knowledge_words.intersection(answer_words))
                score += overlap * 0.5

            scored_candidates.append((answer, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def prepare_training_data(self, raw_examples):

        training_samples = []

        for example_id, example_data in raw_examples.items():
            task = example_data["task"]
            utterances = example_data["utterances"]
            kg_triples = example_data.get("kg", [])

            knowledge_summary = "; ".join([f"{triple[0]} {triple[1]} {triple[2]}" for triple in kg_triples])

            history = []
            for i, utterance in enumerate(utterances):
                user_input = utterance["user"]
                expected_response = utterance["response"]

                history_text = " | ".join([f"User: {h['user']} Assistant: {h['response']}" for h in history])

                if history_text:
                    prompt = f"Context: Previous conversation: {history_text} | Available information: {knowledge_summary}\nQuestion: {user_input}\nAnswer:"
                else:
                    if knowledge_summary:
                        prompt = f"Context: Available information: {knowledge_summary}\nQuestion: {user_input}\nAnswer:"
                    else:
                        prompt = f"Question: {user_input}\nAnswer:"

                training_samples.append({
                    "task": task,
                    "prompt": prompt,
                    "response": expected_response,
                    "input_text": f"{prompt} {expected_response}",
                    "example_id": example_id,
                    "utterance_index": i
                })

                history.append({"user": user_input, "response": expected_response})

        return training_samples

    def train_adapters(self, examples):

        training_samples = self.prepare_training_data(examples)

        task_samples = {}
        for sample in training_samples:
            task = sample["task"]
            if task not in task_samples:
                task_samples[task] = []
            task_samples[task].append(sample)

        if not hasattr(self, 'adapter_finetuner') or self.adapter_finetuner is None:
            from training.adapter_finetuner import DomainAdapterFineTuner
            self.adapter_finetuner = DomainAdapterFineTuner(
                base_model_path=self.base_model_path,
                output_dir=self.adapters_dir,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1
            )

        successful_adapters = self.adapter_finetuner.train_all_domain_adapters(task_samples)

        if successful_adapters and self.current_model is not None:
            self.clean_up_resources()

        return successful_adapters

    def clean_up_resources(self):

        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_domain = None
        torch.cuda.empty_cache()
        gc.collect()
