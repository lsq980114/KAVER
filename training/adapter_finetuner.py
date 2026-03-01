import os
import gc
import torch
import logging
import json
import psutil
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from peft import (
    get_peft_config, 
    PeftModel, 
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForCausalLM:

    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"] 
            labels = feature["labels"]
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }


class DomainAdapterFineTuner:

    def __init__(self, 
                 base_model_path="/home/Experiments/KAVER-main/runs/Qwen1.5-7B",
                 output_dir="adapters",
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.05):

        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.domain_adapters = {}
        self.model = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded from {base_model_path}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Adapter output directory: {output_dir}")
    
    def _create_consistent_training_prompt(self, user_query, system_response, domain, 
                                         knowledge_text="", history_text="", location=""):

        context_parts = []

        if history_text.strip():
            context_parts.append(f"Previous conversation: {history_text}")

        if domain:
            context_parts.append(f"Domain: {domain}")

        if location.strip():
            context_parts.append(f"Location: {location}")

        if knowledge_text.strip():
            context_parts.append(f"Available information: {knowledge_text}")

        if context_parts:
            context = " | ".join(context_parts)
            prompt = f"Context: {context}\nQuestion: {user_query}\nAnswer:"
        else:
            prompt = f"Question: {user_query}\nAnswer:"

        full_text = f"{prompt} {system_response}"
        
        return prompt, full_text

    def _check_domain_specific_quality(self, user_query, system_response, domain, reference_entities):

        user_lower = user_query.lower()
        response_lower = system_response.lower()
        
        if domain == "restaurant":
            restaurant_indicators = [
                "restaurant", "food",
                "expensive", "cheap", "moderate",  # pricerange
                "centre", "north", "south", "east", "west",  # area
                "address", "phone", "pricerange", "area","dontcare", "chinese", "italian", "indian", "european", "asian_oriental",
            "british", "turkish", "mediterranean", "spanish", "french", "international", 
            "vietnamese", "korean", "thai", "portuguese", "modern_european", "gastropub",
            "african", "mexican", "north_american", "steakhouse", "japanese", 
            "lebanese", "seafood", "irish"
            ]
            
            has_restaurant_content = (
                any(indicator in user_lower or indicator in response_lower 
                    for indicator in restaurant_indicators) or
                any(str(entity).lower() in ["restaurant", "food"] for entity in reference_entities)
            )
            
            return has_restaurant_content and len(system_response.split()) >= 5
        
        elif domain == "hotel":
            hotel_indicators = [
                "hotel", "guesthouse", "4_star", "3_star", "stars",
                "north", "east", "west", "centre",
                "moderate", "cheap", "expensive",
                "address", "phone", "stars", "type", "area", "pricerange"
            ]
            
            has_hotel_content = (
                any(indicator in user_lower or indicator in response_lower 
                    for indicator in hotel_indicators) or
                any(str(entity).lower() in ["hotel", "guesthouse"] for entity in reference_entities)
            )
            
            return has_hotel_content and len(system_response.split()) >= 5
        
        elif domain == "attraction":
            attraction_indicators = [
                "museum", "park", "college", "church", "entertainment",
                "centre", "south", "north", "free",
                "address", "phone", "area", "type"
            ]
            
            has_attraction_content = (
                any(indicator in user_lower or indicator in response_lower 
                    for indicator in attraction_indicators) or
                any(str(entity).lower() in ["museum", "park", "college"] for entity in reference_entities)
            )
            
            return has_attraction_content and len(system_response.split()) >= 5

        return len(system_response.split()) >= 5

    def _extract_location_entity(self, reference_entities, domain):

        location_indicators = ["centre", "north", "south", "east", "west"]
        
        for entity in reference_entities:
            entity_str = str(entity).lower()
            if entity_str in location_indicators:
                return entity_str
        
        return ""
    
    def _filter_training_data(self, examples, domain):

        high_quality_samples = []
        total_samples = 0
        filtered_count = 0

        domain_dialogs = []
        for dialog_id, dialog in examples.items():
            if dialog.get("task") == domain:
                domain_dialogs.append(dialog)
        
        logger.info(f"Found {len(domain_dialogs)} dialogs for {domain} domain")

        for dialog in domain_dialogs:
            kg_triples = dialog.get("kg", [])
            utterances = dialog.get("utterances", [])

            knowledge_text = "; ".join([f"{s} {r} {o}" for s, r, o in kg_triples if s and r and o])

            conversation_history = []
            for i, turn in enumerate(utterances):
                user_query = (turn.get("input", "") or 
                            turn.get("user_utterance", "") or 
                            turn.get("user", "")).strip()
                system_response = turn.get("response", "").strip()
                reference_entities = turn.get("reference_entities", [])
                
                total_samples += 1

                if not user_query or not system_response:
                    continue
                

                generic_responses = [
                "thank you , goodbye",
                "goodbye",
                "you ' re welcome . good bye",
                "you are welcome . good bye",  
                "have a great day",
                "thank you for using our service",
                "thank you for using the cambridge",

                "is there anything else i can help",
                "anything else i can help you with",

                "you are welcome",
                "thank you",
            ]

                is_generic = False
                for generic in generic_responses:
                    if generic in system_response.lower():
                        logger.debug(f"Filtered generic response: {system_response}")
                        is_generic = True
                        break
                
                if is_generic:
                    continue
                
                if any(generic in system_response.lower() for generic in generic_responses):
                    logger.debug(f"Filtered generic response: {system_response}")
                    continue

                if len(system_response.split()) < 3:
                    continue

                if user_query.lower() in system_response.lower() and len(system_response) < len(user_query) + 20:
                    continue

                if system_response.lower().strip() in ["you're welcome", "you are welcome", "thank you", "thanks"]:
                    continue

                domain_quality_passed = self._check_domain_specific_quality(
                    user_query, system_response, domain, reference_entities
                )
                
                if not domain_quality_passed:
                    continue
                
                history_text = ""
                if conversation_history:
                    history_parts = []
                    for h in conversation_history[-2:]:  # Last 2 turns
                        history_parts.append(f"User: {h['user']} Assistant: {h['response']}")
                    history_text = " | ".join(history_parts)

                location = self._extract_location_entity(reference_entities, domain)

                prompt, full_text = self._create_consistent_training_prompt(
                    user_query=user_query,
                    system_response=system_response,
                    domain=domain,
                    knowledge_text=knowledge_text,
                    history_text=history_text,
                    location=location
                )

                high_quality_samples.append({
                    "prompt": prompt,
                    "response": system_response,
                    "full_text": full_text,
                    "domain": domain,
                    "entities": reference_entities
                })
                
                filtered_count += 1

                conversation_history.append({"user": user_query, "response": system_response})
        
        logger.info(f"Quality filtering: kept {filtered_count}/{total_samples} samples ({filtered_count/total_samples*100:.1f}%)")
        
        if filtered_count < 5:
            logger.warning(f"Very few high-quality samples for {domain} domain ({filtered_count} samples)")
        
        return high_quality_samples

    def prepare_domain_datasets(self, examples, domain):

        high_quality_samples = self._filter_training_data(examples, domain)
        
        if len(high_quality_samples) == 0:
            logger.error(f"No high-quality training samples found for {domain}")
            return None

        train_data = []
        for sample in high_quality_samples:
            train_data.append({"text": sample["full_text"]})
        
        logger.info(f"Created dataset with {len(train_data)} high-quality examples for {domain} domain")

        if train_data:
            sample_text = train_data[0]["text"]
            logger.info(f"Sample training text: {sample_text[:200]}...")
        
        return Dataset.from_list(train_data)

    def tokenize_function(self, examples):

        model_inputs = self.tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        

        input_ids = model_inputs["input_ids"]
        labels = []
        
        for i, text in enumerate(examples["text"]):

            answer_start = text.find("Answer:")
            if answer_start != -1:

                prompt_part = text[:answer_start + len("Answer:")]
                prompt_tokens = self.tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
                prompt_length = len(prompt_tokens)

                sample_labels = input_ids[i].copy()
                if prompt_length < len(sample_labels):
                    sample_labels[:prompt_length] = [-100] * prompt_length
                labels.append(sample_labels)
            else:

                sample_labels = input_ids[i].copy()
                mask_length = int(len(sample_labels) * 0.5)
                sample_labels[:mask_length] = [-100] * mask_length
                labels.append(sample_labels)
        
        model_inputs["labels"] = labels
        return model_inputs
    
    def load_base_model(self, force_reload=False):

        if self.model is None or force_reload:
            # Clean up memory
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                logger.info(f"Loading base model from {self.base_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

                self.model.config.use_cache = False
                
                logger.info(f"Base model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading base model: {e}")
                self.model = None
                return False
        return True

    def train_domain_adapter(self, domain, examples, epochs=3, batch_size=1, learning_rate=5e-5):

        logger.info(f"Preparing {domain} domain adapter training...")

        if not self.load_base_model():
            return None

        try:
            domain_dataset = self.prepare_domain_datasets(examples, domain)
            if domain_dataset is None:
                return None
                
            logger.info(f"Created dataset with {len(domain_dataset)} examples for {domain} domain")

            if len(domain_dataset) < 3:
                logger.warning(f"Too few examples for {domain} domain ({len(domain_dataset)}), skipping training")
                return None

            tokenized_dataset = domain_dataset.map(
                self.tokenize_function, 
                batched=True,
                desc=f"Tokenizing {domain} dataset"
            )
            logger.info(f"Dataset tokenized successfully: {len(tokenized_dataset)} examples")
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return None

        try:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_r,  # 16
                lora_alpha=self.lora_alpha,  # 32
                lora_dropout=self.lora_dropout,  # 0.1
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More comprehensive
                bias="none"
            )

            model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

            model.train()
        except Exception as e:
            logger.error(f"Error adding LoRA adapter: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        max_samples = 500
        if len(tokenized_dataset) > max_samples:
            logger.info(f"Reducing training samples from {len(tokenized_dataset)} to {max_samples}")
            import random
            random_indices = random.sample(range(len(tokenized_dataset)), max_samples)
            tokenized_dataset = tokenized_dataset.select(random_indices)

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{domain}_training",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size = 8
            num_train_epochs=epochs,  # 4 epochs
            learning_rate=learning_rate,  # 5e-5
            weight_decay=0.01,
            fp16=True,
            logging_dir=f"{self.output_dir}/{domain}/logs",
            logging_steps=10,
            save_strategy="epoch",
            save_steps=50,
            logging_first_step=True,
            optim="adamw_torch",
            max_grad_norm=1.0,  # Gradient clipping
            warmup_ratio=0.1,  # 10% warmup
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none",
            save_total_limit=2,
            seed=42
        )

        try:
            class MemoryMonitorCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step % 10 == 0:
                        mem_percent = psutil.virtual_memory().percent
                        logger.info(f"Step {state.global_step}: Memory usage: {mem_percent:.1f}%")
                        if mem_percent > 90:
                            logger.warning("Memory usage too high (>90%), stopping training")
                            control.should_training_stop = True
                
                def on_epoch_end(self, args, state, control, **kwargs):
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Epoch {state.epoch} completed. Cleared memory cache.")
            
            # Create data collator
            data_collator = DataCollatorForCausalLM(tokenizer=self.tokenizer)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=[MemoryMonitorCallback()]
            )
            
            # Execute training
            logger.info(f"Starting training for {domain} domain with {len(tokenized_dataset)} samples")
            trainer.train()
            
            # Save adapter
            save_path = f"{self.output_dir}/{domain}"
            model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"{domain} adapter trained and saved successfully to {save_path}")
            
            return save_path
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train_all_domain_adapters(self, examples):

        # domains = ["navigate", "weather", "schedule"]
        domains =  [
        "dontcare",      
        "italian",      
        "chinese",     
        "indian",       
        "european",      
        "asian_oriental", 
        "british",
        "turkish",      
        "spanish",       
        "vietnamese",    
        "international",
        "mediterranean", 
        "thai",          
        "french",        
        "portuguese",     
        "modern_european", 
        "gastropub",     
        "african",     
    ]
        successful_adapters = {}
        

        self._analyze_data_quality(examples)
        

        self.load_base_model(force_reload=True)
        
        for domain in domains:
            try:
                logger.info(f"Starting training for {domain} domain")
                adapter_path = self.train_domain_adapter(
                    domain=domain,
                    examples=examples,
                    epochs=3,  
                    batch_size=1,
                    learning_rate=5e-5  
                )
                
                if adapter_path:
                    successful_adapters[domain] = adapter_path
                    logger.info(f"Successfully trained {domain} adapter: {adapter_path}")
                else:
                    logger.warning(f"Failed to train {domain} adapter")
                    
            except Exception as e:
                logger.error(f"Unexpected error training {domain} adapter: {e}")
                continue
                

            torch.cuda.empty_cache()
            gc.collect()
            
        self.domain_adapters = successful_adapters
        return successful_adapters
    
    def _analyze_data_quality(self, examples):

        logger.info("Analyzing training data quality...")
        
        task_stats = {}
        total_samples = 0
        generic_responses = 0
        
        for example_id, example_data in examples.items():
            task = example_data.get("task", "unknown")
            utterances = example_data.get("utterances", [])
            
            if task not in task_stats:
                task_stats[task] = {
                    "total": 0,
                    "generic_count": 0,
                    "high_quality_count": 0,
                    "responses": []
                }
            
            for utterance in utterances:
                response = utterance.get("response", "")
                task_stats[task]["total"] += 1
                task_stats[task]["responses"].append(response)
                total_samples += 1

                generic_patterns = [
                    "need more specific information", "can you be more specific",
                    "what else would you like to know", "anything else i can help",
                    "is there anything else", "would you like me to"
                ]
                
                if any(pattern in response.lower() for pattern in generic_patterns):
                    task_stats[task]["generic_count"] += 1
                    generic_responses += 1
                else:
                    if (len(response.split()) >= 5 and 
                        response.lower().strip() not in ["you're welcome", "you are welcome", "thank you"]):
                        task_stats[task]["high_quality_count"] += 1

        logger.info(f"\n=== Training Data Quality Analysis ===")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Generic responses: {generic_responses} ({generic_responses/total_samples*100:.1f}%)")
        
        for task, stats in task_stats.items():
            logger.info(f"\nTask: {task}")
            logger.info(f"  Total samples: {stats['total']}")
            logger.info(f"  Generic responses: {stats['generic_count']} ({stats['generic_count']/stats['total']*100:.1f}%)")
            logger.info(f"  High-quality responses: {stats['high_quality_count']} ({stats['high_quality_count']/stats['total']*100:.1f}%)")

            unique_responses = list(set(stats['responses']))
            logger.info(f"  Unique responses: {len(unique_responses)}")

        logger.info(f"\n=== Recommendations ===")
        if generic_responses > total_samples * 0.3:
            logger.warning("⚠️  HIGH GENERIC RESPONSE RATE: Over 30% of responses are generic")
            logger.info("   → Filtering will remove these during training")
        
        for task, stats in task_stats.items():
            if stats["high_quality_count"] < 10:
                logger.warning(f"⚠️  LOW HIGH-QUALITY SAMPLE COUNT for {task}: Only {stats['high_quality_count']} samples")
    
    def load_domain_adapter(self, domain):

        adapter_path = f"{self.output_dir}/{domain}"
        if not os.path.exists(adapter_path):
            logger.warning(f"Warning: {domain} domain adapter not found at {adapter_path}. Training required.")
            return None
        
        try:

            logger.info(f"Loading base model for {domain} adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            logger.info(f"Loading {domain} adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info(f"{domain} adapter loaded successfully")
            
            return model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading {domain} adapter: {e}")
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
            return None
