import torch
import torch.nn as nn
import json
from transformers import AutoModel, BertModel, BertConfig
import torch.nn.functional as F


class GraphReasoningModel(nn.Module):
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = getattr(args, 'num_steps', 3)
        self.num_ways = getattr(args, 'num_ways', 2)
        num_relations = len(rel2id)
        self.num_ents = len(ent2id)
        self.ent2id = ent2id
        self.rel2id = rel2id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Tsize = len(triples)
        Esize = len(ent2id)

        if Tsize > 0:
            idx = torch.arange(Tsize, dtype=torch.long)
            indices_subj = torch.stack((idx, triples[:, 0]))
            Msubj = torch.sparse_coo_tensor(indices_subj, torch.ones(Tsize, dtype=torch.float32), (Tsize, Esize)).to(device)
            indices_rel = torch.stack((idx, triples[:, 1]))
            Mrel = torch.sparse_coo_tensor(indices_rel, torch.ones(Tsize, dtype=torch.float32), (Tsize, num_relations)).to(device)
            indices_obj = torch.stack((idx, triples[:, 2]))
            Mobj = torch.sparse_coo_tensor(indices_obj, torch.ones(Tsize, dtype=torch.float32), (Tsize, Esize)).to(device)
        else:
            Msubj = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.tensor([], dtype=torch.float32), (0, Esize)).to(device)
            Mrel = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.tensor([], dtype=torch.float32), (0, num_relations)).to(device)
            Mobj = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.tensor([], dtype=torch.float32), (0, Esize)).to(device)

        self.register_buffer("Msubj", Msubj)
        self.register_buffer("Mrel", Mrel)
        self.register_buffer("Mobj", Mobj)

        if Tsize > 0:
            self.register_buffer("triples", triples)
        else:
            self.register_buffer("triples", torch.empty((0, 3), dtype=torch.long))

        config = BertConfig.from_pretrained(args.bert_name)
        config.return_dict = True
        
        self.bert_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        for i, param in enumerate(self.bert_encoder.parameters()):
            if i < 60:
                param.requires_grad = False
                
        dim_hidden = self.bert_encoder.config.hidden_size
        
        self.step_encoders = nn.ModuleDict()
        self.rel_classifiers = nn.ModuleDict()
        self.hop_selectors = nn.ModuleDict()
        
        self.entity_match = nn.Linear(dim_hidden, dim_hidden)
        self.entity_embeddings = nn.Embedding(self.num_ents, dim_hidden)
        nn.init.normal_(self.entity_embeddings.weight, mean=0.0, std=0.02)
        
        self.relation_type_emb = nn.Embedding(num_relations, dim_hidden)
        nn.init.normal_(self.relation_type_emb.weight, mean=0.0, std=0.02)
        
        self.relation_importance = nn.Parameter(torch.ones(num_relations))
        
        for w in range(self.num_ways):
            for t in range(self.num_steps):
                name = f'way_{w}_step_{t}'
                self.step_encoders[name] = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.Tanh()
                )
            
            self.rel_classifiers[f'way_{w}'] = nn.Linear(dim_hidden, num_relations)
            self.hop_selectors[f'way_{w}'] = nn.Linear(dim_hidden, self.num_steps)
        
        self.path_validator = nn.Sequential(
            nn.Linear(dim_hidden*2, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
        
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def follow(self, e, r):
        device = self.Msubj.device
        
        e = e.to(device).float()
        r = r.to(device).float()
        
        weighted_r = r * self.relation_importance.unsqueeze(0)
        
        x = torch.sparse.mm(self.Msubj, e.t()) * torch.sparse.mm(self.Mrel, weighted_r.t())
        result = torch.sparse.mm(self.Mobj.t(), x).t()  # [bsz, Esize]
        
        result = torch.clamp(result, min=0.0, max=1.0)
        
        max_vals = torch.max(result, dim=1, keepdim=True)[0]
        max_vals = torch.where(max_vals > 0, max_vals, torch.ones_like(max_vals))
        result = result / max_vals
        
        return result

    def follow_reverse(self, e, r):
        device = self.Mobj.device

        e = e.to(device).float()
        r = r.to(device).float()

        weighted_r = r * self.relation_importance.unsqueeze(0)

        x = torch.sparse.mm(self.Mobj, e.t()) * torch.sparse.mm(self.Mrel, weighted_r.t())
        result = torch.sparse.mm(self.Msubj.t(), x).t()  # [bsz, Esize]

        result = torch.clamp(result, min=0.0, max=1.0)

        max_vals = torch.max(result, dim=1, keepdim=True)[0]
        max_vals = torch.where(max_vals > 0, max_vals, torch.ones_like(max_vals))
        result = result / max_vals

        return result

    def forward(self, heads, questions, answers=None, entity_range=None, question_text=None, history_text=None):
        device = next(self.parameters()).device
        heads = heads.to(device).float()
        if answers is not None:
            answers = answers.to(device).float()
        if entity_range is not None:
            entity_range = entity_range.to(device).float()
        questions = {k: v.to(device) for k, v in questions.items()}

        q = self.bert_encoder(**questions)

        if q.pooler_output is None:
            q_embeddings = q.last_hidden_state[:, 0, :]
        else:
            q_embeddings = q.pooler_output

        q_embeddings = self.dropout(q_embeddings)
        q_word_h = q.last_hidden_state
        
        batch_size = heads.size(0)
        
        question_entity_match = self.entity_match(q_embeddings)  # [bsz, dim_h]
        all_entity_embeds = self.entity_embeddings.weight  # [num_ents, dim_h]
        
        direct_entity_scores = torch.matmul(question_entity_match, all_entity_embeds.t())  # [bsz, num_ents]
        direct_entity_scores = torch.sigmoid(direct_entity_scores / torch.sqrt(torch.tensor(q_embeddings.size(-1), dtype=torch.float)))
        
        direct_match_weight = 0.3
        enhanced_heads = heads * (1.0 + direct_entity_scores * direct_match_weight)

        head_sum = torch.sum(enhanced_heads, dim=1, keepdim=True)
        head_sum = torch.where(head_sum > 0, head_sum, torch.ones_like(head_sum))
        enhanced_heads = enhanced_heads / head_sum

        all_e_scores = []
        all_word_attns = []
        all_rel_probs = []
        all_ent_probs = []
        all_hop_attns = []

        question_is_address_query = False
        question_is_poi_query = False
        
        if question_text is not None:
            question_text = str(question_text).lower()
            if "address" in question_text or "where" in question_text or "located" in question_text:
                question_is_address_query = True
            if "restaurant" in question_text or "gas" in question_text or "station" in question_text:
                question_is_poi_query = True
        
        for w in range(self.num_ways):
            if w == 0 or self.num_ways > 1:
                last_e = enhanced_heads.float()
            else:
                if question_is_address_query:
                    last_e = heads.float()
            
            word_attns = []
            rel_probs = []
            ent_probs = []
            
            for t in range(self.num_steps):
                cq_t = self.step_encoders[f'way_{w}_step_{t}'](q_embeddings)  # [bsz, dim_h]
                cq_t = cq_t.float()

                q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) / self.temperature
                q_dist = torch.softmax(q_logits, 1)  # [bsz, max_q]
                q_dist = q_dist * questions['attention_mask'].float()
                q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6)  # [bsz, max_q]
                word_attns.append(q_dist)

                ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1)  # [bsz, dim_h]
                ctx_h = ctx_h.float()

                rel_logit = self.rel_classifiers[f'way_{w}'](ctx_h)  # [bsz, num_relations]

                if question_is_address_query:
                    for rel_name, rel_id in self.rel2id.items():
                        if "address" in str(rel_name).lower():
                            rel_logit[:, rel_id] += 2.0
                
                if question_is_poi_query and t == 0:
                    for rel_name, rel_id in self.rel2id.items():
                        if "poi_type" in str(rel_name).lower():
                            rel_logit[:, rel_id] += 2.0

                rel_dist = torch.softmax(rel_logit, dim=-1)
                rel_probs.append(rel_dist)

                if w == 0 or not question_is_address_query:
                    next_e = self.follow(last_e, rel_dist)
                else:
                    if t == 0 and question_is_address_query:
                        next_e = self.follow_reverse(last_e, rel_dist)
                    else:
                        next_e = self.follow(last_e, rel_dist)

                if t > 0:
                    res_weight = 0.3
                    next_e = (1.0 - res_weight) * next_e + res_weight * last_e

                next_e = next_e.to(device)
                if entity_range is not None:
                    entity_range_t = entity_range.to(device).float()
                    masked_e = next_e * entity_range_t
                else:
                    masked_e = next_e

                e_sum = torch.sum(masked_e, dim=1, keepdim=True)
                e_sum = torch.where(e_sum > 0, e_sum, torch.ones_like(e_sum))
                last_e = masked_e / e_sum
                
                ent_probs.append(last_e)

            ent_probs = [ep.float() for ep in ent_probs]
            hop_res = torch.stack(ent_probs, dim=1)  # [bsz, num_steps, num_ent]
            hop_logit = self.hop_selectors[f'way_{w}'](q_embeddings)

            if question_is_address_query and w == 0:
                hop_logit[:, -1] += 1.0
            elif question_is_poi_query and w == 0:
                if self.num_steps >= 2:
                    hop_logit[:, 1] += 1.0

            hop_attn = torch.softmax(hop_logit, dim=1).unsqueeze(2)

            way_e_score = torch.sum(hop_res * hop_attn, dim=1)  # [bsz, num_ent]

            if w == 0:
                final_direct_weight = 0.15
                way_e_score = way_e_score * (1.0 + direct_entity_scores * final_direct_weight)

                e_sum = torch.sum(way_e_score, dim=1, keepdim=True)
                e_sum = torch.where(e_sum > 0, e_sum, torch.ones_like(e_sum))
                way_e_score = way_e_score / e_sum

            all_e_scores.append(way_e_score)
            all_word_attns.append(word_attns)
            all_rel_probs.append(rel_probs)
            all_ent_probs.append(ent_probs)
            all_hop_attns.append(hop_attn.squeeze(2))

        if self.num_ways > 1:
            if question_is_address_query:
                path_weights = torch.tensor([0.7, 0.3], device=device)
                e_score = torch.zeros_like(all_e_scores[0])
                for i, score in enumerate(all_e_scores):
                    e_score += score * path_weights[i]
            elif question_is_poi_query:
                path_weights = torch.tensor([0.6, 0.4], device=device)
                e_score = torch.zeros_like(all_e_scores[0])
                for i, score in enumerate(all_e_scores):
                    e_score += score * path_weights[i]
            else:
                e_score = sum(all_e_scores) / self.num_ways
        else:
            e_score = all_e_scores[0]

        e_sum = torch.sum(e_score, dim=1, keepdim=True)
        e_sum = torch.where(e_sum > 0, e_sum, torch.ones_like(e_sum))
        e_score = e_score / e_sum

        if not self.training:
            output_dict = {
                'e_score': e_score,
                'word_attns': all_word_attns[0],
                'rel_probs': all_rel_probs[0], 
                'ent_probs': all_ent_probs[0],
                'hop_attn': all_hop_attns[0],
                'direct_entity_scores': direct_entity_scores
            }
            return output_dict
        else:
            gamma = 2.0
            alpha = 0.25

            weight = answers * 19 + 1

            bce_loss = F.binary_cross_entropy(e_score, answers, reduction='none')

            pt = torch.exp(-bce_loss)
            focal_loss = alpha * (1-pt)**gamma * bce_loss

            entity_loss = torch.sum(entity_range * weight * focal_loss) / (torch.sum(entity_range * weight) + 1e-8)

            consistency_loss = 0.0
            if self.num_ways > 1:
                for i in range(self.num_ways-1):
                    for j in range(i+1, self.num_ways):
                        path_similarity = torch.sum(all_e_scores[i] * all_e_scores[j], dim=1)
                        consistency_loss += torch.mean(path_similarity)

                consistency_loss = consistency_loss / (self.num_ways * (self.num_ways - 1) / 2)
                consistency_loss = 0.05 * consistency_loss

            sparsity_loss = 0.02 * torch.mean(torch.sum(e_score**2, dim=1))
            
            total_loss = entity_loss + consistency_loss + sparsity_loss
            
            return {
                'loss': total_loss, 
                'entity_loss': entity_loss, 
                'consistency_loss': consistency_loss,
                'sparsity_loss': sparsity_loss
            }
