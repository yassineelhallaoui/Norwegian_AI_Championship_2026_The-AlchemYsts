import json
import os
import re
from typing import Any, Dict, List, Set
from difflib import SequenceMatcher

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class KnowledgeGraph:
    """
    Experience and rule store for Code_YassY_v3.
    Replaces deterministic error counting with semantic rule storage in plain English.
    Includes semantic matching for relevant rule retrieval.
    """
    
    def __init__(self, storage_path: str = "runtime_knowledge_graph.json", api_key: str = None):
        self.storage_path = os.path.join(os.path.dirname(__file__), "..", storage_path)
        self.rules: List[Dict[str, Any]] = []  # Now stores dicts with metadata
        self.entity_relations: List[Dict[str, Any]] = []
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if genai and self.api_key else None
        self._load()
        
    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle old format (simple strings) vs new format (dicts)
                    loaded_rules = data.get("rules", [])
                    self.rules = []
                    for rule in loaded_rules:
                        if isinstance(rule, str):
                            # Convert old format to new
                            self.rules.append({
                                "text": rule,
                                "keywords": self._extract_keywords(rule),
                                "created_at": "",
                                "success_count": 1
                            })
                        else:
                            self.rules.append(rule)
                    self.entity_relations = data.get("entity_relations", [])
            except Exception as e:
                print(f"Error loading knowledge graph: {e}")
                self.rules = []
                self.entity_relations = []

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rules": self.rules,
                    "entity_relations": self.entity_relations
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from a rule for matching."""
        # Convert to lowercase and extract important words
        text = text.lower()
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = re.findall(r'\b[a-z]+\b', text)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _keyword_overlap_score(self, task_keywords: Set[str], rule_keywords: List[str]) -> float:
        """Calculate overlap between task keywords and rule keywords."""
        if not rule_keywords:
            return 0.0
        rule_keywords_set = set(rule_keywords)
        intersection = task_keywords & rule_keywords_set
        if not intersection:
            return 0.0
        # Jaccard similarity
        return len(intersection) / len(task_keywords | rule_keywords_set)

    def add_rule(self, rule_description: str, context: Dict[str, Any] = None):
        """Store a new rule derived from LLM self-correction."""
        if not rule_description or len(rule_description) < 5:
            return
        
        # Check for duplicates
        for existing in self.rules:
            if existing.get("text", "") == rule_description:
                # Increment success count
                existing["success_count"] = existing.get("success_count", 1) + 1
                self._save()
                return
        
        # Check for very similar rules
        for existing in self.rules:
            if self._calculate_similarity(existing.get("text", ""), rule_description) > 0.85:
                # Update if new rule is more detailed
                if len(rule_description) > len(existing.get("text", "")):
                    existing["text"] = rule_description
                    existing["keywords"] = self._extract_keywords(rule_description)
                    self._save()
                return
        
        new_rule = {
            "text": rule_description,
            "keywords": self._extract_keywords(rule_description),
            "created_at": context.get("timestamp", "") if context else "",
            "endpoint": context.get("endpoint", "") if context else "",
            "error_type": context.get("error_type", "") if context else "",
            "success_count": 1
        }
        
        self.rules.append(new_rule)
        self._save()
    
    def get_applicable_rules(self, task_description: str, top_k: int = 8) -> List[str]:
        """Retrieve historical rules relevant to the current task using semantic matching."""
        if not self.rules:
            return []
        
        task_lower = task_description.lower()
        task_keywords = set(self._extract_keywords(task_description))
        
        # Score each rule
        scored_rules = []
        for rule in self.rules:
            rule_text = rule.get("text", "")
            rule_keywords = rule.get("keywords", [])
            
            # Multiple scoring factors
            scores = []
            
            # 1. Keyword overlap
            keyword_score = self._keyword_overlap_score(task_keywords, rule_keywords)
            scores.append(("keyword", keyword_score, 2.0))
            
            # 2. Direct text similarity
            similarity = self._calculate_similarity(task_lower, rule_text.lower())
            scores.append(("similarity", similarity, 1.5))
            
            # 3. Entity mention (if task mentions entities from rule)
            entity_score = 0.0
            for keyword in rule_keywords:
                if keyword in task_lower:
                    entity_score += 0.2
            scores.append(("entity", min(entity_score, 1.0), 1.0))
            
            # 4. Success count boost (rules that worked before)
            success_boost = min(rule.get("success_count", 1) * 0.05, 0.3)
            scores.append(("success", success_boost, 1.0))
            
            # Calculate weighted total
            total_score = sum(s[1] * s[2] for s in scores)
            scored_rules.append((total_score, rule))
        
        # Sort by score descending
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k rules (or all if less than top_k)
        selected = scored_rules[:top_k]
        
        # If we have LLM client, do a secondary LLM-based ranking for top candidates
        if self.client and len(selected) > 3:
            return self._llm_rank_rules(task_description, selected[:10])
        
        return [r[1].get("text", "") for r in selected]

    def _llm_rank_rules(self, task_description: str, candidate_rules: List[tuple]) -> List[str]:
        """Use LLM to rank and filter the most relevant rules."""
        try:
            rules_text = "\n".join([f"{i}. {r[1].get('text', '')}" for i, r in enumerate(candidate_rules)])
            
            prompt = f'''Given this task: "{task_description}"

Here are historical rules from the knowledge graph:
{rules_text}

Select the 3-5 most relevant rules that would help complete this task successfully.
Return ONLY a JSON array of indices, e.g., [0, 2, 4]
'''
            response = self.client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25"),
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            selected_indices = json.loads(response.text)
            if isinstance(selected_indices, list):
                result = []
                for idx in selected_indices[:5]:
                    if isinstance(idx, int) and 0 <= idx < len(candidate_rules):
                        result.append(candidate_rules[idx][1].get("text", ""))
                return result if result else [r[1].get("text", "") for r in candidate_rules[:5]]
        except Exception:
            pass
        
        # Fallback to top 5
        return [r[1].get("text", "") for r in candidate_rules[:5]]

    def record_entity_relation(self, entity_a: str, entity_b: str, relation: str):
        """Record a semantic link, e.g., 'Customer 123' HAS 'Project 456'."""
        rel = {
            "source": entity_a,
            "target": entity_b,
            "relation": relation
        }
        if rel not in self.entity_relations:
            self.entity_relations.append(rel)
            self._save()
    
    def find_related_entities(self, entity: str) -> List[Dict[str, Any]]:
        """Find all entities related to the given entity."""
        related = []
        for rel in self.entity_relations:
            if rel["source"] == entity or rel["target"] == entity:
                related.append(rel)
        return related
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_rules": len(self.rules),
            "total_relations": len(self.entity_relations),
            "most_successful_rules": sorted(
                self.rules, 
                key=lambda r: r.get("success_count", 0), 
                reverse=True
            )[:5]
        }
