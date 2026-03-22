"""
Schema Intelligence Module
Provides dynamic schema analysis for autonomous error correction.
Uses semantic similarity to suggest correct field names.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)


class SchemaIntelligence:
    """
    Loads and analyzes OpenAPI schema to provide intelligent field suggestions.
    Caches parsed schema for performance.
    """
    
    def __init__(self, openapi_path: str = "Docs/openapi.json"):
        self.openapi_path = openapi_path
        self._schema_cache = None
        self._field_embeddings_cache = {}
        self._load_schema()
    
    def _load_schema(self):
        """Load and cache the OpenAPI schema."""
        try:
            with open(self.openapi_path) as f:
                self._schema_cache = json.load(f)
            logger.info(f"Loaded OpenAPI schema with {len(self._schema_cache.get('paths', {}))} paths")
        except Exception as e:
            logger.error(f"Failed to load OpenAPI schema: {e}")
            self._schema_cache = {}
    
    def get_endpoint_schema(self, path: str, method: str) -> Optional[Dict]:
        """Get the full schema for a specific endpoint."""
        if not self._schema_cache:
            return None
        
        paths = self._schema_cache.get('paths', {})
        path_data = paths.get(path)
        if not path_data:
            return None
        
        method_data = path_data.get(method.lower())
        if not method_data:
            return None
        
        return method_data
    
    def get_request_body_schema(self, path: str, method: str) -> Optional[Dict]:
        """Get the request body schema for an endpoint."""
        endpoint = self.get_endpoint_schema(path, method)
        if not endpoint:
            return None
        
        request_body = endpoint.get('requestBody', {})
        content = request_body.get('content', {})
        
        # Try application/json first, then charset variant
        for content_type in ['application/json', 'application/json; charset=utf-8']:
            if content_type in content:
                schema = content[content_type].get('schema', {})
                return self._resolve_schema_ref(schema)
        
        return None
    
    def _resolve_schema_ref(self, schema: Dict) -> Dict:
        """Resolve $ref references in schema."""
        if '$ref' in schema:
            ref_path = schema['$ref']
            # Extract component name from #/components/schemas/Name
            if ref_path.startswith('#/components/schemas/'):
                schema_name = ref_path.split('/')[-1]
                schemas = self._schema_cache.get('components', {}).get('schemas', {})
                return schemas.get(schema_name, schema)
        return schema
    
    def get_valid_fields(self, path: str, method: str) -> Dict[str, Dict]:
        """
        Get all valid fields for an endpoint with their types.
        Returns: {field_name: {type, description, required, ...}}
        """
        body_schema = self.get_request_body_schema(path, method)
        if not body_schema:
            return {}
        
        properties = body_schema.get('properties', {})
        required = body_schema.get('required', [])
        
        fields = {}
        for name, prop in properties.items():
            fields[name] = {
                'name': name,
                'type': prop.get('type', 'unknown'),
                'description': prop.get('description', ''),
                'required': name in required,
                'schema': prop
            }
        
        return fields
    
    def get_nested_fields(self, path: str, method: str, field_path: str) -> Dict[str, Dict]:
        """
        Get valid fields for a nested object (e.g., orderLines items).
        field_path: dot-separated path like 'orderLines.items'
        """
        body_schema = self.get_request_body_schema(path, method)
        if not body_schema:
            return {}
        
        parts = field_path.split('.')
        current = body_schema
        
        for part in parts:
            if part == 'items':
                # Array items
                current = current.get('items', {})
            else:
                # Object property
                props = current.get('properties', {})
                current = props.get(part, {})
            
            # Resolve ref if needed
            current = self._resolve_schema_ref(current)
        
        # Get properties of the final schema
        properties = current.get('properties', {})
        fields = {}
        for name, prop in properties.items():
            fields[name] = {
                'name': name,
                'type': prop.get('type', 'unknown'),
                'description': prop.get('description', ''),
                'schema': prop
            }
        
        return fields
    
    def find_similar_fields(
        self, 
        wrong_field: str, 
        valid_fields: Dict[str, Dict],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar field names using semantic similarity.
        Returns: [(field_name, score), ...] sorted by score desc
        """
        if not valid_fields:
            return []
        
        wrong_lower = wrong_field.lower()
        scores = []
        
        for valid_name, field_info in valid_fields.items():
            valid_lower = valid_name.lower()
            
            # Multiple similarity metrics combined
            score = self._calculate_similarity(wrong_lower, valid_lower, field_info)
            scores.append((valid_name, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out low scores and return top_k (lower threshold for more suggestions)
        return [(name, score) for name, score in scores[:top_k] if score > 0.35]
    
    def _calculate_similarity(
        self, 
        wrong: str, 
        valid: str, 
        field_info: Dict
    ) -> float:
        """
        Calculate composite similarity score.
        Optimized for field name corrections.
        """
        scores = []
        wrong_lower = wrong.lower()
        valid_lower = valid.lower()
        
        # 1. EXACT MATCH - highest priority
        if wrong_lower == valid_lower:
            return 1.0
        
        # 2. PREFIX MATCHING (e.g., "unitPrice" matches "unitPriceExcludingVatCurrency")
        # This is critical for field name corrections
        if valid_lower.startswith(wrong_lower):
            # Calculate how much of the valid field is covered by the wrong field
            coverage = len(wrong_lower) / len(valid_lower)
            # High bonus for good prefix match (0.7-0.9)
            prefix_score = 0.7 + (coverage * 0.2)
            scores.append(prefix_score)
        elif wrong_lower.startswith(valid_lower):
            # Partial match (wrong is longer)
            coverage = len(valid_lower) / len(wrong_lower)
            scores.append(0.5 + (coverage * 0.2))
        
        # 3. SUBSTRING MATCHING (e.g., "price" in "priceExcludingVatCurrency")
        elif wrong_lower in valid_lower or valid_lower in wrong_lower:
            # Find position of match
            if wrong_lower in valid_lower:
                pos = valid_lower.index(wrong_lower)
                # Bonus if at start or end
                if pos == 0:
                    scores.append(0.6)  # At start
                elif pos + len(wrong_lower) == len(valid_lower):
                    scores.append(0.5)  # At end
                else:
                    scores.append(0.4)  # In middle
            else:
                scores.append(0.35)
        
        # 4. TOKEN-BASED SIMILARITY (with camelCase splitting)
        def split_camel_and_tokens(s):
            # Split camelCase: "accountNumber" -> ["account", "number"]
            camel_split = re.findall(r'[a-z]+|[A-Z][a-z]*', s)
            # Split on number boundaries: "account123" -> ["account", "123"]
            number_split = re.findall(r'[a-z]+|\d+', s)
            # Also extract any tokens
            token_split = re.findall(r'[a-z]+', s)
            # Common compound word split for field names
            compound_words = ['number', 'price', 'account', 'date', 'amount', 'vat', 'currency', 
                            'supplier', 'customer', 'invoice', 'product', 'order', 'line']
            compound_split = [w for w in compound_words if w in s]
            return set(camel_split + number_split + token_split + compound_split)
        
        wrong_tokens = split_camel_and_tokens(wrong_lower)
        valid_tokens = split_camel_and_tokens(valid_lower)
        
        if wrong_tokens and valid_tokens:
            intersection = wrong_tokens & valid_tokens
            if intersection:
                # Jaccard similarity
                union = wrong_tokens | valid_tokens
                jaccard = len(intersection) / len(union) if union else 0
                scores.append(jaccard * 0.3)
        
        # 5. SEQUENCE SIMILARITY (Levenshtein-like)
        seq_sim = SequenceMatcher(None, wrong_lower, valid_lower).ratio()
        # Only add if not already covered by prefix/substring
        if seq_sim > 0.6:
            scores.append(seq_sim * 0.2)
        
        # 6. SEMANTIC HINTS FROM DESCRIPTION
        description = field_info.get('description', '').lower()
        if description and wrong_tokens:
            desc_tokens = set(re.findall(r'[a-z]+', description))
            matching = wrong_tokens & desc_tokens
            if matching:
                desc_match = len(matching) / len(wrong_tokens)
                scores.append(desc_match * 0.1)
        
        # Return highest score (not sum, to avoid over-scoring)
        # But also consider cumulative evidence
        if len(scores) >= 2:
            # If multiple signals agree, boost the score
            top_two = sorted(scores, reverse=True)[:2]
            return min(max(scores) + (top_two[1] * 0.1), 1.0)
        
        return max(scores) if scores else 0.0
    
    def get_field_info(self, path: str, method: str, field: str) -> Optional[Dict]:
        """Get detailed info about a specific field."""
        fields = self.get_valid_fields(path, method)
        return fields.get(field)
    
    def get_required_fields(self, path: str, method: str) -> List[str]:
        """Get list of required field names."""
        body_schema = self.get_request_body_schema(path, method)
        if not body_schema:
            return []
        return body_schema.get('required', [])
    
    def get_example_payload(self, path: str, method: str) -> Optional[Dict]:
        """Get example payload if available in schema."""
        endpoint = self.get_endpoint_schema(path, method)
        if not endpoint:
            return None
        
        # Try to get from requestBody examples
        request_body = endpoint.get('requestBody', {})
        content = request_body.get('content', {})
        
        for content_type, content_data in content.items():
            examples = content_data.get('examples', {})
            if examples:
                # Return first example
                first_example = list(examples.values())[0]
                return first_example.get('value', {})
        
        return None
    
    def suggest_field_fix(
        self, 
        path: str, 
        method: str, 
        wrong_field: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Full suggestion pipeline for a wrong field.
        Returns comprehensive fix suggestion.
        
        Args:
            path: API endpoint path
            method: HTTP method
            wrong_field: The incorrect field name
            context: Optional context like 'orderLines.items' for nested fields
        """
        # Determine if we're dealing with a nested field
        parent_field = None
        nested_path = None
        
        # Check if context indicates nested field
        if context and '.items' in context:
            parent_field = context.split('.')[0]
            nested_path = context
        # Check if wrong_field itself contains path info
        elif '.' in wrong_field:
            parts = wrong_field.split('.')
            parent_field = parts[0]
            nested_path = f"{parent_field}.items"
            wrong_field = parts[-1]  # Use just the field name
        
        # If we have a nested context, try to get nested fields
        if parent_field and nested_path:
            nested = self.get_nested_fields(path, method, nested_path)
            if nested:
                similar = self.find_similar_fields(wrong_field, nested)
                return {
                    'wrong_field': wrong_field,
                    'parent_field': parent_field,
                    'is_nested': True,
                    'suggestions': similar,
                    'valid_parent': True,
                    'context': context,
                    'nested_fields_available': len(nested)
                }
        
        # Top-level field
        valid_fields = self.get_valid_fields(path, method)
        similar = self.find_similar_fields(wrong_field, valid_fields)
        
        # Get details for top suggestions
        detailed_suggestions = []
        for field_name, score in similar[:3]:
            field_info = valid_fields.get(field_name, {})
            detailed_suggestions.append({
                'field': field_name,
                'score': score,
                'type': field_info.get('type'),
                'description': field_info.get('description', '')[:100],
                'required': field_info.get('required', False)
            })
        
        return {
            'wrong_field': wrong_field,
            'is_nested': False,
            'suggestions': detailed_suggestions,
            'all_valid_fields': list(valid_fields.keys())[:20],
            'context': context
        }


# Singleton instance for reuse
_schema_intel_instance = None

def get_schema_intelligence(openapi_path: str = "Docs/openapi.json") -> SchemaIntelligence:
    """Get or create singleton SchemaIntelligence instance."""
    global _schema_intel_instance
    if _schema_intel_instance is None:
        _schema_intel_instance = SchemaIntelligence(openapi_path)
    return _schema_intel_instance
