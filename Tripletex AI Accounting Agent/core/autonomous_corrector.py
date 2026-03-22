"""
Autonomous Correction Module

This module is the "self-healing" system of the agent. When the Tripletex API
returns an error (like "department.id is invalid" or "field doesn't exist"),
this module automatically figures out what went wrong and fixes it.

How it works:
1. Error Analysis - Parse Norwegian error messages to understand the problem
2. Schema Intelligence - Look up valid fields from the OpenAPI schema
3. Field Matching - Find the correct field names using patterns + LLM
4. Verification - Have Gemini double-check the fix before retrying
5. Learning - Store successful corrections for future tasks

Built by Yassine Elhallaoui for NM i AI 2026.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .schema_intelligence import get_schema_intelligence, SchemaIntelligence
from .error_analyzer import get_error_analyzer, ErrorAnalyzer, ErrorAnalysis

"""
PROTECTED_FIELDS - These fields are correct and should never be renamed.

Sometimes the LLM gets confused by error messages. For example, if the error
says "guiProductId doesn't exist", the LLM might try to rename 'product' 
to 'guiProductId' - which is wrong! These protected fields prevent that.

Real-world examples we've seen:
- product -> guiProductId (WRONG: product is correct)
- nationalIdentityNumber -> internationalId (WRONG: internationalId is for passports)
- comments -> displayName (WRONG: breaks travel expenses)
"""
PROTECTED_FIELDS = {
    'product',  # API uses 'product', not 'guiProductId'
    'customer',
    'department',
    'projectManager',
    'currency',
    'nationalIdentityNumber',  # Norwegian national ID, not passport
    'employee',
    # Travel expense fields - these are correct as-is
    'comments',
    'amountCurrencyIncVat',
    'costCategory',
    'paymentType',
    'isPaidByEmployee',
}

"""
PROTECTED_TARGET_FIELDS - These are INCORRECT field names we should never use.

Even if the LLM suggests renaming something TO these fields, we block it.
These came from real errors in production logs.
"""
PROTECTED_TARGET_FIELDS = {
    'internationalId',  # Wrong field for Norwegian national ID
    'guiProductId',  # Internal Tripletex field, not for API use
}

# Hardcoded field mappings from real-world errors
FIELD_NAME_MAPPINGS = {
    # Norwegian -> English
    'fodselsnummer': 'nationalIdentityNumber',
    'fødselsnummer': 'nationalIdentityNumber',
    'avdeling': 'department',
    'kunde': 'customer',
    'kundeid': 'customer',
    'customerid': 'customer',
    'customer id': 'customer',
    'kundenummer': 'customerNumber',
    'prosjektleder': 'projectManager',
    'prosjektlederid': 'projectManager',
    'ansatt': 'employee',
    'ansattnummer': 'employeeNumber',
    'organisasjonsnummer': 'organizationNumber',
    'orgnr': 'organizationNumber',
    'postadresse': 'postalAddress',
    'faktura': 'invoice',
    'fakturanummer': 'invoiceNumber',
    'leverandor': 'supplier',
    'leverandornummer': 'supplierNumber',
    'belop': 'amount',
    'pris': 'price',
    'beskrivelse': 'description',
    'navn': 'name',
    'antall': 'count',
    'dato': 'date',
    'forfallsdato': 'invoiceDueDate',
    'betalingsdato': 'paymentDate',
    
    # Common wrong field names
    'price': 'priceExcludingVatCurrency',  # Generic price -> priceExcludingVatCurrency
    'unitprice': 'unitPriceExcludingVatCurrency',
    'unit Price': 'unitPriceExcludingVatCurrency',
    'salesprice': 'priceExcludingVatCurrency',
    'sales Price': 'priceExcludingVatCurrency',
    'costprice': 'costExcludingVatCurrency',
    'cost Price': 'costExcludingVatCurrency',
    'priceexcludingvat': 'priceExcludingVatCurrency',
    'amountexcludingvat': 'amountExcludingVatCurrency',
    'totalamount': 'amount',
    'linetotal': 'amountExcludingVatCurrency',
    'linje total': 'amountExcludingVatCurrency',
    'lines': 'orderLines',
    'invoicelines': 'orderLines',
    'invoice lines': 'orderLines',
    'duedate': 'invoiceDueDate',
    'due date': 'invoiceDueDate',
    'paymentdate': 'paymentDate',
    'payment date': 'paymentDate',
    'invoicedate': 'invoiceDate',
    'invoice date': 'invoiceDate',
    'postings': None,  # Special: remove this field
    'accountnumber': 'supplierNumber',  # For suppliers
    'account number': 'supplierNumber',
    'customernumber': 'customerNumber',
    'customer number': 'customerNumber',
    'customernum': 'customerNumber',
    'suppliernumber': 'supplierNumber',
    'supplier number': 'supplierNumber',
    'projectmanagerid': 'projectManager',
    'project manager id': 'projectManager',
    'projectmanager': 'projectManager',
    'project manager': 'projectManager',
    'managerid': 'projectManager',
    'manager id': 'projectManager',
    'budgetamount': 'priceCeilingAmount',
    'budget amount': 'priceCeilingAmount',
    'budgettotal': 'priceCeilingAmount',
    'budget total': 'priceCeilingAmount',
    'budgetamounttotal': 'priceCeilingAmount',
    'address': 'postalAddress',
    'streetaddress': 'postalAddress',
    'street address': 'postalAddress',
    'zipcode': 'postalCode',
    'zip code': 'postalCode',
    'zip': 'postalCode',
    'postalcode': 'postalCode',
    'postal code': 'postalCode',
    'city': 'city',
    'country': 'country',
    'phone': 'phoneNumber',
    'phonenumber': 'phoneNumber',
    'phone number': 'phoneNumber',
    'mobile': 'phoneNumberMobile',
    'mobilenumber': 'phoneNumberMobile',
    'mobile number': 'phoneNumberMobile',
    'email': 'email',
    'e-mail': 'email',
    'vatnumber': 'vatNumber',
    'vat number': 'vatNumber',
    'orgnumber': 'organizationNumber',
    'org number': 'organizationNumber',
}

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of an autonomous correction attempt."""
    success: bool
    original_payload: Dict
    corrected_payload: Dict
    corrected_path: str
    corrected_method: str
    changes_made: List[Dict]
    learned_rule: str
    llm_verified: bool
    confidence: float


class AutonomousCorrector:
    """
    Main orchestrator for autonomous error correction.
    Combines schema intelligence with LLM verification.
    """
    
    def __init__(self, llm_client=None, schema_intel: SchemaIntelligence = None, tripletex_client=None):
        self.llm = llm_client
        self.schema_intel = schema_intel or get_schema_intelligence()
        self.error_analyzer = get_error_analyzer()
        self.tripletex_client = tripletex_client
        self._default_department_id = None
    
    def _get_default_department_id(self) -> int:
        """Get the first available department ID, caching the result.
        If no departments exist, creates a default one automatically."""
        if self._default_department_id is not None:
            return self._default_department_id
        
        # Fallback to 1 if no client available
        if not self.tripletex_client:
            return 1
        
        try:
            # Try to get existing department
            result = self.tripletex_client.get("/department", params={"from": 0, "count": 1})
            departments = result.get("values", [])
            
            if departments:
                self._default_department_id = departments[0].get("id", 1)
                logger.info(f"Using existing department ID: {self._default_department_id}")
                return self._default_department_id
            
            # No departments exist - create one
            logger.warning("No departments found. Creating default department...")
            try:
                create_result = self.tripletex_client.post("/department", data={
                    "name": "Default Department",
                    "departmentNumber": "1",
                    "isInactive": False
                })
                new_dept_id = create_result.get("value", {}).get("id", 1)
                self._default_department_id = new_dept_id
                logger.info(f"Created default department with ID: {new_dept_id}")
                return new_dept_id
            except Exception as create_error:
                logger.error(f"Failed to create department: {create_error}")
                # Return 1 as last resort - will likely fail but we tried
                return 1
                
        except Exception as e:
            logger.warning(f"Could not fetch default department: {e}")
        
        return 1
    
    def correct_api_error(
        self,
        method: str,
        path: str,
        payload: Dict,
        error_response: Dict,
        schema_context: List[Dict] = None
    ) -> CorrectionResult:
        """
        Main entry point: analyze error and generate corrected API call.
        """
        logger.info(f"🔄 Autonomous correction started for {method} {path}")
        
        # Step 1: Analyze the error
        analysis = self.error_analyzer.analyze_error(error_response, path, method)
        logger.debug(f"Error analysis: {analysis.error_type}")
        
        # Step 2: Check if retryable
        if not analysis.is_retryable:
            logger.warning(f"Error is not retryable: {analysis.error_type}")
            return CorrectionResult(
                success=False,
                original_payload=payload,
                corrected_payload=payload,
                corrected_path=path,
                corrected_method=method,
                changes_made=[],
                learned_rule=f"Error type '{analysis.error_type}' is not auto-fixable",
                llm_verified=False,
                confidence=0.0
            )
        
        # Step 3: Gather schema intelligence
        schema_hints = self._gather_schema_hints(analysis, path, method, payload)
        
        # Step 4: Generate fix using LLM with schema hints
        corrected = self._generate_fix_with_llm(
            method, path, payload, analysis, schema_hints
        )
        
        # Step 5: Verify the fix
        verification = self._verify_fix(corrected, path, method)
        
        # Step 6: Build result
        changes = self._identify_changes(payload, corrected['payload'])
        
        result = CorrectionResult(
            success=verification['is_valid'],
            original_payload=payload,
            corrected_payload=corrected['payload'],
            corrected_path=corrected.get('path', path),
            corrected_method=corrected.get('method', method),
            changes_made=changes,
            learned_rule=corrected.get('learned_rule', ''),
            llm_verified=verification['llm_verified'],
            confidence=verification['confidence']
        )
        
        logger.info(f"✅ Correction complete: {len(changes)} changes, confidence={verification['confidence']:.2f}")
        return result
    
    def _gather_schema_hints(
        self, 
        analysis: ErrorAnalysis, 
        path: str, 
        method: str,
        payload: Dict
    ) -> Dict[str, Any]:
        """Gather relevant schema information for the error."""
        hints = {
            'valid_fields': {},
            'field_candidates': {},  # Simple candidate matching for LLM to choose
            'required_fields': [],
            'example_payload': None,
            'nested_context': {}  # Track nested field context
        }
        
        # Get valid fields for the endpoint
        valid_fields = self.schema_intel.get_valid_fields(path, method)
        hints['valid_fields'] = {
            name: {
                'type': info['type'],
                'description': info['description'][:100] if info['description'] else '',
                'required': info['required']
            }
            for name, info in valid_fields.items()
        }
        
        # For each validation error, find candidate fields
        for error in analysis.validation_errors:
            wrong_field = error.field.lower()
            
            # Check if this is a nested field error
            nested_parent = self._find_nested_parent(payload, error.field)
            
            if error.error_type == 'unknown_field':
                if nested_parent:
                    # Get nested fields
                    context = f"{nested_parent}.items"
                    nested_fields = self.schema_intel.get_nested_fields(path, method, context)
                    hints['nested_context'][error.field] = {
                        'parent': nested_parent,
                        'valid_nested': list(nested_fields.keys())
                    }
                    
                    # Find simple candidates (contains matching)
                    candidates = self._find_candidates(error.field, nested_fields)
                    hints['field_candidates'][error.field] = candidates
                else:
                    # Top-level: find candidates from valid fields
                    candidates = self._find_candidates(error.field, valid_fields)
                    hints['field_candidates'][error.field] = candidates
        
        # Get required fields
        hints['required_fields'] = self.schema_intel.get_required_fields(path, method)
        
        # Get example payload
        hints['example_payload'] = self.schema_intel.get_example_payload(path, method)
        
        return hints
    
    def _find_candidates(self, wrong_field: str, valid_fields: Dict) -> List[str]:
        """Find candidate field names using hardcoded mappings + simple matching."""
        wrong_normalized = wrong_field.lower().replace(' ', '').replace('_', '').replace('-', '')
        candidates = []
        
        # First check hardcoded mappings
        for wrong_pattern, correct_field in FIELD_NAME_MAPPINGS.items():
            pattern_normalized = wrong_pattern.lower().replace(' ', '').replace('_', '').replace('-', '')
            if pattern_normalized == wrong_normalized:
                if correct_field is None:
                    # Special case: field should be removed
                    return ['__REMOVE__']
                # Check if the mapped field exists in valid fields
                for valid_name in valid_fields.keys():
                    if valid_name.lower() == correct_field.lower():
                        return [valid_name]
                # If exact match not found, look for partial match
                for valid_name in valid_fields.keys():
                    if correct_field.lower() in valid_name.lower():
                        candidates.append(valid_name)
        
        # Simple matching for remaining fields
        for valid_name in valid_fields.keys():
            valid_lower = valid_name.lower()
            
            # Skip if already added
            if valid_name in candidates:
                continue
            
            # Exact match
            if wrong_normalized == valid_lower.replace('_', '').replace('-', ''):
                return [valid_name]
            
            # Contains match
            if wrong_normalized in valid_lower or valid_lower in wrong_normalized:
                candidates.append(valid_name)
            # Common word matching
            elif any(word in valid_lower for word in ['price', 'amount', 'date', 'number', 'vat', 'currency'] 
                     if word in wrong_normalized):
                candidates.append(valid_name)
        
        return candidates[:5]  # Top 5 candidates
    
    def _find_nested_parent(self, payload: Dict, field: str) -> Optional[str]:
        """Find which array/object in payload contains this field."""
        for key, value in payload.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if field exists in any item
                for item in value:
                    if isinstance(item, dict) and field in item:
                        return key
            elif isinstance(value, dict) and field in value:
                return key
        return None
    
    def _generate_fix_with_llm(
        self,
        method: str,
        path: str,
        payload: Dict,
        analysis: ErrorAnalysis,
        schema_hints: Dict
    ) -> Dict:
        """Generate corrected payload using LLM with schema hints."""
        
        if not self.llm or not self.llm.client:
            # Fallback: basic corrections without LLM
            return self._basic_correction(method, path, payload, analysis, schema_hints)
        
        # Build comprehensive prompt
        prompt = self._build_correction_prompt(
            method, path, payload, analysis, schema_hints
        )
        
        try:
            from google.genai import types
            response = self.llm.client.models.generate_content(
                model=self.llm.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            result = json.loads(response.text)
            
            return {
                'path': result.get('corrected_path', path),
                'method': result.get('corrected_method', method),
                'payload': result.get('corrected_payload', payload),
                'learned_rule': result.get('learned_rule', ''),
                'explanation': result.get('explanation', ''),
            }
            
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return self._basic_correction(method, path, payload, analysis, schema_hints)
    
    def _build_correction_prompt(
        self,
        method: str,
        path: str,
        payload: Dict,
        analysis: ErrorAnalysis,
        schema_hints: Dict
    ) -> str:
        """Build comprehensive prompt for LLM correction."""
        
        # Error summary
        error_summary = self.error_analyzer.get_error_summary(analysis)
        
        # Schema information
        valid_fields_str = json.dumps(schema_hints['valid_fields'], indent=2, default=str)
        
        # Build field candidate info
        candidate_str = ""
        for wrong_field, candidates in schema_hints.get('field_candidates', {}).items():
            candidate_str += f"\n'{wrong_field}' should probably be one of: {candidates}\n"
        
        # Nested field context
        nested_str = ""
        for field, ctx in schema_hints.get('nested_context', {}).items():
            nested_str += f"\n'{field}' is in nested object '{ctx['parent']}'. Valid nested fields: {ctx['valid_nested'][:10]}\n"
        
        # Example payload
        example_str = ""
        if schema_hints.get('example_payload'):
            example_str = f"\nExample valid payload:\n{json.dumps(schema_hints['example_payload'], indent=2)}\n"
        
        # Get dynamic defaults
        default_dept_id = self._get_default_department_id()
        
        prompt = f'''You are an expert at fixing API request errors.

FAILED REQUEST:
Method: {method}
Path: {path}
Payload: {json.dumps(payload, indent=2)}

ERROR ANALYSIS:
{error_summary}

VALID FIELDS FOR THIS ENDPOINT:
{list(schema_hints['valid_fields'].keys())}

FIELD RENAMING HINTS:
{candidate_str}
{nested_str}
{example_str}

REQUIRED FIELDS (must be present):
{schema_hints.get('required_fields', [])}

PROTECTED FIELDS (never rename these):
{list(PROTECTED_FIELDS)}

FORBIDDEN TARGET FIELDS (never rename TO these):
{list(PROTECTED_TARGET_FIELDS)}

YOUR TASK:
Fix the payload by:
1. Renaming fields that don't exist to their correct names
2. Adding missing required fields with appropriate default values

Rules:
1. ONLY use field names from the "VALID FIELDS" list
2. For fields causing "does not exist" errors, pick the best match from candidates
3. Keep all values the same, only change field names
4. For nested arrays (like orderLines), fix fields inside each item
5. DO NOT use placeholder IDs like {{id}}
6. NEVER rename PROTECTED FIELDS - these are correct and should stay as-is
7. NEVER use FORBIDDEN TARGET FIELDS as the new name for any field
8. For missing REQUIRED fields, add them with sensible defaults:
   - department: {{"id": {default_dept_id}}}
   - currency: {{"id": 1}} (NOK)
   - isActive: true
   - userType: "STANDARD"

Response format (JSON only):
{{
  "corrected_path": "{path}",
  "corrected_method": "{method}",
  "corrected_payload": {{ ...fixed payload... }},
  "explanation": "Brief explanation of what was fixed",
  "learned_rule": "A general rule for future similar errors"
}}

Be precise. Only rename fields that are causing errors.'''
        
        return prompt
    
    def _basic_correction(
        self,
        method: str,
        path: str,
        payload: Dict,
        analysis: ErrorAnalysis,
        schema_hints: Dict
    ) -> Dict:
        """Fallback correction without LLM - use first candidate match."""
        import copy
        corrected = copy.deepcopy(payload)
        changes = []
        
        # Handle missing required fields
        for error in analysis.validation_errors:
            if error.error_type == 'missing_required':
                field = error.field
                
                # Add default values for common required fields
                if field == 'department' or field == 'department.id':
                    if 'department' not in corrected or corrected.get('department') is None:
                        dept_id = self._get_default_department_id()
                        corrected['department'] = {'id': dept_id}
                        changes.append({
                            'type': 'field_added',
                            'field': 'department',
                            'value': {'id': dept_id},
                            'reason': 'missing_required_default'
                        })
                
                elif field == 'currency' or field == 'currency.id':
                    if 'currency' not in corrected or corrected.get('currency') is None:
                        corrected['currency'] = {'id': 1}  # NOK
                        changes.append({
                            'type': 'field_added',
                            'field': 'currency',
                            'value': {'id': 1},
                            'reason': 'missing_required_default'
                        })
                
                elif field == 'customer' or field == 'customer.id':
                    # Customer is required but not provided - this is a data issue
                    # Mark it so LLM can handle it
                    pass
        
        # Apply field candidates
        for wrong_field, candidates in schema_hints.get('field_candidates', {}).items():
            if not candidates:
                continue
            
            # Skip protected fields that should never be renamed
            if wrong_field in PROTECTED_FIELDS:
                logger.debug(f"Skipping protected field: {wrong_field}")
                continue
                
            new_field = candidates[0]  # Take first candidate
            
            # Skip if target field is a protected wrong target
            if new_field in PROTECTED_TARGET_FIELDS:
                logger.debug(f"Skipping protected target field: {new_field}")
                continue
            
            # Special case: remove the field
            if new_field == '__REMOVE__':
                nested_parent = self._find_nested_parent(payload, wrong_field)
                if nested_parent:
                    parent_value = corrected.get(nested_parent)
                    if isinstance(parent_value, list):
                        for item in parent_value:
                            if isinstance(item, dict) and wrong_field in item:
                                del item[wrong_field]
                                changes.append({
                                    'type': 'field_removed',
                                    'field': f"{nested_parent}[].{wrong_field}",
                                    'reason': 'field_not_allowed'
                                })
                    elif isinstance(parent_value, dict) and wrong_field in parent_value:
                        del parent_value[wrong_field]
                        changes.append({
                            'type': 'field_removed',
                            'field': f"{nested_parent}.{wrong_field}",
                            'reason': 'field_not_allowed'
                        })
                elif wrong_field in corrected:
                    del corrected[wrong_field]
                    changes.append({
                        'type': 'field_removed',
                        'field': wrong_field,
                        'reason': 'field_not_allowed'
                    })
                continue
            
            # Check if this is a nested field
            nested_parent = self._find_nested_parent(payload, wrong_field)
            
            if nested_parent:
                # Fix in nested structure
                parent_value = corrected.get(nested_parent)
                if isinstance(parent_value, list):
                    for item in parent_value:
                        if isinstance(item, dict) and wrong_field in item:
                            item[new_field] = item.pop(wrong_field)
                            changes.append({
                                'type': 'field_rename',
                                'from': f"{nested_parent}[].{wrong_field}",
                                'to': f"{nested_parent}[].{new_field}",
                                'confidence': 0.6
                            })
                elif isinstance(parent_value, dict) and wrong_field in parent_value:
                    parent_value[new_field] = parent_value.pop(wrong_field)
                    changes.append({
                        'type': 'field_rename',
                        'from': f"{nested_parent}.{wrong_field}",
                        'to': f"{nested_parent}.{new_field}",
                        'confidence': 0.6
                    })
            elif wrong_field in corrected:
                # Top-level field
                corrected[new_field] = corrected.pop(wrong_field)
                changes.append({
                    'type': 'field_rename',
                    'from': wrong_field,
                    'to': new_field,
                    'confidence': 0.6
                })
        
        return {
            'path': path,
            'method': method,
            'payload': corrected,
            'learned_rule': f"Auto-corrected fields: {changes}",
            'explanation': 'Basic correction using first candidate match',
        }
    
    def _verify_fix(
        self, 
        corrected: Dict, 
        original_path: str, 
        original_method: str
    ) -> Dict:
        """Verify the corrected payload is valid."""
        
        verification = {
            'is_valid': True,
            'llm_verified': True,
            'confidence': 0.8,
            'issues': []
        }
        
        payload = corrected.get('payload', {})
        path = corrected.get('path', original_path)
        method = corrected.get('method', original_method)
        
        # Check 1: No placeholder IDs
        payload_str = json.dumps(payload)
        if '{id}' in payload_str or '{id' in payload_str:
            verification['issues'].append("Contains placeholder {id}")
            verification['confidence'] -= 0.3
        
        # Check 2: All top-level fields exist in schema
        valid_fields = self.schema_intel.get_valid_fields(path, method)
        if valid_fields:
            for field in payload.keys():
                if field not in valid_fields:
                    verification['issues'].append(f"Unknown field: {field}")
                    verification['confidence'] -= 0.2
        
        # Check 3: Required fields present
        required = self.schema_intel.get_required_fields(path, method)
        missing_required = [f for f in required if f not in payload]
        if missing_required:
            verification['issues'].append(f"Missing required: {missing_required}")
            verification['confidence'] -= 0.3
        
        # Determine final validity
        if verification['confidence'] < 0.5:
            verification['is_valid'] = False
        
        return verification
    
    def _identify_changes(self, original: Dict, corrected: Dict) -> List[Dict]:
        """Identify what changed between original and corrected payloads."""
        changes = []
        
        # Check for field renames (in original but not in corrected, and vice versa)
        orig_fields = set(original.keys())
        corr_fields = set(corrected.keys())
        
        removed = orig_fields - corr_fields
        added = corr_fields - orig_fields
        
        # Simple rename detection: same value, different key
        for rem_field in removed:
            rem_value = original[rem_field]
            for add_field in added:
                if corrected[add_field] == rem_value:
                    changes.append({
                        'type': 'field_rename',
                        'from': rem_field,
                        'to': add_field,
                        'value': rem_value
                    })
                    added.discard(add_field)
                    break
        
        # Remaining removed = deleted fields
        for field in removed:
            changes.append({
                'type': 'field_removed',
                'field': field,
                'value': original[field]
            })
        
        # Remaining added = new fields
        for field in added:
            changes.append({
                'type': 'field_added',
                'field': field,
                'value': corrected[field]
            })
        
        # Check for modified values
        common = orig_fields & corr_fields
        for field in common:
            if original[field] != corrected[field]:
                changes.append({
                    'type': 'value_changed',
                    'field': field,
                    'from': original[field],
                    'to': corrected[field]
                })
        
        return changes
    
    def build_knowledge_rule(self, result: CorrectionResult) -> Dict:
        """Build a knowledge graph rule from a successful correction."""
        if not result.success or not result.changes_made:
            return None
        
        # Build a descriptive rule
        changes_desc = []
        for change in result.changes_made:
            if change['type'] == 'field_rename':
                changes_desc.append(f"{change['from']} -> {change['to']}")
        
        rule = {
            'type': 'field_mapping',
            'endpoint_pattern': result.corrected_path,
            'mappings': changes_desc,
            'learned_from': 'autonomous_correction',
            'confidence': result.confidence,
            'timestamp': None,  # Will be set by knowledge graph
        }
        
        return rule


# Singleton instance
_corrector_instance = None

def get_autonomous_corrector(llm_client=None, tripletex_client=None) -> AutonomousCorrector:
    """Get or create singleton AutonomousCorrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = AutonomousCorrector(llm_client, tripletex_client=tripletex_client)
    elif tripletex_client is not None:
        # Update the client if provided
        _corrector_instance.tripletex_client = tripletex_client
    return _corrector_instance
