"""
Error Analysis Module

The Tripletex API returns errors in Norwegian (mostly). This module parses
those errors and figures out what actually went wrong.

Norwegian error examples we handle:
- "Feltet eksisterer ikke i objektet" -> Field doesn't exist
- "Feltet må fylles ut" -> Field is required
- "Kan ikke være null" -> Cannot be null
- "Ugyldig avdeling" -> Invalid department

This allows the agent to self-correct without human intervention.

Built by Yassine Elhallaoui for NM i AI 2026.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a single validation error."""
    field: str
    message: str
    error_type: str  # unknown_field, wrong_type, missing_required, etc.
    original_error: str


@dataclass
class ErrorAnalysis:
    """Complete analysis of an API error response."""
    status_code: int
    path: str
    method: str
    error_type: str
    message: str
    validation_errors: List[ValidationError]
    raw_response: Dict
    
    # Additional context
    is_retryable: bool
    suggestions: Dict[str, Any]


class ErrorAnalyzer:
    """
    Analyzes Tripletex API errors and categorizes them.
    Provides actionable insights for self-correction.
    """
    
    # Norwegian error patterns
    ERROR_PATTERNS = {
        # Field doesn't exist
        'unknown_field': [
            r'Feltet eksisterer ikke i objektet',
            r'Field does not exist',
            r'Request mapping failed',
            r'Unknown field',
        ],
        # Missing required field
        'missing_required': [
            r'må fylles ut',
            r'must be filled',
            r'required',
            r'Required field',
            r'cannot be null',
            r'cannot be empty',
        ],
        # Wrong type
        'wrong_type': [
            r'Expected a (\w+) value',
            r'should be (\w+)',
            r'Invalid type',
            r'Expected type',
        ],
        # Reference/ID errors
        'invalid_reference': [
            r'ID-en må referere til et gyldig objekt',
            r'ID must refer to a valid object',
            r'Object not found',
            r'not found',
            r'does not exist',
        ],
        # Enum/validation errors
        'invalid_value': [
            r'Value .* is not one of the allowed enum values',
            r'Invalid value',
            r'must be one of',
        ],
        # Permission errors
        'permission_denied': [
            r'Forbidden',
            r'403',
            r'Permission denied',
            r'Access denied',
        ],
        # Not found (endpoint)
        'endpoint_not_found': [
            r'Object not found',
            r'404',
            r'Endpoint not found',
        ],
    }
    
    # Field extraction patterns
    FIELD_PATTERNS = [
        # Norwegian format: field | message
        r'\|\s*([^|]+)\s*\|\s*Feltet',
        # Field in validation messages
        r'field[:\s]+["\']?([^"\'\s,]+)["\']?',
        # body.fieldName
        r'body\.([^\s,]+)',
        # query.fieldName
        r'query\.([^\s,]+)',
        # In error message: 'fieldName' 
        r"'([^']+)'[^']*?(?:felt|field|parameter)",
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        self._compiled_patterns = {}
        for error_type, patterns in self.ERROR_PATTERNS.items():
            self._compiled_patterns[error_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def analyze_error(
        self, 
        error_response: Dict, 
        path: str = "", 
        method: str = ""
    ) -> ErrorAnalysis:
        """
        Main entry point: analyze an error response and return structured info.
        """
        status_code = error_response.get('status_code', 0)
        message = error_response.get('message', '')
        validation_messages = error_response.get('validationMessages', [])
        
        # Categorize the error
        error_type = self._categorize_error(message, status_code)
        
        # Extract validation errors
        validation_errors = self._extract_validation_errors(
            validation_messages, message
        )
        
        # Determine if retryable
        is_retryable = self._is_retryable(error_type, status_code)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            error_type, validation_errors, path, method
        )
        
        return ErrorAnalysis(
            status_code=status_code,
            path=path,
            method=method,
            error_type=error_type,
            message=message,
            validation_errors=validation_errors,
            raw_response=error_response,
            is_retryable=is_retryable,
            suggestions=suggestions
        )
    
    def _categorize_error(self, message: str, status_code: int) -> str:
        """Categorize error based on message and status code."""
        message_lower = message.lower()
        
        # Check status code first
        if status_code == 403:
            return 'permission_denied'
        if status_code == 404:
            return 'endpoint_not_found'
        
        # Check patterns
        for error_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    return error_type
        
        # Default categorization
        if status_code >= 400 and status_code < 500:
            return 'validation_error'
        elif status_code >= 500:
            return 'server_error'
        
        return 'unknown'
    
    def _extract_validation_errors(
        self, 
        validation_messages: List[Dict], 
        main_message: str
    ) -> List[ValidationError]:
        """Extract individual validation errors."""
        errors = []
        
        for msg in validation_messages:
            field = msg.get('field', '')
            message = msg.get('message', '')
            
            # If field not explicitly set, try to extract from message
            if not field:
                field = self._extract_field_name(message) or 'unknown'
            
            # Determine error type
            error_type = self._categorize_error(message, 0)
            
            errors.append(ValidationError(
                field=field,
                message=message,
                error_type=error_type,
                original_error=message
            ))
        
        # If no structured validation messages, try main message
        if not errors:
            field = self._extract_field_name(main_message)
            if field:
                error_type = self._categorize_error(main_message, 0)
                errors.append(ValidationError(
                    field=field,
                    message=main_message,
                    error_type=error_type,
                    original_error=main_message
                ))
        
        return errors
    
    def _extract_field_name(self, message: str) -> Optional[str]:
        """Extract field name from error message using patterns."""
        for pattern_str in self.FIELD_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(message)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for CamelCase words that might be field names
        camel_case = re.findall(r'[a-z]+[A-Z][a-zA-Z]+', message)
        if camel_case:
            # Return the most likely field name (longest camelCase)
            return max(camel_case, key=len)
        
        return None
    
    def _is_retryable(self, error_type: str, status_code: int) -> bool:
        """Determine if the error is potentially fixable."""
        retryable_types = [
            'unknown_field',
            'missing_required', 
            'wrong_type',
            'invalid_reference',
            'invalid_value',
        ]
        
        if error_type in retryable_types:
            return True
        
        # Some 4xx errors might be retryable with corrections
        if status_code in [400, 422, 460]:
            return True
        
        return False
    
    def _generate_suggestions(
        self, 
        error_type: str, 
        validation_errors: List[ValidationError],
        path: str,
        method: str
    ) -> Dict[str, Any]:
        """Generate suggestions for fixing the error."""
        suggestions = {
            'error_type': error_type,
            'fields_to_fix': [],
            'general_advice': '',
        }
        
        if error_type == 'unknown_field':
            suggestions['general_advice'] = (
                "The field name is incorrect. Check the schema for valid field names. "
                "Common fixes: unitPrice -> unitPriceExcludingVatCurrency, "
                "price -> priceExcludingVatCurrency, lines -> orderLines"
            )
            suggestions['fields_to_fix'] = [
                {'field': e.field, 'issue': 'unknown_field'} 
                for e in validation_errors
            ]
        
        elif error_type == 'missing_required':
            suggestions['general_advice'] = (
                "A required field is missing. Add the missing field to the payload."
            )
            suggestions['fields_to_fix'] = [
                {'field': e.field, 'issue': 'missing_required'} 
                for e in validation_errors
            ]
        
        elif error_type == 'wrong_type':
            suggestions['general_advice'] = (
                "Field has wrong type. Convert to the expected type."
            )
            suggestions['fields_to_fix'] = [
                {'field': e.field, 'issue': 'wrong_type'} 
                for e in validation_errors
            ]
        
        elif error_type == 'invalid_reference':
            suggestions['general_advice'] = (
                "Referenced entity does not exist. Create the entity first or use correct ID."
            )
            suggestions['fields_to_fix'] = [
                {'field': e.field, 'issue': 'invalid_reference'} 
                for e in validation_errors
            ]
        
        elif error_type == 'permission_denied':
            suggestions['general_advice'] = (
                "API key lacks permission for this operation. This endpoint cannot be used."
            )
            suggestions['is_blocked'] = True
        
        elif error_type == 'endpoint_not_found':
            suggestions['general_advice'] = (
                "Endpoint path is incorrect. Try alternative endpoints."
            )
            suggestions['try_alternatives'] = True
        
        return suggestions
    
    def get_error_summary(self, analysis: ErrorAnalysis) -> str:
        """Get human-readable summary of the error."""
        lines = [
            f"Error Type: {analysis.error_type}",
            f"Status Code: {analysis.status_code}",
            f"Message: {analysis.message[:100]}..." if len(analysis.message) > 100 else f"Message: {analysis.message}",
            f"Retryable: {analysis.is_retryable}",
            "",
            f"Validation Errors ({len(analysis.validation_errors)}):",
        ]
        
        for i, err in enumerate(analysis.validation_errors[:5], 1):
            lines.append(f"  {i}. Field '{err.field}': {err.error_type} - {err.message[:60]}")
        
        if analysis.suggestions.get('general_advice'):
            lines.extend(["", "Advice:", analysis.suggestions['general_advice']])
        
        return '\n'.join(lines)


# Singleton instance
_analyzer_instance = None

def get_error_analyzer() -> ErrorAnalyzer:
    """Get or create singleton ErrorAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ErrorAnalyzer()
    return _analyzer_instance
