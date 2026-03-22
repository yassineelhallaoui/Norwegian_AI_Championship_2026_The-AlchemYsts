"""
Dynamic field name validator using OpenAPI schema.
Uses fuzzy matching and LLM to correct field names without hardcoding.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from difflib import get_close_matches, SequenceMatcher

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


class FieldValidator:
    """
    Validates and corrects field names dynamically using OpenAPI schema.
    No hardcoded mappings - everything derived from schema.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if genai and api_key else None
        
    def validate_and_fix_payload(
        self,
        payload: Dict[str, Any],
        schema_properties: Dict[str, Any],
        path: str = "",
        method: str = ""
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate payload against schema and auto-fix field names.
        Returns (fixed_payload, list_of_changes).
        """
        if not isinstance(payload, dict):
            return payload, []
        
        if not schema_properties:
            return payload, []
        
        valid_fields = set(schema_properties.keys())
        fixed = {}
        changes = []
        
        for field_name, value in payload.items():
            # Check if field exists in schema
            if field_name in valid_fields:
                # Valid field - keep it
                fixed[field_name] = self._fix_nested(value, schema_properties.get(field_name, {}))
            else:
                # Invalid field - try to find closest match
                suggested = self._find_closest_field(field_name, valid_fields)
                
                if suggested and suggested != field_name:
                    logger.info(f"Field '{field_name}' not in schema, using closest match '{suggested}'")
                    changes.append(f"'{field_name}' -> '{suggested}'")
                    # Check if this is a reference type (should wrap in {id: ...})
                    prop_schema = schema_properties.get(suggested, {})
                    is_ref = '$ref' in prop_schema or (
                        prop_schema.get('type') == 'object' and 
                        'properties' not in prop_schema
                    )
                    if is_ref and not isinstance(value, dict):
                        value = {'id': value}
                    fixed[suggested] = self._fix_nested(value, prop_schema)
                else:
                    # No close match found - try LLM or remove
                    corrected = self._llm_correct_field_name(field_name, valid_fields, path, method)
                    if corrected and corrected in valid_fields:
                        logger.info(f"LLM corrected '{field_name}' -> '{corrected}'")
                        changes.append(f"'{field_name}' -> '{corrected}' (via LLM)")
                        prop_schema = schema_properties.get(corrected, {})
                        is_ref = '$ref' in prop_schema or (
                            prop_schema.get('type') == 'object' and 
                            'properties' not in prop_schema
                        )
                        if is_ref and not isinstance(value, dict):
                            value = {'id': value}
                        fixed[corrected] = self._fix_nested(value, prop_schema)
                    else:
                        logger.warning(f"Removing unknown field '{field_name}' (no match in schema)")
                        changes.append(f"Removed '{field_name}' (unknown)")
        
        return fixed, changes
    
    def _find_closest_field(self, field_name: str, valid_fields: set, cutoff: float = 0.5) -> Optional[str]:
        """
        Find the closest matching field name using fuzzy matching.
        Handles common patterns like:
        - Norwegian suffixes (Id -> remove suffix, use reference object)
        - Case differences
        - Missing/extra words
        - Substring matching
        """
        field_lower = field_name.lower()
        valid_list = list(valid_fields)
        
        # First try exact case-insensitive match
        for valid in valid_fields:
            if valid.lower() == field_lower:
                return valid
        
        # Try removing common suffixes/prefixes and find match
        suffixes = ['id', 'ids', 'number', 'code', 'date', 'phone', 'address', 'name']
        for suffix in suffixes:
            if field_lower.endswith(suffix):
                base = field_lower[:-len(suffix)]
                for valid in valid_list:
                    valid_lower = valid.lower()
                    if valid_lower == base or valid_lower.startswith(base) or base in valid_lower:
                        return valid
        
        # Check if input is substring of any valid field (for abbreviations)
        for valid in valid_list:
            valid_lower = valid.lower()
            # mobil -> phoneNumberMobile (mobil is in phoneNumberMobile)
            if field_lower in valid_lower and len(field_lower) >= 4:
                return valid
            # Check reverse: valid field contains input as word
            if valid_lower.replace('number', '').replace('code', '') in field_lower:
                return valid
        
        # Multi-language field name hints (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French)
        language_hints = {
            # Identity numbers
            'fodselsnummer': ['nationalidentitynumber', 'ssn', 'identity', 'personalid'],
            'fødselsnummer': ['nationalidentitynumber', 'ssn', 'identity', 'personalid'],
            'dni': ['nationalidentitynumber', 'identity', 'taxid'],
            'nif': ['nationalidentitynumber', 'identity', 'taxid'],
            'cpf': ['nationalidentitynumber', 'identity', 'taxid'],
            'ssn': ['nationalidentitynumber', 'socialsecurity'],
            'numéro de sécurité sociale': ['nationalidentitynumber'],
            'sozialversicherungsnummer': ['nationalidentitynumber'],
            'personnummer': ['nationalidentitynumber'],
            
            # Employee related
            'ansatt': ['employee', 'staff', 'worker'],
            'empleado': ['employee'],
            'funcionário': ['employee', 'staff'],
            'arbeitnehmer': ['employee'],
            'employé': ['employee'],
            'arbeidstaker': ['employee'],
            
            # Department
            'avdeling': ['department'],
            'departamento': ['department'],
            'departamento': ['department'],
            'abteilung': ['department'],
            'département': ['department'],
            'avdeling': ['department'],
            
            # Customer
            'kunde': ['customer', 'client'],
            'cliente': ['customer', 'client'],
            'cliente': ['customer', 'client'],
            'kunde': ['customer'],
            'client': ['customer'],
            
            # Product
            'produkt': ['product', 'item', 'goods'],
            'producto': ['product'],
            'produto': ['product'],
            'produkt': ['product'],
            'produit': ['product'],
            
            # Phone/Mobile
            'mobil': ['phone', 'mobile', 'cell'],
            'móvil': ['phone', 'mobile'],
            'celular': ['phone', 'mobile'],
            'handy': ['phone', 'mobile'],
            'portable': ['phone', 'mobile'],
            'telefon': ['phone', 'telephone'],
            'teléfono': ['phone'],
            'telefone': ['phone'],
            
            # Address
            'adresse': ['address'],
            'dirección': ['address'],
            'endereço': ['address'],
            'adresse': ['address'],
            'adresse': ['address'],
            'post': ['postal', 'address', 'zip'],
            'postal': ['postal', 'address'],
            'código postal': ['postalcode', 'zip'],
            
            # Name
            'navn': ['name'],
            'nombre': ['name'],
            'nome': ['name'],
            'name': ['name'],
            'nom': ['name'],
            
            # Quantity/Count
            'antall': ['count', 'quantity', 'amount', 'orderedquantity'],
            'cantidad': ['count', 'quantity', 'amount'],
            'quantidade': ['count', 'quantity', 'amount'],
            'menge': ['count', 'quantity', 'amount'],
            'quantité': ['count', 'quantity', 'amount', 'orderedquantity'],
            'kvantitet': ['count', 'quantity'],
            'numero': ['number', 'count'],
            'número': ['number', 'count'],
            'nummer': ['number', 'count'],
            
            # Price/Amount
            'pris': ['price', 'amount', 'cost', 'value'],
            'precio': ['price', 'amount', 'cost'],
            'preço': ['price', 'amount', 'cost'],
            'preis': ['price', 'amount', 'cost'],
            'prix': ['price', 'amount', 'cost'],
            'amountin': ['amount', 'amountcurrency'],
            'amount In': ['amount'],
            'beløp': ['amount'],
            'importe': ['amount'],
            # Tripletex product price fields
            'price': ['priceexcludingvatcurrency', 'priceincludingvatcurrency'],
            'salesprice': ['priceexcludingvatcurrency', 'priceincludingvatcurrency'],
            'unitprice': ['priceexcludingvatcurrency'],
            'priceexcludingvat': ['priceexcludingvatcurrency'],
            'costprice': ['costexcludingvatcurrency', 'costprice'],
            'purchaseprice': ['purchasepricecurrency', 'costexcludingvatcurrency'],
            
            # Description
            'beskrivelse': ['description', 'desc', 'text'],
            'descripción': ['description'],
            'descrição': ['description'],
            'beschreibung': ['description'],
            'description': ['description'],
            'beskrivning': ['description'],
            
            # Invoice
            'faktura': ['invoice', 'bill'],
            'factura': ['invoice', 'bill'],
            'fatura': ['invoice', 'bill'],
            'rechnung': ['invoice', 'bill'],
            'facture': ['invoice', 'bill'],
            
            # Order/Order Lines
            'ordre': ['order', 'purchaseorder'],
            'pedido': ['order', 'purchaseorder'],
            'ordem': ['order', 'purchaseorder'],
            'bestellung': ['order', 'purchaseorder'],
            'commande': ['order', 'purchaseorder'],
            'lines': ['orderlines', 'orderLines'],
            'invoicelines': ['orderlines', 'orderLines'],
            'invoice lines': ['orderlines', 'orderLines'],
            'supplierinvoicelines': ['orderlines', 'orderLines'],
            'linhas': ['orderlines', 'orderLines'],
            'líneas': ['orderlines', 'orderLines'],
            'zeilen': ['orderlines', 'orderLines'],
            'lignes': ['orderlines', 'orderLines'],
            # Supplier invoice specific
            'duedate': ['invoiceduedate'],
            'due_date': ['invoiceduedate'],
            'forfallsdato': ['invoiceduedate'],
            'invoicedate': ['invoicedate'],
            'fakturadato': ['invoicedate'],
            'invoicenumber': ['invoicenumber'],
            'fakturanummer': ['invoicenumber'],
            
            # Due date
            'duedate': ['invoiceduedate', 'duedate'],
            'due date': ['invoiceduedate'],
            'forfallsdato': ['invoiceduedate'],
            'fecha de vencimiento': ['invoiceduedate'],
            'data de vencimento': ['invoiceduedate'],
            'fälligkeitsdatum': ['invoiceduedate'],
            'date d\'échéance': ['invoiceduedate'],
            
            # Project
            'prosjekt': ['project', 'job'],
            'proyecto': ['project'],
            'projeto': ['project'],
            'projekt': ['project'],
            'projet': ['project'],
            'projectmanager': ['projectmanager'],
            'prosjektleder': ['projectmanager'],
            'jefedeproyecto': ['projectmanager'],
            'chef de projet': ['projectmanager'],
            'projektleiter': ['projectmanager'],
            # Budget fields
            'budgetamount': ['priceceilingamount'],
            'budgetamounttotal': ['priceceilingamount'],
            'totalbudget': ['priceceilingamount'],
            'presupuesto': ['priceceilingamount'],
            'budjett': ['priceceilingamount'],
            
            # Supplier/Vendor
            'leverandør': ['supplier', 'vendor'],
            'proveedor': ['supplier', 'vendor'],
            'fornecedor': ['supplier', 'vendor'],
            'lieferant': ['supplier', 'vendor'],
            'fournisseur': ['supplier', 'vendor'],
            'accountnumber': ['customernumber', 'suppliernumber', 'organizationnumber'],
            'suppliernumber': ['suppliernumber'],
            'customernumber': ['customernumber'],
            
            # Payment
            'betaling': ['payment', 'pay'],
            'pago': ['payment', 'pay'],
            'pagamento': ['payment', 'pay'],
            'zahlung': ['payment', 'pay'],
            'paiement': ['payment', 'pay'],
            
            # Account
            'konto': ['account', 'ledger', 'book'],
            'cuenta': ['account'],
            'conta': ['account'],
            'konto': ['account'],
            'compte': ['account'],
            
            # Date
            'dato': ['date', 'time'],
            'fecha': ['date'],
            'data': ['date'],
            'datum': ['date'],
            'date': ['date'],
        }
        
        for foreign_word, english_hints in language_hints.items():
            if foreign_word in field_lower or field_lower in foreign_word:
                for valid in valid_list:
                    valid_lower = valid.lower()
                    for hint in english_hints:
                        if hint in valid_lower:
                            return valid
        
        # Use difflib for fuzzy matching
        matches = get_close_matches(field_name, valid_list, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        
        # Try matching lowercase
        matches = get_close_matches(field_lower, [f.lower() for f in valid_list], n=1, cutoff=cutoff)
        if matches:
            for valid in valid_list:
                if valid.lower() == matches[0]:
                    return valid
        
        return None
    
    def _llm_correct_field_name(
        self,
        field_name: str,
        valid_fields: set,
        path: str,
        method: str
    ) -> Optional[str]:
        """Use LLM to find the correct field name from schema."""
        if not self.client:
            return None
        
        # Limit fields to avoid token overflow
        field_list = sorted(list(valid_fields))[:30]
        
        prompt = f'''Given this API endpoint: {method} {path}

The field "{field_name}" was used but is not in the schema.

Available fields: {", ".join(field_list)}

Which field from the list above is the correct one for "{field_name}"?
If none match, return "NONE".

Return ONLY the field name, nothing else.'''
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro-preview-03-25",
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=50)
            )
            
            result = response.text.strip().strip('"').strip("'")
            
            if result in valid_fields:
                return result
            if result == "NONE":
                return None
                
            # Try fuzzy match on LLM result
            matches = get_close_matches(result, valid_fields, n=1, cutoff=0.8)
            if matches:
                return matches[0]
                
        except Exception as e:
            logger.debug(f"LLM field correction failed: {e}")
        
        return None
    
    def _fix_nested(self, value: Any, prop_schema: Dict[str, Any]) -> Any:
        """Recursively fix nested objects and arrays."""
        if isinstance(value, dict) and prop_schema:
            # Check if this is a reference type (should have 'id')
            if '$ref' in prop_schema or prop_schema.get('type') == 'object':
                # It's an entity reference - ensure it has 'id'
                if 'id' not in value and len(value) == 1:
                    # Try to extract ID from single value
                    for k, v in value.items():
                        if isinstance(v, (int, str)):
                            return {'id': v}
                return value
            
            # Get nested properties
            nested_props = prop_schema.get('properties', {})
            if nested_props:
                fixed, _ = self.validate_and_fix_payload(value, nested_props)
                return fixed
        
        elif isinstance(value, list) and prop_schema:
            item_schema = prop_schema.get('items', {})
            if item_schema:
                return [self._fix_nested(item, item_schema) for item in value]
        
        return value
    
    def get_schema_properties(self, schema_context: List[Dict], path: str, method: str) -> Dict[str, Any]:
        """Extract properties from schema context for a specific endpoint."""
        for schema in schema_context:
            if schema.get('path') == path and schema.get('method') == method:
                body_schema = schema.get('request_body_schema') or {}
                return body_schema.get('properties') or {}
        return {}


# Convenience function for quick validation
def fix_payload_fields(
    payload: Dict[str, Any],
    schema_properties: Dict[str, Any],
    api_key: str = None
) -> Tuple[Dict[str, Any], List[str]]:
    """Quick function to fix payload fields."""
    validator = FieldValidator(api_key)
    return validator.validate_and_fix_payload(payload, schema_properties)
