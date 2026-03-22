import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class OpenApiContext:
    """
    Manages OpenAPI schema retrieval so the LLM knows what to query.
    Uses LLM to select the best endpoint from candidates.
    Includes example payloads for critical endpoints.
    """
    
    # Known good example payloads for critical endpoints
    EXAMPLE_PAYLOADS = {
        "/employee": {
            "POST": {
                "firstName": "Ola",
                "lastName": "Nordmann",
                "email": "ola.nordmann@example.com",
                "department": {"id": 1},
                "userType": "STANDARD",
                "dateOfBirth": "1990-01-15",
                "nationalIdentityNumber": "15019012345",
                "employeeNumber": "EMP001",
                "address": {
                    "addressLine1": "Storgata 1",
                    "postalCode": "0155",
                    "city": "Oslo"
                }
            }
        },
        "/customer": {
            "POST": {
                "name": "Acme AS",
                "email": "post@acme.no",
                "organizationNumber": "123456789",
                "postalAddress": {
                    "addressLine1": "Storgata 1",
                    "postalCode": "0155",
                    "city": "Oslo"
                },
                "invoiceSendMethod": "EMAIL"
            }
        },
        "/project": {
            "POST": {
                "name": "Website Development",
                "description": "New company website",
                "startDate": "2026-03-01",
                "endDate": "2026-06-30",
                "projectManager": {"id": 1},
                "customer": {"id": 1},
                "isOffer": False,
                "isInternal": False
            }
        },
        "/product": {
            "POST": {
                "name": "Consulting Hours",
                "number": "KONSULT",
                "description": "Hourly consulting services",
                "priceExcludingVatCurrency": 1000.00,
                "costExcludingVatCurrency": 500.00,
                "vatType": {"id": 3}
            }
        },
        "/department": {
            "POST": {
                "name": "Sales Department"
            }
        },
        "/ledger/voucher": {
            "POST": {
                "date": "2026-03-22",
                "description": "Manual journal entry",
                "postings": [
                    {
                        "account": {"id": 6020},
                        "amount": 1000.00,
                        "description": "Debit entry"
                    },
                    {
                        "account": {"id": 1090},
                        "amount": -1000.00,
                        "description": "Credit entry"
                    }
                ]
            }
        },
        "/order": {
            "POST": {
                "customer": {"id": 1},
                "orderDate": "2026-03-22",
                "deliveryDate": "2026-04-01"
            }
        },
        "/invoice": {
            "POST": {
                "invoiceDate": "2026-03-22",
                "invoiceDueDate": "2026-04-22",
                "orders": [
                    {
                        "customer": {"id": 1},
                        "orderDate": "2026-03-22",
                        "deliveryDate": "2026-03-22",
                        "orderLines": [
                            {
                                "product": {"id": 1},
                                "description": "Consulting services",
                                "count": 10.0,
                                "unitPriceExcludingVatCurrency": 1000.00
                            }
                        ]
                    }
                ]
            }
        },
        "/supplier": {
            "POST": {
                "name": "Office Supplies AS",
                "organizationNumber": "987654321",
                "invoiceEmail": "faktura@officesupplies.no"
            }
        },
        "/travelExpense": {
            "POST": {
                "employee": {"id": 1},
                "title": "Business trip description for Employee Name",
                "travelDetails": {
                    "isDayTrip": False,
                    "departureDate": "2026-03-20",
                    "returnDate": "2026-03-22",
                    "destination": "Oslo"
                },
                "costs": [
                    {
                        "date": "2026-03-22",
                        "comments": "Per diem 2 days",
                        "amountCurrencyIncVat": 1600.0,
                        "isPaidByEmployee": True
                    },
                    {
                        "date": "2026-03-22",
                        "comments": "Flight ticket",
                        "amountCurrencyIncVat": 3600.0,
                        "isPaidByEmployee": True
                    }
                ]
            }
        },
        "/incomingInvoice": {
            "POST": {
                "invoiceHeader": {
                    "vendorId": 1,
                    "invoiceNumber": "INV-001",
                    "invoiceDate": "2026-03-22",
                    "dueDate": "2026-04-22",
                    "invoiceAmount": 10000.0,
                    "description": "Consulting services"
                },
                "orderLines": [
                    {"description": "Consulting services", "count": 1, "unitPriceExcludingVatCurrency": 10000.0}
                ]
            }
        },
        "/timesheet/entry": {
            "POST": {
                "employee": {"id": 1},
                "project": {"id": 1},
                "activity": {"id": 1},
                "date": "2026-03-22",
                "hours": 8.0,
                "comment": "Development work"
            }
        },
        "/ledger/account": {
            "GET": {"isBankAccount": True, "count": 50}
        }
    }
    
    # Field mappings to help LLM avoid Norwegian field names
    FIELD_MAPPINGS = {
        # Norwegian → English field names
        "fodselsnummer": "nationalIdentityNumber",
        "fødselsnummer": "nationalIdentityNumber",
        "ansattnummer": "employeeNumber",
        "avdeling": "department",
        "kunde": "customer",
        "produkt": "product",
        "prosjekt": "project",
        "budsjett": "budget",  # Note: budget field might not exist
        "startdato": "startDate",
        "sluttdato": "endDate",
        "navn": "name",
        "beskrivelse": "description",
        "epost": "email",
        "telefon": "phoneNumber",
        "organisasjonsnummer": "organizationNumber",
        "pris": "priceExcludingVatCurrency",
        "mva": "vatType",
        "konto": "account",
        "bilag": "voucher",
        "postering": "posting",
        "ledager": "ledgerAccount",
        "ansettelsesdetaljer": "employments",
        "ansettelsestype": "employments",  # Use employments array instead
    }
    
    def __init__(self, schema_path: str = "../Docs/openapi.json", api_key: str = None):
        base_dir = os.path.dirname(__file__)
        self.schema_path = os.path.abspath(os.path.join(base_dir, schema_path))
        self.spec = self._load_schema()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if genai and self.api_key else None

    def _load_schema(self) -> Dict[str, Any]:
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"OpenAPI schema not found at {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_ref(self, ref_str: str) -> Dict[str, Any]:
        if not ref_str.startswith("#/"):
            return {}
        parts = ref_str[2:].split("/")
        current = self.spec
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {}
        return current

    def _dereference_schema(self, schema: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        if depth > 5 or not isinstance(schema, dict):
            return schema
        
        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            return self._dereference_schema(resolved, depth + 1)
            
        result = {}
        for k, v in schema.items():
            if isinstance(v, dict):
                result[k] = self._dereference_schema(v, depth + 1)
            elif isinstance(v, list):
                result[k] = [self._dereference_schema(i, depth + 1) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result

    def _get_candidate_endpoints(self, intent_description: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """Get candidate endpoints using expanded keyword matching."""
        intent_lower = intent_description.lower()
        keywords = intent_lower.split()
        
        # Extract action words
        action_words = []
        if any(w in intent_lower for w in ["create", "add", "new", "register", "make", "registrer",
                                            "crie", "erstellen", "oprett", "registre"]):
            action_words.extend(["create", "add", "post", "new"])
        if any(w in intent_lower for w in ["update", "change", "modify", "edit"]):
            action_words.extend(["update", "put", "change"])
        if any(w in intent_lower for w in ["delete", "remove", "cancel"]):
            action_words.extend(["delete", "remove"])
        if any(w in intent_lower for w in ["get", "find", "search", "list", "lookup", "fetch"]):
            action_words.extend(["get", "search", "find", "list"])
        if any(w in intent_lower for w in ["pay", "payment", "register payment", "paid"]):
            action_words.extend(["payment", "pay", ":payment"])
        
        # Entity detection
        entities = []
        if any(w in intent_lower for w in ["employee", "staff", "worker", "person", "ansatt"]):
            entities.append("employee")
        if any(w in intent_lower for w in ["customer", "client", "kunde", "buyer"]):
            entities.append("customer")
        if any(w in intent_lower for w in ["product", "item", "vare", "tjeneste", "goods"]):
            entities.append("product")
        if any(w in intent_lower for w in ["project", "prosjekt", "job"]):
            entities.append("project")
        if any(w in intent_lower for w in ["invoice", "faktura", "bill", "fatura", "factura", "rechnung"]):
            entities.append("invoice")
        if any(w in intent_lower for w in ["voucher", "ledger", "bilag", "posting", "accounting entry",
                                            "journal", "buchung", "avskriving", "abschreibung",
                                            "depreciation", "year-end", "årsavslutning", "jahresabschluss"]):
            entities.append("voucher")
        if any(w in intent_lower for w in ["department", "avdeling", "division"]):
            entities.append("department")
        if any(w in intent_lower for w in ["supplier", "vendor", "leverandør", "seller"]):
            entities.append("supplier")
        if any(w in intent_lower for w in ["order", "ordre", "purchase order"]):
            entities.append("order")
        if any(w in intent_lower for w in ["travel", "expense", "reise", "reiserekning", "reisekostenabrechnung",
                                            "diett", "per diem", "mileage", "trip"]):
            entities.append("travelExpense")
        if any(w in intent_lower for w in ["supplier invoice", "leverandørfaktura", "supplierinvoice",
                                            "incoming invoice", "inngående faktura", "lieferantenrechnung"]):
            entities.append("supplierInvoice")
            entities.append("incomingInvoice")
        if any(w in intent_lower for w in ["time", "hours", "timesheet", "log time", "timer",
                                            "zeiterfassung", "timeføring", "arbeidstimer"]):
            entities.append("timesheet")
        if any(w in intent_lower for w in ["credit", "credit note", "kreditnota", "gutschrift",
                                            "reverse", "reversal", "kreditere", "creditnote"]):
            entities.append("invoice")  # Credit notes use invoice endpoint
        
        possible_endpoints = []
        
        paths = self.spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() not in ["GET", "POST", "PUT", "DELETE"]:
                    continue
                    
                summary = details.get("summary", "").lower()
                desc = details.get("description", "").lower()
                tags = details.get("tags", [])
                operation_id = details.get("operationId", "").lower()
                
                score = 0
                # Entity matching (highest priority)
                for entity in entities:
                    if entity in path.lower():
                        score += 10
                    if entity in operation_id:
                        score += 5
                    if entity in summary:
                        score += 3
                
                # Action word matching
                for aw in action_words:
                    if aw in operation_id:
                        score += 4
                    if aw in summary:
                        score += 3
                    if method.lower() == aw or (aw == "create" and method == "post"):
                        score += 5
                    if aw == ":payment" and aw in path:
                        score += 8
                
                # Keyword matching
                for kw in keywords:
                    if kw in summary:
                        score += 2
                    if kw in path.lower():
                        score += 1
                
                if score > 0:
                    possible_endpoints.append({
                        "path": path,
                        "method": method.upper(),
                        "summary": details.get("summary"),
                        "description": details.get("description"),
                        "operationId": details.get("operationId"),
                        "score": score,
                        "details": details
                    })
                    
        possible_endpoints.sort(key=lambda x: x["score"], reverse=True)
        return possible_endpoints[:top_n]

    def _extract_schema_info(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant schema info for an endpoint with examples."""
        details = endpoint["details"]
        path = endpoint["path"]
        method = endpoint["method"]
        
        # Resolve query parameters
        params = details.get("parameters", [])
        resolved_params = []
        for p in params:
            resolved = self._dereference_schema(p)
            param_info = {
                "name": resolved.get("name"),
                "in": resolved.get("in"),
                "required": resolved.get("required", False),
                "type": resolved.get("schema", {}).get("type") if resolved.get("schema") else None,
                "description": resolved.get("description", "")[:100]
            }
            resolved_params.append(param_info)
        
        # Resolve request body schema
        req_body = details.get("requestBody", {})
        if "$ref" in req_body:
            req_body = self._resolve_ref(req_body["$ref"])
        
        content = req_body.get("content", {})
        json_schema = content.get("application/json", {}).get("schema", {})
        
        request_body_schema = None
        if json_schema:
            request_body_schema = self._dereference_schema(json_schema)
            request_body_schema = self._simplify_schema(request_body_schema)
        
        result = {
            "path": path,
            "method": method,
            "summary": endpoint["summary"],
            "operationId": endpoint.get("operationId"),
            "parameters": resolved_params,
            "request_body_schema": request_body_schema,
            "request_body_required": req_body.get("required", False) if isinstance(req_body, dict) else False
        }
        
        # Add example payload if available
        example = self.EXAMPLE_PAYLOADS.get(path, {}).get(method)
        if example:
            result["example_payload"] = example
        
        return result

    def _simplify_schema(self, schema: Dict[str, Any], max_depth: int = 3, current_depth: int = 0) -> Any:
        """Simplify schema to reduce token count while keeping essential info."""
        if current_depth >= max_depth:
            return {"type": schema.get("type", "object"), "_note": "... truncated"}
        
        if not isinstance(schema, dict):
            return schema
        
        result = {}
        
        # Essential fields to keep
        essential = ["type", "format", "required", "enum", "description", "properties", 
                     "items", "minimum", "maximum", "minLength", "maxLength", "pattern", "readOnly"]
        
        for key in essential:
            if key in schema:
                if key == "properties" and isinstance(schema[key], dict):
                    # Keep all properties but simplify them
                    result[key] = {}
                    for prop_name, prop_schema in schema[key].items():
                        # Skip readOnly fields for POST/PUT
                        if isinstance(prop_schema, dict) and prop_schema.get("readOnly"):
                            continue
                        simplified = self._simplify_schema(prop_schema, max_depth, current_depth + 1)
                        if isinstance(simplified, dict) and len(simplified) > 0:
                            result[key][prop_name] = simplified
                elif key == "items" and isinstance(schema[key], dict):
                    result[key] = self._simplify_schema(schema[key], max_depth, current_depth + 1)
                elif key == "description" and isinstance(schema[key], str):
                    result[key] = schema[key][:150]
                else:
                    result[key] = schema[key]
        
        return result

    def get_endpoints_for_intent(self, intent_description: str, use_llm_selection: bool = True) -> List[Dict[str, Any]]:
        """Get relevant endpoints for the intent with field mapping warnings."""
        candidates = self._get_candidate_endpoints(intent_description, top_n=12)
        
        if not candidates:
            return []
        
        # Extract full schema info
        results = [self._extract_schema_info(e) for e in candidates[:5]]
        
        # Add field mapping warnings to the first result
        if results:
            results[0]["field_mapping_note"] = (
                "IMPORTANT: Use EXACT English field names from the schema. "
                "Common mistakes: fodselsnummer -> nationalIdentityNumber, "
                "budsjett -> no budget field exists, avdeling -> department"
            )
        
        return results

    def validate_payload_against_schema(self, method: str, path: str, payload: Dict[str, Any]) -> List[str]:
        """Validate a payload against the endpoint's request schema."""
        paths = self.spec.get("paths", {})
        endpoint_info = paths.get(path, {})
        operation = endpoint_info.get(method.lower(), {})
        
        if not operation:
            return []
        
        errors = []
        
        # Get request body schema
        req_body = operation.get("requestBody", {})
        if "$ref" in req_body:
            req_body = self._resolve_ref(req_body["$ref"])
        
        if not req_body.get("required"):
            return []
        
        content = req_body.get("content", {})
        json_schema = content.get("application/json", {}).get("schema", {})
        if not json_schema:
            return []
        
        schema = self._dereference_schema(json_schema)
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in payload or payload[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check property types and translate Norwegian field names
        properties = schema.get("properties", {})
        for field, value in payload.items():
            # Check if this is a Norwegian field name
            if field.lower() in self.FIELD_MAPPINGS:
                correct_field = self.FIELD_MAPPINGS[field.lower()]
                errors.append(f"Wrong field name: '{field}' should be '{correct_field}'")
            
            if field in properties:
                prop_schema = properties[field]
                field_errors = self._validate_field_type(field, value, prop_schema)
                errors.extend(field_errors)
            else:
                # Field doesn't exist in schema
                if not schema.get("additionalProperties", True):
                    errors.append(f"Field '{field}' does not exist in the schema")
        
        return errors

    def _validate_field_type(self, field: str, value: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate a single field value against its schema."""
        errors = []
        schema_type = schema.get("type")
        
        # Handle reference types (objects)
        if schema_type is None and "$ref" in schema:
            # It's a reference type, value should be an object with id
            if not isinstance(value, dict):
                errors.append(f"Field {field} should be an object (e.g., {{'id': 1}}), got {type(value).__name__}")
            elif "id" not in value:
                errors.append(f"Field {field} object must have an 'id' property")
            return errors
        
        if schema_type == "string" and not isinstance(value, str):
            errors.append(f"Field {field} should be string, got {type(value).__name__}")
        elif schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"Field {field} should be integer, got {type(value).__name__}")
        elif schema_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"Field {field} should be number, got {type(value).__name__}")
        elif schema_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field {field} should be boolean, got {type(value).__name__}")
        elif schema_type == "array":
            if not isinstance(value, list):
                errors.append(f"Field {field} should be array, got {type(value).__name__}")
            elif value and schema.get("items"):
                # Validate first item
                item_schema = schema["items"]
                errors.extend(self._validate_field_type(f"{field}[0]", value[0], item_schema))
        elif schema_type == "object":
            if not isinstance(value, dict):
                errors.append(f"Field {field} should be object, got {type(value).__name__}")
        
        # Check enum values
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"Field {field} value '{value}' not in allowed values: {schema['enum']}")
        
        return errors

    def get_schema_for_endpoint(self, method: str, path: str) -> Optional[Dict[str, Any]]:
        """Get full schema info for a specific endpoint."""
        paths = self.spec.get("paths", {})
        endpoint_info = paths.get(path, {})
        operation = endpoint_info.get(method.lower(), {})
        
        if not operation:
            return None
        
        return self._extract_schema_info({
            "path": path,
            "method": method,
            "details": operation,
            "summary": operation.get("summary"),
            "operationId": operation.get("operationId")
        })

    def get_example_payload(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Get example payload for an endpoint."""
        return self.EXAMPLE_PAYLOADS.get(path, {}).get(method)
