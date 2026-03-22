try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Wrapper for Gemini inference.
    Handles synthesis of API calls based on prompt + schema + knowledge.
    Includes validation of generated payloads.
    """

    # Critical field mappings and workflow guidance
    FIELD_NAME_HINTS = """
CRITICAL WORKFLOW - STEP BY STEP:
1. If task mentions entity by NAME (e.g., "invoice 'Cloud Storage'", "customer Acme AS"):
   - FIRST call GET /invoice or GET /customer with name parameter to find the ID
   - THEN use that ID in subsequent calls
2. NEVER use placeholder like {id} in paths - use actual numbers from previous responses
3. For PROJECT creation: ALWAYS first GET /employee to find the employee ID for projectManager
   - projectManager requires a VALID employee ID from the system
   - If the task mentions a person by name/email, first search for them: GET /employee?email=...
   - Then use the returned employee ID as projectManager: {"id": <found_id>}
4. For multi-step tasks (create project + invoice + payment):
   - Step 1: GET existing entities (customer, employee, product) to find their IDs
   - Step 2: Create new entities (project, invoice) using those IDs
   - Step 3: Perform actions (payment, posting) using created entity IDs

CRITICAL FIELD NAME MAPPINGS (Norwegian → English):
- fodselsnummer / fødselsnummer → nationalIdentityNumber
- avdeling → department (use as object: {"id": 1})
- kunde → customer (use as object: {"id": 1})
- produkt → product (use as object: {"id": 1})
- prosjektleder → projectManager (use as object: {"id": 1})
- ansattnummer → employeeNumber
- organisasjonsnummer → organizationNumber
- postadresse → postalAddress
- faktura → invoice
- vare → product
- tjeneste → product/service

CRITICAL REFERENCE FORMAT:
- For entity references, ALWAYS use: {"id": <number>}
- Example: customer: {"id": 123}, department: {"id": 1}
- NEVER use just the ID number directly
- NEVER use customerId, departmentId (use customer, department with nested id)
- NEVER use {id} placeholder - always use actual numeric IDs from previous API responses

COMMON MISTAKES TO AVOID:
- Do NOT use: customerId: 123
- Do use: customer: {"id": 123}
- Do NOT use: fodselsnummer: "..."
- Do use: nationalIdentityNumber: "..."
- Do NOT use: path "/invoice/{id}/:payment" with {id} literal
- Do use: path "/invoice/12345/:payment" with actual ID from search

========================================
SCHEMA-SPECIFIC FIELD MAPPINGS
========================================

INVOICE CREATION (/invoice POST) - CRITICAL STRUCTURE:
The Invoice uses an "orders" array (NOT "orderLines" at top level).
Each order contains "orderLines" with line items.
Structure:
{
  "invoiceDate": "2026-03-22",
  "invoiceDueDate": "2026-04-22",
  "orders": [
    {
      "customer": {"id": X},
      "orderDate": "2026-03-22",
      "deliveryDate": "2026-03-22",
      "orderLines": [
        {
          "product": {"id": X},
          "description": "Item description",
          "count": 1,
          "unitPriceExcludingVatCurrency": 1000.00
        }
      ]
    }
  ]
}
WRONG: putting orderLines at invoice top level, using quantity/unitPrice
CORRECT: orders[] -> orderLines[] with count, unitPriceExcludingVatCurrency

INVOICE SEND (PUT /invoice/{id}/:send):
- To send an invoice to customer, use PUT /invoice/{id}/:send
- Query params: sendType (EMAIL/EHF), overrideEmailAddress (optional)
- Also possible on creation: POST /invoice with query param sendToCustomer=true

INVOICE PAYMENT (PUT /invoice/{id}/:payment):
- Uses QUERY PARAMETERS, NOT body payload
- Required query params: paymentDate, paymentTypeId, paidAmount
- paymentTypeId: get valid IDs from GET /invoice/paymentType
- Example: PUT /invoice/123/:payment?paymentDate=2026-03-22&paymentTypeId=1&paidAmount=5000
- The body/payload should be EMPTY ({})
- For PARTIAL payment, use paidAmount less than invoice total

INVOICE REMINDER (PUT /invoice/{id}/:createReminder):
- To create a reminder/late fee for an overdue invoice, use PUT /invoice/{id}/:createReminder
- Do NOT manually create a ledger voucher for reminder fees - use this action endpoint instead
- The system handles the accounting entries (debit/credit) automatically
- Query params may include: date, comment

OVERDUE INVOICE WORKFLOW:
1. GET /invoice (with invoiceDateFrom/To) to find the overdue invoice
2. PUT /invoice/{id}/:createReminder to create reminder/late fee
3. Create a NEW invoice for the reminder fee amount (POST /invoice with orderLines)
   - IMPORTANT: First GET /product to find a valid product ID, or create one
   - Do NOT use made-up product IDs
4. PUT /invoice/{id}/:send to send the reminder invoice
5. PUT /invoice/{id}/:payment for partial payment on the overdue invoice

CRITICAL: Always look up entity IDs before using them:
- GET /product before referencing product IDs in invoices
- GET /customer before referencing customer IDs
- GET /employee before referencing employee IDs
- NEVER use made-up/guessed IDs - always search first

ORDER LINE FIELDS:
- count (quantity - NOT "quantity")
- unitPriceExcludingVatCurrency (price per unit BEFORE VAT)
- unitPriceIncludingVatCurrency (price per unit WITH VAT)
- amountExcludingVatCurrency (line total BEFORE VAT)
- amountIncludingVatCurrency (line total WITH VAT)
- description (line item description)
- product: {"id": X} (product reference)
- vatType: {"id": X} (VAT type reference)
WRONG: unitPrice, price, amount, salesPrice, quantity
CORRECT: unitPriceExcludingVatCurrency, count

PRODUCT FIELDS:
- name (product name)
- number (product number)
- description
- priceExcludingVatCurrency (sale price before VAT)
- priceIncludingVatCurrency (sale price with VAT)
- costExcludingVatCurrency (cost price)
- vatType: {"id": X}
WRONG: price, salesPrice, unitPrice, priceExcludingVat
CORRECT: priceExcludingVatCurrency

SUPPLIER INVOICE SEARCH (/supplierInvoice GET):
- GET /supplierInvoice is for SEARCHING existing supplier invoices (read-only)
- Params: invoiceDateFrom, invoiceDateTo, invoiceNumber, supplierId

SUPPLIER INVOICE CREATION (/incomingInvoice POST):
- To CREATE a supplier invoice, use POST /incomingInvoice
- Body: {invoiceHeader: {vendorId: X, invoiceNumber: "...", invoiceDate: "...", dueDate: "...", invoiceAmount: 1000, description: "..."}, orderLines: [...]}
- vendorId is the supplier ID (get from GET /supplier)
WRONG: POST /supplierInvoice (does NOT exist)
CORRECT: POST /incomingInvoice

LEDGER VOUCHER / JOURNAL ENTRY (/ledger/voucher POST):
- The ONLY correct endpoint for journal entries/vouchers is /ledger/voucher
- WRONG paths: /voucher, /vouchers, /journal, /journal-entry, /journalEntry, /accounting/journalEntry, /accounting/journalEntries
- CORRECT path: /ledger/voucher
- Postings MUST be included in the body - they define the debit/credit entries
- Use gross amounts (positive = debit, negative = credit), postings must balance to zero
- account field must be object format: {"id": <account_number>}
- Example body:
  {
    "date": "2026-03-22",
    "description": "Journal entry description",
    "postings": [
      {"account": {"id": 6020}, "amount": 1000.00, "description": "Debit entry"},
      {"account": {"id": 1090}, "amount": -1000.00, "description": "Credit entry"}
    ]
  }
- Do NOT include: type, year, row, systemGenerated, id, version, guiRow in postings
- IMPORTANT: Some accounts (e.g. 1500 customer receivables, 2400 supplier payables) have
  system-controlled posting rules and will reject manual postings with error "systemgenererte".
  For reminder fees, use PUT /invoice/{id}/:createReminder instead of manual voucher.
  For general journal entries, use accounts like 6020 (expenses), 3000 (revenue), 1090 (bank)

EMPLOYEE CREATION (/employee POST):
- Only use fields that exist on the Employee schema
- VALID fields: firstName, lastName, email, dateOfBirth, nationalIdentityNumber,
  employeeNumber, department ({"id": X}), userType ("STANDARD"/"EXTENDED"/"NO_ACCESS"),
  phoneNumberMobile, phoneNumberHome, phoneNumberWork, address (NOT postalAddress!),
  bankAccountNumber, comments, dnumber, internationalId
- address uses Address schema: {addressLine1, addressLine2, postalCode, city, country: {id: X}}
- INVALID fields (do NOT use): startDate, yearlySalary, occupationCode,
  employmentPercentage, employmentDetails, salary, occupation, jobTitle, position,
  postalAddress, homeAddress
- department is REQUIRED: department: {"id": X} - MUST be included
- userType is REQUIRED: must be "STANDARD", "EXTENDED", or "NO_ACCESS"
- Employment details (salary, start date, percentage, occupation code) are managed
  via POST /employee/employment AFTER creating the employee
- Then POST /employee/employment/details for salary details

TRAVEL EXPENSE (/travelExpense POST):
- employee: {"id": X} (who traveled - REQUIRED, must look up employee first)
- title: "Trip description for Employee Name" (REQUIRED - NOT 'description' or 'text' or 'name')
- travelDetails: {isDayTrip: false, departureDate: "2026-03-20", returnDate: "2026-03-22", destination: "City"}
- costs: array of cost items, each with:
    - date: "2026-03-22"
    - comments: "Expense description" (NOT 'description' or 'text' or 'name')
    - amountCurrencyIncVat: 1000.0 (NOT 'amount' or 'amountNOK')
    - paymentType: {"id": X} (get from GET /travelExpense/paymentType)
    - costCategory: {"id": X} (get from GET /travelExpense/costCategory)
    - isPaidByEmployee: true
- Per diem should be ONE cost item: comments="Per diem/Diett X days", amountCurrencyIncVat=days*rate
- Each expense (flight, taxi, etc.) is a separate cost item
WRONG: description (top-level), text, name, numberOfDays, lines, rows, amount (in costs)
CORRECT: title (top-level), comments (in costs), amountCurrencyIncVat (in costs)

EMPLOYEE USER TYPE (enum - EXACT VALUES ONLY):
- "STANDARD" (limited access user)
- "EXTENDED" (full access user)
- "NO_ACCESS" (cannot log in)
WRONG: "user", "employee", "admin", "1", "0"
CORRECT: "STANDARD" or "EXTENDED"

PROJECT FIELDS:
- name (project name)
- projectManager: {"id": X} (employee ID)
- priceCeilingAmount (budget/limit amount)
- customer: {"id": X}
- startDate, endDate
WRONG: projectManagerId, budgetAmountTotal, budgetAmount
CORRECT: projectManager, priceCeilingAmount

TIME LOGGING / TIMESHEET ENTRY (/timesheet/entry POST):
- To log time on a project, use POST /timesheet/entry (NOT /project/{id}/time)
- Required fields:
  - employee: {"id": X}
  - project: {"id": X}
  - activity: {"id": X} (get from GET /activity)
  - date: "2026-03-22"
  - hours: 8.0
WRONG endpoints: /project/{id}/time, /project/{id}/hours, /project/time, /timesheetEntry
CORRECT endpoint: /timesheet/entry

CREDIT NOTE (PUT /invoice/{id}/:createCreditNote):
- To create a credit note reversing an invoice, use PUT /invoice/{id}/:createCreditNote
- The invoice ID must exist - first search with GET /invoice (with invoiceDateFrom/To params)
- Query params: date (REQUIRED), comment, creditNoteEmail, sendToCustomer, sendType
- The body should be EMPTY - all params go in query string
- Example: PUT /invoice/12345/:createCreditNote?date=2026-03-22
WRONG: /invoice/{id}/:credit, POST, creditNoteDate
CORRECT: PUT /invoice/{id}/:createCreditNote?date=2026-03-22

SUPPLIER INVOICE CREATION:
- To create a supplier invoice, use POST /incomingInvoice (NOT POST /supplierInvoice)
- /supplierInvoice is GET-only for searching existing supplier invoices
- Body structure: {invoiceHeader: {vendorId, invoiceNumber, invoiceDate, dueDate, invoiceAmount, currencyId, description}, orderLines: [...]}
WRONG: POST /supplierInvoice
CORRECT: POST /incomingInvoice

EMPLOYEE EMPLOYMENT (after creating employee):
- POST /employee/employment to create employment record
- Body: {employee: {id: X}, startDate, division: {id: X}, employmentEndReason, ...}
- Then POST /employee/employment/details for salary/occupation: {employment: {id: X}, date, annualSalary, occupationCode: {id: X}, percentageOfFullTimeEquivalent, workingHoursScheme}
- GET /employee/employment/occupationCode to find valid occupation codes
- GET /employee/employment/workingHoursScheme for valid schemes

CUSTOMER/CONTACT FIELDS:
- name (company name)
- organizationNumber (org.nr.)
- customerNumber (your customer ID)
- email
- phoneNumber
- postalAddress: {addressLine1, postalCode, city}
- physicalAddress: {addressLine1, postalCode, city}
WRONG: address, address1, accountNumber
CORRECT: postalAddress object with nested fields

SUPPLIER FIELDS:
- name
- organizationNumber
- supplierNumber (your supplier ID)
- email
- phoneNumber
- postalAddress object
WRONG: accountNumber
CORRECT: supplierNumber or organizationNumber

TIME ENTRY (/project/:id/time):
- employee: {"id": X}
- activity: {"id": X} (activity code)
- hours (number of hours)
- date (entry date)
- description
- hourlyRate (rate in NOK)

========================================
WHEN YOU GET "Feltet eksisterer ikke i objektet" ERROR:
This means "Field does not exist in object". 

Common fixes:
1. unitPrice → unitPriceExcludingVatCurrency
2. price → priceExcludingVatCurrency  
3. amount (in lines) → amountExcludingVatCurrency
4. lines (in supplierInvoice) → orderLines
5. dueDate (in supplierInvoice) → invoiceDueDate
6. lines (in travelExpense) → costs or perDiemCompensations
7. description (top-level in travel) → title + costs array
8. userType values → must be "STANDARD", "EXTENDED", or "NO_ACCESS"
9. address → postalAddress object with addressLine1, postalCode, city
10. budgetAmountTotal → priceCeilingAmount
"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25")
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. LLMEngine will not function properly.")
            self.client = None
        elif genai:
            self.client = genai.Client(api_key=self.api_key)
        else:
            logger.error("google-genai not installed!")
            self.client = None

    def synthesize_api_call(
        self, 
        task_prompt: str, 
        schema_context: List[Dict[str, Any]], 
        historical_rules: list[str],
        previous_responses: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Given the task, the API shape, and previous rules, output the optimal 
        API path, method, and payload to execute in JSON.
        """
        if not self.client:
            return {"error": "GenerativeAI not configured."}
            
        prev_str = ""
        if previous_responses:
            prev_str = f"PREVIOUS API RESPONSES (Use IDs from here for references):\n{json.dumps(previous_responses, indent=2)}\n\n"
        
        # Format schema context with examples
        schema_parts = []
        for i, schema in enumerate(schema_context):
            schema_text = f"--- Option {i+1}: {schema['method']} {schema['path']} ---\n"
            schema_text += f"Summary: {schema.get('summary', 'N/A')}\n"
            
            # Add example payload if available
            if "example_payload" in schema:
                schema_text += f"Example payload:\n{json.dumps(schema['example_payload'], indent=2)}\n"
            
            # Add schema structure
            if schema.get("request_body_schema"):
                schema_text += f"Schema structure:\n{json.dumps(schema['request_body_schema'], indent=2)[:1500]}\n"
            
            schema_parts.append(schema_text)
        
        schema_str = "\n".join(schema_parts) if schema_parts else "No matching endpoints found."
        
        # Format rules
        rules_str = ""
        if historical_rules:
            rules_str = "CRITICAL HISTORICAL RULES:\n"
            for i, rule in enumerate(historical_rules[:8], 1):
                rules_str += f"{i}. {rule}\n"
            rules_str += "\n"

        prompt = f'''You are an expert API agent for the Tripletex Accounting API.

TASK: {task_prompt}

{self.FIELD_NAME_HINTS}

{prev_str}{rules_str}AVAILABLE API ENDPOINTS:
{schema_str}

Your job is to generate the EXACT JSON payload needed for this task.

REQUIREMENTS:
1. Use ONLY the field names from the schema/example above
2. For entity references (customer, department, projectManager, etc.), use {{"id": <number>}} format
3. Date format: "yyyy-MM-dd" (e.g., "2026-03-22")
4. Use nationalIdentityNumber (NOT fodselsnummer)
5. Use employeeNumber (NOT ansattnummer)

RETURN FORMAT:
If task requires an API call, return ONLY this JSON structure:
{{
  "path": "/exact/api/path",
  "method": "POST",
  "payload": {{ ...exact body... }},
  "query_params": {{ }}
}}

If the task is complete based on PREVIOUS API RESPONSES, return:
{{"status": "completed"}}

IMPORTANT: 
- Return ONLY valid JSON, no markdown, no explanations
- Double-check all field names match the schema exactly
- Do NOT use Norwegian field names
'''
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            result = json.loads(response.text)
            
            # Handle case where LLM returns a list instead of dict
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    logger.warning(f"LLM returned a list, extracting first element")
                    result = result[0]
                else:
                    logger.error(f"LLM returned an invalid list: {result}")
                    return {"error": "LLM returned a list instead of object"}
            
            if not isinstance(result, dict):
                logger.error(f"LLM returned non-dict: {type(result)} - {result}")
                return {"error": f"LLM returned {type(result).__name__} instead of object"}
            
            # Validate the result structure
            if "error" in result:
                return result
            
            if result.get("status") == "completed":
                return result
            
            # Validate required fields
            if "path" not in result or "method" not in result:
                logger.error(f"LLM response missing required fields: {result}")
                return {"error": "LLM response missing path or method"}
            
            # Ensure payload exists
            if "payload" not in result:
                result["payload"] = {}
            
            # Ensure query_params exists
            if "query_params" not in result:
                result["query_params"] = {}
            
            # Note: Field validation is now done dynamically in the agent using schema
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return {"error": f"Invalid JSON from LLM: {e}"}
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return {"error": str(e)}

    def analyze_error_and_correct(
        self,
        method: str,
        path: str,
        failed_payload: Dict[str, Any],
        error_response: Dict[str, Any],
        schema_context: List[Dict[str, Any]],
        correction_history: List[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any], str]:
        """
        If an API call fails, analyze the error mapping to the schema.
        Returns the corrected path, method, payload, and a new plain English rule.
        """
        if not self.client:
            return path, method, failed_payload, "Model offline, cannot correct."

        # Extract error information
        error_message = error_response.get("message", "Unknown error")
        validation_messages = error_response.get("validationMessages", [])
        status_code = error_response.get("status_code", 0)
        
        # Format validation errors
        validation_str = ""
        if validation_messages:
            validation_str = "Validation Errors:\n"
            for msg in validation_messages:
                field = msg.get("field", "unknown")
                message = msg.get("message", "")
                validation_str += f"  - Field '{field}': {message}\n"
        
        # Get example payload if available
        example_str = ""
        for schema in schema_context:
            if schema.get("path") == path and schema.get("method") == method:
                if "example_payload" in schema:
                    example_str = f"\nExample correct payload for {path}:\n{json.dumps(schema['example_payload'], indent=2)}\n"
                break

        # Extract valid fields from schema for better guidance
        valid_fields_str = ""
        for schema in schema_context:
            if schema.get("path") == path and schema.get("method") == method:
                body_schema = schema.get("request_body_schema") or {}
                properties = body_schema.get("properties", {})
                if properties:
                    valid_fields_str = "\nValid fields for this endpoint:\n"
                    for field_name, field_info in list(properties.items())[:30]:
                        field_type = field_info.get("type", "unknown")
                        valid_fields_str += f"  - {field_name} ({field_type})\n"
                # Also check for nested schema in items (for line items)
                if "orderLines" in properties:
                    items = properties["orderLines"].get("items", {})
                    if "$ref" in items:
                        valid_fields_str += "\nFor orderLines array items, valid fields include:\n"
                        valid_fields_str += "  - count, unitPriceExcludingVatCurrency, amountExcludingVatCurrency\n"
                        valid_fields_str += "  - description, product, vatType, discount\n"
                break
        
        # Format correction history so LLM doesn't repeat mistakes
        history_str = ""
        if correction_history:
            history_str = "\nPREVIOUS FAILED ATTEMPTS (do NOT repeat these):\n"
            for h in correction_history:
                history_str += f"  Attempt {h['attempt']}: {h['method']} {h['path']} -> {h['error'][:150]}\n"
            history_str += "\nYou MUST try something DIFFERENT from the above attempts.\n"

        prompt = f'''An API request failed.

Request: {method} {path}
Status Code: {status_code}
Error Message: {error_message}
{validation_str}
Failed Payload: {json.dumps(failed_payload, indent=2)}
{example_str}
{valid_fields_str}
{history_str}
{self.FIELD_NAME_HINTS}

Fix the request. Common issues:
1. Wrong field name (e.g., fodselsnummer → nationalIdentityNumber)
2. Wrong reference format (use {{"id": X}}, not just X or XId)
3. Field doesn't exist in schema (remove it or use correct name from valid fields list)
4. Missing required field
5. Wrong endpoint entirely (check if different path is needed)

CRITICAL: If error says "Feltet eksisterer ikke i objektet" (field does not exist):
- Check the valid fields list above
- For price fields: use unitPriceExcludingVatCurrency, NOT unitPrice
- For line items: use orderLines with proper structure
- For dates in supplierInvoice: use invoiceDueDate, NOT dueDate

Return ONLY a JSON object:
{{
  "corrected_path": "/the/new/api/path",
  "corrected_method": "POST",
  "corrected_payload": {{ ...fixed payload... }},
  "learned_rule": "Plain English rule about what we learned"
}}
'''
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            
            corrected_path = data.get("corrected_path", path)
            corrected_method = data.get("corrected_method", method)
            corrected_payload = data.get("corrected_payload", failed_payload)
            learned_rule = data.get("learned_rule", "")
            
            # Ensure payload is valid
            if not isinstance(corrected_payload, dict):
                corrected_payload = failed_payload
            
            if not learned_rule or len(learned_rule) < 10:
                learned_rule = f"Failed with {status_code}: {error_message[:100]}"
            
            return (
                corrected_path,
                corrected_method,
                corrected_payload,
                learned_rule
            )
        except json.JSONDecodeError as e:
            logger.error(f"LLM correction returned invalid JSON: {e}")
            return path, method, failed_payload, f"Failed to parse correction: {error_message}"
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return path, method, failed_payload, f"Correction failed: {str(e)[:100]}"

    def select_alternative_endpoint(
        self,
        task_prompt: str,
        failed_path: str,
        failed_method: str,
        error_message: str,
        all_candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """When an endpoint completely fails (404, 403), ask LLM to pick an alternative."""
        if not self.client or not all_candidates:
            return None
        
        # Filter out the failed endpoint
        alternatives = [
            c for c in all_candidates 
            if not (c.get("path") == failed_path and c.get("method") == failed_method)
        ][:8]
        
        if not alternatives:
            return None
        
        prompt = f'''The API endpoint {failed_method} {failed_path} failed with error: {error_message}

Task: {task_prompt}

Alternative endpoints:
{json.dumps([{"path": c["path"], "method": c["method"], "summary": c.get("summary", "")} for c in alternatives], indent=2)}

Select the best alternative endpoint for this task, or return -1 if none suitable.

Return ONLY: {{"selected_index": 0}} or {{"selected_index": -1}}
'''
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            selected_idx = data.get("selected_index", -1)
            
            if isinstance(selected_idx, int) and 0 <= selected_idx < len(alternatives):
                return alternatives[selected_idx]
            return None
        except Exception as e:
            logger.error(f"Failed to select alternative endpoint: {e}")
            return None
