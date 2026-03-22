import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.llm_engine import LLMEngine
from core.knowledge_graph import KnowledgeGraph
from core.openapi_context import OpenApiContext
from core.field_validator import FieldValidator
from core.autonomous_corrector import AutonomousCorrector, get_autonomous_corrector

logger = logging.getLogger(__name__)

# To seamlessly integrate with Code_YassY_v2 base client
try:
    from tripletex_client import TripletexAPIError
except ImportError:
    class TripletexAPIError(Exception):
        pass

class V3Agent:
    """
    Main orchestrator for the Tripletex AI Accounting Agent.
    
    This class implements a ReAct (Reasoning + Acting) loop that:
    1. Understands natural language accounting tasks
    2. Maps them to Tripletex API endpoints
    3. Executes API calls with automatic error correction
    4. Handles multi-step workflows (cascading tasks)
    
    Built by Yassine Elhallaoui for NM i AI 2026 competition.
    """

    def __init__(self, client: Any, api_key: str = None):
        self.client = client
        self.llm = LLMEngine()
        self.knowledge = KnowledgeGraph()
        self.schema_ctx = OpenApiContext()
        self.field_validator = FieldValidator(api_key)
        self.all_endpoint_candidates = []
        # Autonomous correction system - self-healing API errors
        self.corrector = get_autonomous_corrector(self.llm, self.client)

    def process_task(self, prompt: str) -> Dict[str, Any]:
        """
        Main execution loop that processes a natural language accounting task.
        
        This is the heart of the agent - it takes a user prompt like 
        "Create invoice for customer Acme AS" and translates it into 
        actual Tripletex API calls.
        
        The loop can handle multi-step tasks (like creating a customer 
        first, then an order, then an invoice) automatically.
        
        Args:
            prompt: Natural language task description (supports multiple languages)
            
        Returns:
            Dict with status, history of API calls, and attempt count
        """
        logger.info(f"=" * 60)
        logger.info(f"Processing new task: {prompt}")
        logger.info(f"=" * 60)
        
        # Step 1: Find relevant API endpoints from the OpenAPI schema
        # based on the user's intent (e.g., "invoice" -> POST /invoice)
        schema_context = self.schema_ctx.get_endpoints_for_intent(prompt, use_llm_selection=True)
        
        # Step 2: Load historical rules from previous successful/failed tasks
        # This helps the agent learn from past mistakes
        rules = self.knowledge.get_applicable_rules(prompt)
        
        logger.info(f"Selected {len(schema_context)} endpoint(s) for this task")
        logger.info(f"Retrieved {len(rules)} relevant rule(s) from knowledge graph")

        responses = []
        max_turns = 6  # Prevent infinite loops
        turn = 0
        total_attempts = 0
        last_error = None

        # ReAct Loop: Reasoning + Acting
        # The LLM decides what API call to make next based on results so far
        while turn < max_turns:
            turn += 1
            logger.info(f"\n--- Task Turn {turn}/{max_turns} ---")
            
            # Ask Gemini what to do next
            try:
                api_instruction = self.llm.synthesize_api_call(
                    prompt, schema_context, rules, previous_responses=responses
                )
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                return {"status": "failed", "error": f"LLM error: {e}", "attempts": total_attempts}
            
            logger.debug(f"LLM instruction: {json.dumps(api_instruction, indent=2)}")
            
            if "error" in api_instruction:
                logger.error(f"LLM returned error: {api_instruction['error']}")
                return {"status": "failed", "error": api_instruction["error"], "history": responses, "attempts": total_attempts}
            
            # LLM can signal task completion (e.g., "invoice created successfully")
            if api_instruction.get("status") == "completed":
                logger.info("Task completed successfully determined by LLM.")
                return {"status": "success", "history": responses, "attempts": total_attempts}

            # Extract the API call details from LLM response
            method = api_instruction.get('method', 'GET')
            path = api_instruction.get('path', '')
            payload = api_instruction.get('payload') or {}
            query_params = api_instruction.get('query_params') or {}
            
            logger.info(f"Attempting: {method} {path}")
            if payload:
                logger.debug(f"Original payload: {json.dumps(payload, indent=2)}")

            # Step 3: Apply preprocessing fixes (hardcoded rules from production logs)
            # These fix common issues like wrong field names, missing required fields, etc.
            path, method, payload, query_params = self._preprocess(path, method, payload, query_params, responses)

            # Step 4: Validate payload against Tripletex OpenAPI schema
            # This catches errors before sending to the API
            schema_props = self.field_validator.get_schema_properties(schema_context, path, method)
            if schema_props and payload:
                payload, changes = self.field_validator.validate_and_fix_payload(
                    payload, schema_props, path, method
                )
                if changes:
                    logger.info(f"Field name corrections: {changes}")
                    logger.info(f"Fixed payload: {json.dumps(payload, indent=2)}")

            # Additional validation against schema
            validation_errors = self.schema_ctx.validate_payload_against_schema(method, path, payload)
            if validation_errors:
                logger.warning(f"Payload validation errors: {validation_errors}")
            
            # Preflight validation: Check payload against schema before sending
            if payload and method.upper() in ['POST', 'PUT', 'PATCH']:
                preflight_issues = self.corrector.schema_intel.get_valid_fields(path, method)
                if preflight_issues:
                    unknown_fields = [f for f in payload.keys() if f not in preflight_issues]
                    if unknown_fields:
                        logger.warning(f"⚠️  Preflight warning: Unknown fields in payload: {unknown_fields}")
                        # Try to auto-correct before sending
                        for wrong_field in unknown_fields:
                            suggestion = self.corrector.schema_intel.suggest_field_fix(
                                path, method, wrong_field
                            )
                            if suggestion.get('suggestions'):
                                best = suggestion['suggestions'][0]
                                logger.info(f"   Suggest: {wrong_field} -> {best['field']} (score: {best['score']:.2f})")

            # 3. Inner Execution & Self-Correction Loop for the single API call
            step_success = False
            max_retries = 3
            step_attempts = 0
            last_error = None
            
            # Store original instruction for potential endpoint switching
            original_path = path
            original_method = method
            original_payload = (payload or {}).copy()
            correction_history = []  # Track what we already tried

            while not step_success and step_attempts < max_retries:
                step_attempts += 1
                total_attempts += 1
                
                try:
                    response = self._execute_dynamic(method, path, payload, query_params)
                    
                    step_success = True
                    responses.append({
                        "executed_method": method,
                        "executed_path": path,
                        "executed_payload": payload,
                        "response": response
                    })
                    logger.info(f"Step succeeded on attempt {step_attempts}.")
                    
                    # Record entity relation if we created something
                    if method.upper() == "POST" and isinstance(response, dict):
                        self._record_entity_from_response(prompt, response, path)
                    
                    break  # Break inner loop, go back to outer turn loop
                    
                except TripletexAPIError as e:
                    error_response = {}
                    try:
                        error_response = e.payload if hasattr(e, 'payload') else {"message": str(e)}
                    except Exception:
                        error_response = {"message": str(e)}
                    
                    status_code = getattr(e, 'status_code', 400)
                    last_error = {
                        "message": str(e), 
                        "details": error_response, 
                        "status_code": status_code
                    }
                    logger.warning(f"Tripletex API Error ({status_code}): {last_error['message'][:200]}")
                    
                    # Handle specific error types
                    if status_code == 404 and step_attempts == 1:
                        # Endpoint not found - try to switch endpoints
                        logger.info("404 error - attempting endpoint switch...")
                        alternative = self._try_alternative_endpoint(prompt, method, path, error_response)
                        if alternative:
                            path = alternative.get("path", path)
                            method = alternative.get("method", method)
                            logger.info(f"Switched to alternative: {method} {path}")
                            continue
                    
                    if status_code == 403:
                        # Permission error - add to knowledge and try alternative
                        logger.warning("403 Forbidden - recording in knowledge graph")
                        self.knowledge.add_rule(
                            f"Do not use {path}, user lacks permissions for this endpoint.",
                            {"endpoint": path, "error_type": "403"}
                        )
                        # Try to find alternative endpoint
                        if step_attempts < 3:
                            alternative = self._try_alternative_endpoint(prompt, method, path, error_response)
                            if alternative:
                                path = alternative.get("path", path)
                                method = alternative.get("method", method)
                                continue
                    
                    if status_code == 422:
                        # Validation error - detailed logging
                        validation_msgs = error_response.get("validationMessages") or []
                        for msg in validation_msgs:
                            if isinstance(msg, dict):
                                logger.warning(f"Validation error: {msg.get('field')} - {msg.get('message')}")

                        # Auto-remediate bank account blocker for invoices
                        error_text = str(last_error.get("message", "")).lower()
                        if 'bankkontonummer' in error_text or 'bank account' in error_text:
                            logger.info("Detected bank account blocker - attempting auto-remediation via /ledger/account")
                            remediated = self._remediate_bank_account()
                            if remediated:
                                logger.info("Bank account remediation succeeded - retrying invoice")
                                continue  # Retry the same call
                    
                except Exception as e:
                    last_error = {"message": str(e), "status_code": 500}
                    logger.warning(f"Unexpected Execution Error: {last_error['message'][:200]}")
                
                # Autonomous correction with LLM verification
                if step_attempts < max_retries:
                    logger.info(f"Attempting autonomous correction (attempt {step_attempts}/{max_retries})...")

                    # Record what we tried so far to avoid repeating
                    correction_history.append({
                        "attempt": step_attempts,
                        "path": path,
                        "method": method,
                        "payload_keys": list((payload or {}).keys()),
                        "error": str(last_error.get("message", ""))[:200]
                    })

                    # Step 1: Try autonomous correction with schema intelligence
                    correction_result = self.corrector.correct_api_error(
                        method, path, payload, last_error, schema_context
                    )
                    
                    if correction_result.success and correction_result.llm_verified:
                        logger.info(f"✅ Autonomous correction successful ({len(correction_result.changes_made)} changes)")
                        for change in correction_result.changes_made:
                            logger.info(f"   - {change.get('type')}: {change.get('from', '')} -> {change.get('to', change.get('field', ''))}")
                        
                        path = correction_result.corrected_path
                        method = correction_result.corrected_method
                        payload = correction_result.corrected_payload
                        new_rule = correction_result.learned_rule
                    else:
                        # Step 2: Fall back to traditional LLM correction
                        logger.info(f"⚠️  Autonomous correction had issues, falling back to LLM...")
                        if correction_result.changes_made:
                            logger.info(f"   (Using {len(correction_result.changes_made)} auto-detected changes as hints)")
                        
                        corrected_path, corrected_method, corrected_payload, new_rule = self.llm.analyze_error_and_correct(
                            method, path, payload, last_error, schema_context,
                            correction_history=correction_history
                        )
                        path = corrected_path
                        method = corrected_method
                        payload = corrected_payload

                    # Re-apply preprocessing on corrected instruction
                    path, method, payload, query_params = self._preprocess(path, method, payload, query_params, responses)
                    
                    # Learn from successful corrections
                    if new_rule and len(new_rule) > 10:
                        self.knowledge.add_rule(
                            new_rule,
                            {"endpoint": path, "error_type": str(last_error.get("status_code", "")), "verified": True}
                        )
                        logger.info(f"Learned new rule: {new_rule[:100]}...")
                        
                        if new_rule not in rules:
                            rules.append(new_rule)
                    
                    # Also store successful field mappings in knowledge graph
                    if correction_result.success and correction_result.changes_made:
                        for change in correction_result.changes_made:
                            if change.get('type') == 'field_rename':
                                mapping_rule = f"Field '{change['from']}' should be '{change['to']}' for {path}"
                                self.knowledge.add_rule(
                                    mapping_rule,
                                    {"endpoint": path, "wrong_field": change['from'], "correct_field": change['to'], "auto_learned": True}
                                )

            if not step_success:
                logger.error(f"Failed to complete step after {max_retries} retries.")
                return {
                    "status": "failed",
                    "error": last_error,
                    "attempts": total_attempts,
                    "history": responses
                }

        logger.error(f"Max cascading turns ({max_turns}) reached.")
        return {
            "status": "failed",
            "error": "Max cascading turns reached.",
            "attempts": total_attempts,
            "history": responses
        }

    def _preprocess(self, path: str, method: str, payload: Dict[str, Any],
                     query_params: Dict[str, Any], responses: List[Dict[str, Any]]
                     ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        """Apply all deterministic payload/path fixes before execution.

        This runs BOTH on initial LLM output AND after self-correction so that
        the same fixups are always applied regardless of the code path.
        """
        payload = json.loads(json.dumps(payload or {}))
        query_params = dict(query_params or {})

        # --- GET /invoice requires date range params ---
        if path == '/invoice' and method == 'GET':
            if 'invoiceDateFrom' not in query_params:
                query_params['invoiceDateFrom'] = '2020-01-01'
            if 'invoiceDateTo' not in query_params:
                query_params['invoiceDateTo'] = '2026-12-31'

        # --- Redirect wrong voucher paths ---
        if method in ('POST', 'PUT') and path in (
                '/voucher', '/vouchers', '/journal', '/journal-entry',
                '/journalEntry', '/journal_entry', '/accounting/journalEntry',
                '/accounting/journalEntries', '/accounting/journal'):
            path = '/ledger/voucher'

        # --- Ledger voucher postings cleanup ---
        if path.startswith('/ledger/voucher') and method == 'POST' and payload:
            postings = payload.get('postings', [])
            if isinstance(postings, list):
                for posting in postings:
                    if isinstance(posting, dict):
                        acct = posting.get('account')
                        if isinstance(acct, (int, float)):
                            posting['account'] = {"id": int(acct)}
                        for sys_field in ('systemGenerated', 'row', 'guiRow', 'id', 'version', 'type'):
                            posting.pop(sys_field, None)

        # --- Invoice POST: wrap orderLines into orders ---
        if path == '/invoice' and method == 'POST' and payload:
            if 'orderLines' in payload and 'orders' not in payload:
                customer = payload.pop('customer', None)
                order_lines = payload.pop('orderLines')
                order_obj = {
                    "orderDate": payload.get('invoiceDate', '2026-03-22'),
                    "deliveryDate": payload.get('invoiceDate', '2026-03-22'),
                    "orderLines": order_lines
                }
                if customer:
                    order_obj["customer"] = customer
                payload['orders'] = [order_obj]
            for order in payload.get('orders', []):
                if isinstance(order, dict):
                    for line in order.get('orderLines', []):
                        if isinstance(line, dict):
                            if 'quantity' in line and 'count' not in line:
                                line['count'] = line.pop('quantity')
                            if 'unitPrice' in line and 'unitPriceExcludingVatCurrency' not in line:
                                line['unitPriceExcludingVatCurrency'] = line.pop('unitPrice')
                            if 'price' in line and 'unitPriceExcludingVatCurrency' not in line:
                                line['unitPriceExcludingVatCurrency'] = line.pop('price')
                            if 'amount' in line and 'amountExcludingVatCurrency' not in line:
                                line['amountExcludingVatCurrency'] = line.pop('amount')
                            for inv_line_field in ('project', 'projectId', 'orderIds', 'orderId'):
                                line.pop(inv_line_field, None)
            # Strip top-level invalid fields
            for field in ('orderIds', 'orderId'):
                payload.pop(field, None)

        # --- Fix plural invoices path ---
        if path.startswith('/invoices/'):
            path = '/invoice/' + path[len('/invoices/'):]

        # --- Invoice payment: move fields to query params ---
        if '/:payment' in path and method == 'PUT' and payload:
            for field in ('paymentDate', 'paymentTypeId', 'paidAmount', 'paidAmountCurrency'):
                if field in payload:
                    query_params[field] = payload.pop(field)

        # --- Resolve {id} placeholders ---
        if '{id}' in path:
            actual_id = None
            for id_field in ('invoiceId', 'id', 'employeeId', 'customerId',
                             'projectId', 'supplierId', 'voucherId'):
                if payload and id_field in payload:
                    actual_id = payload.pop(id_field)
                    break
            if not actual_id:
                for id_field in ('invoiceId', 'id'):
                    if id_field in query_params:
                        actual_id = query_params.pop(id_field)
                        break
            # Try to resolve from previous responses
            if not actual_id and responses:
                for prev in reversed(responses):
                    resp = prev.get("response", {})
                    if isinstance(resp, dict):
                        val = resp.get("value", {})
                        if isinstance(val, dict) and val.get("id"):
                            actual_id = val["id"]
                            break
            if actual_id:
                path = path.replace('{id}', str(actual_id))

        # --- Employee: strip invalid fields, fix mappings ---
        if '/employee' in path and method == 'POST' and payload:
            for field in ('startDate', 'endDate', 'yearlySalary', 'salary', 'monthlySalary',
                          'occupationCode', 'employmentPercentage', 'employmentDetails',
                          'employmentType', 'jobTitle', 'position', 'occupation',
                          'percentageOfFullTimeEquivalent', 'annualSalary', 'hourlyWage',
                          'workingHoursScheme', 'employmentEndReason'):
                payload.pop(field, None)
            # Norwegian ID field aliases
            if 'fodselsnummer' in payload and 'nationalIdentityNumber' not in payload:
                payload['nationalIdentityNumber'] = payload.pop('fodselsnummer')
            if 'fødselsnummer' in payload and 'nationalIdentityNumber' not in payload:
                payload['nationalIdentityNumber'] = payload.pop('fødselsnummer')
            # Fix userType
            user_type = payload.get('userType')
            if user_type and user_type not in ('STANDARD', 'EXTENDED', 'NO_ACCESS'):
                payload['userType'] = 'STANDARD'
            if not payload.get('userType'):
                payload['userType'] = 'STANDARD'
            # postalAddress → address
            if 'postalAddress' in payload and 'address' not in payload:
                payload['address'] = payload.pop('postalAddress')
            if 'homeAddress' in payload and 'address' not in payload:
                payload['address'] = payload.pop('homeAddress')

        # --- Supplier invoice: POST /supplierInvoice doesn't exist, redirect to /incomingInvoice ---
        if path == '/supplierInvoice' and method == 'POST':
            path = '/incomingInvoice'
            logger.info("Redirected POST /supplierInvoice to POST /incomingInvoice")
            # Transform payload to incomingInvoice format if needed
            if payload and 'invoiceHeader' not in payload:
                header = {}
                # Map supplier to vendorId
                supplier = payload.pop('supplier', None)
                if isinstance(supplier, dict) and 'id' in supplier:
                    header['vendorId'] = supplier['id']
                elif isinstance(supplier, (int, float)):
                    header['vendorId'] = int(supplier)
                for hf, pf in [('invoiceNumber', 'invoiceNumber'), ('invoiceDate', 'invoiceDate'),
                                ('dueDate', 'invoiceDueDate'), ('dueDate', 'dueDate'),
                                ('invoiceAmount', 'amount'), ('invoiceAmount', 'amountCurrency'),
                                ('description', 'description'), ('currencyId', 'currencyId')]:
                    if pf in payload and hf not in header:
                        header[hf] = payload.pop(pf)
                # Also try totalAmount
                if 'totalAmount' in payload and 'invoiceAmount' not in header:
                    header['invoiceAmount'] = payload.pop('totalAmount')
                order_lines = payload.pop('orderLines', payload.pop('lines', payload.pop('invoiceLines', [])))
                payload = {"invoiceHeader": header, "orderLines": order_lines}

        # --- Fix /incomingInvoice payload (when LLM sends directly without redirect) ---
        if path == '/incomingInvoice' and method == 'POST' and payload:
            # If LLM sent flat fields instead of invoiceHeader structure
            if 'invoiceHeader' not in payload and any(k in payload for k in ('invoiceNumber', 'invoiceDate', 'supplier', 'vendorId')):
                header = {}
                supplier = payload.pop('supplier', None)
                if isinstance(supplier, dict) and 'id' in supplier:
                    header['vendorId'] = supplier['id']
                elif isinstance(supplier, (int, float)):
                    header['vendorId'] = int(supplier)
                if 'vendorId' in payload:
                    header['vendorId'] = payload.pop('vendorId')
                for hf, pf in [('invoiceNumber', 'invoiceNumber'), ('invoiceDate', 'invoiceDate'),
                                ('dueDate', 'invoiceDueDate'), ('dueDate', 'dueDate'),
                                ('invoiceAmount', 'amount'), ('invoiceAmount', 'amountCurrency'),
                                ('invoiceAmount', 'totalAmount'),
                                ('description', 'description'), ('currencyId', 'currencyId')]:
                    if pf in payload and hf not in header:
                        header[hf] = payload.pop(pf)
                order_lines = payload.pop('orderLines', payload.pop('lines', payload.pop('invoiceLines', [])))
                payload = {"invoiceHeader": header, "orderLines": order_lines}
            # Also fix header if invoiceDueDate used instead of dueDate
            if 'invoiceHeader' in payload:
                h = payload['invoiceHeader']
                if 'invoiceDueDate' in h and 'dueDate' not in h:
                    h['dueDate'] = h.pop('invoiceDueDate')

        # --- Fix /incomingInvoice orderLine fields (uses flat IDs, not objects) ---
        if path == '/incomingInvoice' and method == 'POST' and payload:
            for line in payload.get('orderLines', []):
                if isinstance(line, dict):
                    # Convert object refs to flat IDs
                    for obj_field, id_field in [('product', 'productId'), ('department', 'departmentId'),
                                                 ('customer', 'customerId'), ('employee', 'employeeId'),
                                                 ('vatType', 'vatTypeId'), ('account', 'accountId')]:
                        val = line.pop(obj_field, None)
                        if isinstance(val, dict) and 'id' in val and id_field not in line:
                            line[id_field] = val['id']
                        elif isinstance(val, (int, float)) and id_field not in line:
                            line[id_field] = int(val)
                    # Map amount fields
                    for wrong, right in [('unitPriceExcludingVatCurrency', 'amountInclVat'),
                                          ('unitPrice', 'amountInclVat'),
                                          ('price', 'amountInclVat'),
                                          ('amount', 'amountInclVat'),
                                          ('amountExcludingVatCurrency', 'amountInclVat')]:
                        if wrong in line and right not in line:
                            line[right] = line.pop(wrong)
                    # quantity -> count
                    if 'quantity' in line and 'count' not in line:
                        line['count'] = line.pop('quantity')
                    # Strip invalid fields
                    for inv_field in ('unitPriceExcludingVatCurrency', 'unitPrice', 'price',
                                       'project', 'projectId_obj', 'supplier'):
                        line.pop(inv_field, None)

        # --- Supplier invoice fixes (for existing supplierInvoice paths) ---
        if '/supplierInvoice' in path and method in ('PUT', 'POST') and payload:
            for field in ('account', 'postings', 'expenseAccount', 'accountNumber',
                          'ledgerAccount', 'costAccount', 'debitAccount', 'creditAccount'):
                payload.pop(field, None)
            if 'lines' in payload and 'orderLines' not in payload:
                payload['orderLines'] = payload.pop('lines')
            if 'invoiceLines' in payload and 'orderLines' not in payload:
                payload['orderLines'] = payload.pop('invoiceLines')
            if 'dueDate' in payload and 'invoiceDueDate' not in payload:
                payload['invoiceDueDate'] = payload.pop('dueDate')
            if 'totalAmount' in payload and 'amount' not in payload:
                payload['amount'] = payload.pop('totalAmount')
            supplier = payload.get('supplier')
            if isinstance(supplier, (int, float)):
                payload['supplier'] = {"id": int(supplier)}

        # --- Company: strip bank-related fields ---
        if '/company' in path and method in ('PUT', 'POST') and payload:
            for field in ('bankAccountNumber', 'bankAccount', 'accountNumber', 'iban', 'bic', 'swift'):
                payload.pop(field, None)

        # --- Travel expense fixes ---
        if '/travelExpense' in path and method == 'POST' and payload:
            if 'description' in payload and 'title' not in payload:
                payload['title'] = payload.pop('description')
            if 'text' in payload and 'title' not in payload:
                payload['title'] = payload.pop('text')
            if 'name' in payload and 'title' not in payload:
                payload['title'] = payload.pop('name')
            for field in ('numberOfDays', 'days', 'duration', 'perDiem', 'perDiemRate',
                          'dailyRate', 'dailyAllowance', 'dayRate', 'diett'):
                payload.pop(field, None)
            for pdc in payload.get('perDiemCompensations', []):
                if isinstance(pdc, dict):
                    if 'numberOfDays' in pdc and 'count' not in pdc:
                        pdc['count'] = pdc.pop('numberOfDays')
                    if 'days' in pdc and 'count' not in pdc:
                        pdc['count'] = pdc.pop('days')
                    if 'dailyRate' in pdc and 'rateAmount' not in pdc:
                        pdc['rateAmount'] = pdc.pop('dailyRate')
                    if 'rate' in pdc and 'rateAmount' not in pdc:
                        pdc['rateAmount'] = pdc.pop('rate')
            for cost in payload.get('costs', []):
                if isinstance(cost, dict):
                    if 'description' in cost and 'comments' not in cost:
                        cost['comments'] = cost.pop('description')
                    if 'text' in cost and 'comments' not in cost:
                        cost['comments'] = cost.pop('text')
                    if 'name' in cost and 'comments' not in cost:
                        cost['comments'] = cost.pop('name')
                    if 'amount' in cost and 'amountCurrencyIncVat' not in cost:
                        cost['amountCurrencyIncVat'] = cost.pop('amount')
                    if 'amountNOK' in cost and 'amountCurrencyIncVat' not in cost:
                        cost['amountCurrencyIncVat'] = cost.pop('amountNOK')
            emp = payload.get('employee')
            if isinstance(emp, (int, float)):
                payload['employee'] = {"id": int(emp)}

        # --- Project fixes ---
        if '/project' in path and method == 'POST' and payload:
            # Ensure projectManager is object format
            pm = payload.get('projectManager')
            if isinstance(pm, (int, float)):
                payload['projectManager'] = {"id": int(pm)}

        # --- Redirect wrong time logging endpoints ---
        import re as _re
        if _re.match(r'/project/\d+/time', path) and method == 'POST':
            # Extract project ID from path and redirect to /timesheet/entry
            proj_id_match = _re.search(r'/project/(\d+)/time', path)
            if proj_id_match:
                proj_id = proj_id_match.group(1)
                path = '/timesheet/entry'
                if payload and 'project' not in payload:
                    payload['project'] = {"id": int(proj_id)}
                logger.info(f"Redirected /project/{proj_id}/time to /timesheet/entry")

        # --- /project/list expects array of Project objects ---
        if path == '/project/list' and method == 'POST' and payload:
            if isinstance(payload, dict):
                # Wrap single project dict into array
                payload = [payload]

        # --- Credit note path fix ---
        if _re.match(r'/invoice/\d+/:credit$', path):
            path = path.replace('/:credit', '/:createCreditNote')
            method = 'PUT'  # Must be PUT per API spec
            logger.info(f"Fixed credit note path: {path}")

        # --- Credit note: date must be query param ---
        if '/:createCreditNote' in path and method == 'PUT':
            if isinstance(payload, dict):
                for date_field in ('date', 'creditNoteDate', 'creditDate'):
                    if date_field in payload:
                        query_params['date'] = payload.pop(date_field)
                        break
                # Move other credit note query params
                for field in ('comment', 'creditNoteEmail', 'sendToCustomer', 'sendType'):
                    if field in payload:
                        query_params[field] = payload.pop(field)

        logger.info(f"After preprocessing: {method} {path}")
        if payload:
            logger.debug(f"Preprocessed payload: {json.dumps(payload, indent=2)}")

        return path, method, payload, query_params

    def _remediate_bank_account(self) -> bool:
        """Auto-remediate the 'company must register bank account' blocker.

        The fix is to find/create a ledger account (1920) and set it as a bank account
        with a valid Norwegian bank account number. This is done via /ledger/account,
        NOT via /company (which doesn't have bankAccountNumber field).
        """
        try:
            # Step 1: Check if a bank account already exists
            accounts = self.client.get("/ledger/account", params={"isBankAccount": True, "count": 50}).get("values", [])
            if any(str(item.get("bankAccountNumber") or "").strip() for item in accounts if isinstance(item, dict)):
                logger.info("Bank account already exists - blocker may be stale")
                return True  # Retry the invoice

            # Step 2: Find account 1920 (standard Norwegian bank account)
            candidates = self.client.get("/ledger/account", params={"number": 1920}).get("values", [])
            candidate = candidates[0] if candidates else None

            if candidate is None:
                # Try broader search
                candidates = self.client.get("/ledger/account", params={"numberFrom": 1900, "numberTo": 1930}).get("values", [])
                candidate = candidates[0] if candidates else None

            if candidate is None:
                logger.warning("No suitable ledger account found for bank account remediation")
                return False

            # Step 3: Generate a valid-looking Norwegian bank account number and update
            # Norwegian bank account: 11 digits, with MOD11 check digit
            import random
            for attempt in range(5):
                # Generate candidates with known bank identifier prefixes
                prefix = random.choice(["0500", "0530", "1503", "6011", "9710"])
                middle = f"{random.randint(10, 99)}"
                base = prefix + middle + f"{random.randint(1000, 9999)}"
                # Calculate MOD11 check digit
                weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
                total = sum(int(base[i]) * weights[i] for i in range(10))
                remainder = total % 11
                if remainder == 0:
                    check = 0
                elif remainder == 1:
                    continue  # Invalid, skip
                else:
                    check = 11 - remainder
                bank_number = base + str(check)

                payload = {
                    "id": candidate["id"],
                    "version": candidate.get("version"),
                    "number": candidate.get("number"),
                    "name": candidate.get("name", "Bankinnskudd"),
                    "isBankAccount": True,
                    "isInvoiceAccount": True,
                    "bankAccountNumber": bank_number,
                }
                # Preserve required fields
                for field in ["ledgerType", "isCloseable", "isApplicableForSupplierInvoice",
                              "requireReconciliation", "isInactive"]:
                    if field in candidate:
                        payload[field] = candidate[field]
                if isinstance(candidate.get("vatType"), dict):
                    payload["vatType"] = {"id": candidate["vatType"]["id"]}
                if isinstance(candidate.get("currency"), dict):
                    payload["currency"] = {"id": candidate["currency"]["id"]}

                try:
                    self.client.put(f"/ledger/account/{candidate['id']}", data=payload)
                    logger.info(f"Successfully remediated bank account on ledger account {candidate['id']}")
                    return True
                except Exception as exc:
                    error_str = str(exc).lower()
                    if "ugyldig" in error_str or "invalid" in error_str or "kontonummer" in error_str:
                        logger.info(f"Bank number {bank_number} rejected, trying next candidate")
                        continue
                    logger.warning(f"Bank account remediation failed: {exc}")
                    return False

            logger.warning("Exhausted bank account number candidates")
            return False
        except Exception as e:
            logger.warning(f"Bank account remediation error: {e}")
            return False

    def _execute_dynamic(self, method: str, path: str, payload: Dict[str, Any], query_params: Dict[str, Any] = None) -> Any:
        """Execute directly against the client."""
        logger.debug(f"Executing {method} {path} with payload: {json.dumps(payload)}")
        
        kwargs = {}
        if payload and method.upper() in ["POST", "PUT"]:
            kwargs["data"] = payload
        if query_params:
            kwargs["params"] = query_params
            
        try:
            if method.upper() == "POST":
                return self.client.post(path, **kwargs)
            elif method.upper() == "PUT":
                return self.client.put(path, **kwargs)
            elif method.upper() == "GET":
                return self.client.get(path, **kwargs)
            elif method.upper() == "DELETE":
                return self.client.delete(path, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except TripletexAPIError:
            raise
        except Exception as e:
            logger.error(f"Client execution error: {e}")
            raise

    def _auto_fix_payload(self, payload: Dict[str, Any], validation_errors: List[str]) -> Dict[str, Any]:
        """Attempt to auto-fix common payload issues."""
        fixed = (payload or {}).copy()
        
        for error in validation_errors:
            if "should be integer" in error.lower() and "got str" in error.lower():
                # Try to convert string to integer
                field_match = re.search(r"field\s+(\w+)", error.lower())
                if field_match:
                    field = field_match.group(1)
                    if field in fixed and isinstance(fixed[field], str):
                        try:
                            fixed[field] = int(fixed[field])
                            logger.info(f"Auto-converted {field} to integer")
                        except ValueError:
                            pass
            
            elif "missing required field" in error.lower():
                field_match = re.search(r"field:\s*(\w+)", error.lower())
                if field_match:
                    field = field_match.group(1)
                    if field not in fixed or fixed[field] is None:
                        # Set common defaults
                        if field in ["departmentId", "department"]:
                            fixed[field] = {"id": 1}  # Default department
                            logger.info(f"Added default departmentId")
                        elif field == "currency":
                            fixed[field] = {"id": 1}  # NOK
                        elif field in ["isActive", "isClosed", "isCustomer"]:
                            fixed[field] = True
        
        return fixed

    def _try_alternative_endpoint(self, task_prompt: str, failed_method: str, failed_path: str, error_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to find an alternative endpoint when current one fails."""
        # Get all candidates
        all_candidates = self.schema_ctx._get_candidate_endpoints(task_prompt, top_n=20)
        
        # Filter out the failed one
        alternatives = [
            c for c in all_candidates 
            if not (c.get("path") == failed_path and c.get("method") == failed_method)
        ]
        
        if not alternatives:
            return None
        
        error_message = error_response.get("message", "Unknown error")
        
        # Use LLM to select alternative
        selected = self.llm.select_alternative_endpoint(
            task_prompt, failed_path, failed_method, error_message, alternatives
        )
        
        if selected:
            return self.schema_ctx._extract_schema_info(selected)
        
        return None

    def _record_entity_from_response(self, task_prompt: str, response: Dict[str, Any], path: str):
        """Record entity creation for future reference."""
        try:
            value = response.get("value", {})
            if not isinstance(value, dict):
                return
            
            entity_id = value.get("id")
            entity_name = value.get("name") or value.get("firstName") or value.get("description", "")
            
            if entity_id:
                # Determine entity type from path
                entity_type = "unknown"
                if "/employee" in path:
                    entity_type = "employee"
                elif "/customer" in path:
                    entity_type = "customer"
                elif "/product" in path:
                    entity_type = "product"
                elif "/project" in path:
                    entity_type = "project"
                elif "/invoice" in path:
                    entity_type = "invoice"
                elif "/department" in path:
                    entity_type = "department"
                elif "/order" in path:
                    entity_type = "order"
                elif "/ledger/voucher" in path:
                    entity_type = "voucher"
                
                entity_key = f"{entity_type}:{entity_id}"
                
                # Record the entity creation
                if entity_name:
                    logger.info(f"Recorded entity: {entity_key} ({entity_name})")
                
                # Store in knowledge graph for cross-referencing
                self.knowledge.record_entity_relation(
                    task_prompt[:50],  # Task as source
                    entity_key,
                    "created"
                )
        except Exception as e:
            logger.debug(f"Error recording entity: {e}")

    # Helper methods for common lookups - these can be called by LLM if needed
    def find_customer(self, name: str = None, email: str = None, org_number: str = None) -> Optional[Dict[str, Any]]:
        """Find a customer by name, email, or org number."""
        try:
            params = {}
            if name:
                params["name"] = name
            if email:
                params["email"] = email
            if org_number:
                params["organizationNumber"] = org_number
            
            if not params:
                return None
            
            result = self.client.get("/customer", params=params)
            values = result.get("values", [])
            return values[0] if values else None
        except Exception as e:
            logger.warning(f"Customer lookup failed: {e}")
            return None

    def find_product(self, name: str = None, number: str = None) -> Optional[Dict[str, Any]]:
        """Find a product by name or number."""
        try:
            params = {}
            if name:
                params["name"] = name
            if number:
                params["number"] = number
            
            if not params:
                return None
            
            result = self.client.get("/product", params=params)
            values = result.get("values", [])
            return values[0] if values else None
        except Exception as e:
            logger.warning(f"Product lookup failed: {e}")
            return None

    def find_department(self, name: str = None) -> Optional[Dict[str, Any]]:
        """Find a department by name."""
        try:
            if not name:
                return None
            
            result = self.client.get("/department", params={"name": name})
            values = result.get("values", [])
            return values[0] if values else None
        except Exception as e:
            logger.warning(f"Department lookup failed: {e}")
            return None

    def find_invoice(self, invoice_number: str = None, customer_id: int = None) -> Optional[Dict[str, Any]]:
        """Find an invoice by number or customer."""
        try:
            params = {}
            if invoice_number:
                params["invoiceNumber"] = invoice_number
            if customer_id:
                params["customerId"] = customer_id
            
            if not params:
                return None
            
            result = self.client.get("/invoice", params=params)
            values = result.get("values", [])
            return values[0] if values else None
        except Exception as e:
            logger.warning(f"Invoice lookup failed: {e}")
            return None

    def get_default_department_id(self) -> int:
        """Get a default department ID for employee creation."""
        try:
            # Try to get any existing department
            result = self.client.get("/department", params={"from": 0, "count": 1})
            values = result.get("values", [])
            if values:
                return values[0].get("id", 1)
        except Exception as e:
            logger.warning(f"Could not get default department: {e}")
        
        return 1  # Fallback
