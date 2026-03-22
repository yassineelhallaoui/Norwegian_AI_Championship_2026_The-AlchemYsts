"""Offline tests for V3 agent pre-processing logic.
Replays every real failure scenario from GCloud logs and validates the fix.
"""
import json
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


def simulate_preprocess(path, method, payload, query_params=None):
    """Simulate the agent's pre-processing logic (mirrors agent.py exactly)."""
    logging.disable(logging.CRITICAL)
    if query_params is None:
        query_params = {}
    payload = json.loads(json.dumps(payload or {}))
    query_params = dict(query_params)

    # GET /invoice date params
    if path == '/invoice' and method == 'GET':
        if 'invoiceDateFrom' not in query_params:
            query_params['invoiceDateFrom'] = '2020-01-01'
        if 'invoiceDateTo' not in query_params:
            query_params['invoiceDateTo'] = '2026-12-31'

    # Redirect voucher paths
    if method in ('POST', 'PUT') and path in ('/voucher', '/vouchers', '/journal', '/journal-entry',
            '/journalEntry', '/journal_entry', '/accounting/journalEntry',
            '/accounting/journalEntries', '/accounting/journal'):
        path = '/ledger/voucher'

    # Ledger voucher postings fix
    if path.startswith('/ledger/voucher') and method == 'POST' and payload:
        postings = payload.get('postings', [])
        if isinstance(postings, list):
            for posting in postings:
                if isinstance(posting, dict):
                    acct = posting.get('account')
                    if isinstance(acct, (int, float)):
                        posting['account'] = {"id": int(acct)}
                    posting.pop('systemGenerated', None)
                    posting.pop('row', None)
                    posting.pop('guiRow', None)
                    posting.pop('id', None)
                    posting.pop('version', None)
                    posting.pop('type', None)

    # Invoice POST - wrap orderLines into orders
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
                        # Strip invalid fields from orderLines
                        for inv_line_field in ['project', 'projectId', 'orderIds', 'orderId']:
                            line.pop(inv_line_field, None)

    # Fix plural invoices path
    if path.startswith('/invoices/'):
        path = '/invoice/' + path[len('/invoices/'):]

    # Invoice payment - move to query params
    if '/:payment' in path and method == 'PUT' and payload:
        for field in ['paymentDate', 'paymentTypeId', 'paidAmount', 'paidAmountCurrency']:
            if field in payload:
                query_params[field] = payload.pop(field)

    # Resolve {id} placeholders in any path
    if '{id}' in path:
        actual_id = None
        for id_field in ['invoiceId', 'id', 'employeeId', 'customerId', 'projectId', 'supplierId', 'voucherId']:
            if payload and id_field in payload:
                actual_id = payload.pop(id_field)
                break
        if not actual_id:
            for id_field in ['invoiceId', 'id']:
                if id_field in query_params:
                    actual_id = query_params.pop(id_field)
                    break
        if actual_id:
            path = path.replace('{id}', str(actual_id))

    # Employee - strip invalid fields
    if '/employee' in path and method == 'POST' and payload:
        for field in ['startDate', 'endDate', 'yearlySalary', 'salary', 'monthlySalary',
                       'occupationCode', 'employmentPercentage', 'employmentDetails',
                       'employmentType', 'jobTitle', 'position', 'occupation',
                       'percentageOfFullTimeEquivalent', 'annualSalary', 'hourlyWage',
                       'workingHoursScheme', 'employmentEndReason']:
            payload.pop(field, None)
        if 'fodselsnummer' in payload and 'nationalIdentityNumber' not in payload:
            payload['nationalIdentityNumber'] = payload.pop('fodselsnummer')
        if 'fødselsnummer' in payload and 'nationalIdentityNumber' not in payload:
            payload['nationalIdentityNumber'] = payload.pop('fødselsnummer')
        user_type = payload.get('userType')
        if user_type and user_type not in ['STANDARD', 'EXTENDED', 'NO_ACCESS']:
            payload['userType'] = 'STANDARD'
        if not payload.get('userType'):
            payload['userType'] = 'STANDARD'
        # postalAddress → address
        if 'postalAddress' in payload and 'address' not in payload:
            payload['address'] = payload.pop('postalAddress')
        if 'homeAddress' in payload and 'address' not in payload:
            payload['address'] = payload.pop('homeAddress')

    # Supplier invoice - strip invalid fields
    if '/supplierInvoice' in path and method == 'POST' and payload:
        for field in ['account', 'postings', 'expenseAccount', 'accountNumber',
                       'ledgerAccount', 'costAccount', 'debitAccount', 'creditAccount']:
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

    # Invoice - strip invalid fields
    if path == '/invoice' and method == 'POST' and payload:
        for field in ['orderIds', 'orderId']:
            payload.pop(field, None)

    # Company - strip invalid fields
    if '/company' in path and method in ('PUT', 'POST') and payload:
        for field in ['bankAccountNumber', 'bankAccount', 'accountNumber', 'iban', 'bic', 'swift']:
            payload.pop(field, None)

    # Travel expense - fix field names
    if '/travelExpense' in path and method == 'POST' and payload:
        if 'description' in payload and 'title' not in payload:
            payload['title'] = payload.pop('description')
        if 'text' in payload and 'title' not in payload:
            payload['title'] = payload.pop('text')
        if 'name' in payload and 'title' not in payload:
            payload['title'] = payload.pop('name')
        for field in ['numberOfDays', 'days', 'duration', 'perDiem', 'perDiemRate',
                       'dailyRate', 'dailyAllowance', 'dayRate', 'diett']:
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

    # Redirect wrong time logging endpoints
    import re as _re
    if _re.match(r'/project/\d+/time', path) and method == 'POST':
        proj_id_match = _re.search(r'/project/(\d+)/time', path)
        if proj_id_match:
            proj_id = proj_id_match.group(1)
            path = '/timesheet/entry'
            if payload and 'project' not in payload:
                payload['project'] = {"id": int(proj_id)}

    # /project/list expects array of Project objects
    if path == '/project/list' and method == 'POST' and payload:
        if isinstance(payload, dict):
            payload = [payload]

    # Credit note path fix
    if _re.match(r'/invoice/\d+/:credit$', path):
        path = path.replace('/:credit', '/:createCreditNote')
        method = 'PUT'

    # Credit note: date must be query param
    if '/:createCreditNote' in path and method == 'PUT':
        if isinstance(payload, dict):
            for date_field in ('date', 'creditNoteDate', 'creditDate'):
                if date_field in payload:
                    query_params['date'] = payload.pop(date_field)
                    break
            for field in ('comment', 'creditNoteEmail', 'sendToCustomer', 'sendType'):
                if field in payload:
                    query_params[field] = payload.pop(field)

    # Supplier invoice: POST /supplierInvoice doesn't exist, redirect
    if path == '/supplierInvoice' and method == 'POST':
        path = '/incomingInvoice'
        if payload and 'invoiceHeader' not in payload:
            header = {}
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
            if 'totalAmount' in payload and 'invoiceAmount' not in header:
                header['invoiceAmount'] = payload.pop('totalAmount')
            order_lines = payload.pop('orderLines', payload.pop('lines', payload.pop('invoiceLines', [])))
            payload = {"invoiceHeader": header, "orderLines": order_lines}

    # Fix /incomingInvoice payload when LLM sends directly
    if path == '/incomingInvoice' and method == 'POST' and payload:
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
        if 'invoiceHeader' in payload:
            h = payload['invoiceHeader']
            if 'invoiceDueDate' in h and 'dueDate' not in h:
                h['dueDate'] = h.pop('invoiceDueDate')

    # Fix /incomingInvoice orderLine fields
    if path == '/incomingInvoice' and method == 'POST' and payload:
        for line in payload.get('orderLines', []):
            if isinstance(line, dict):
                for obj_field, id_field in [('product', 'productId'), ('department', 'departmentId'),
                                             ('customer', 'customerId'), ('employee', 'employeeId'),
                                             ('vatType', 'vatTypeId'), ('account', 'accountId')]:
                    val = line.pop(obj_field, None)
                    if isinstance(val, dict) and 'id' in val and id_field not in line:
                        line[id_field] = val['id']
                    elif isinstance(val, (int, float)) and id_field not in line:
                        line[id_field] = int(val)
                for wrong, right in [('unitPriceExcludingVatCurrency', 'amountInclVat'),
                                      ('unitPrice', 'amountInclVat'),
                                      ('price', 'amountInclVat'),
                                      ('amount', 'amountInclVat'),
                                      ('amountExcludingVatCurrency', 'amountInclVat')]:
                    if wrong in line and right not in line:
                        line[right] = line.pop(wrong)
                if 'quantity' in line and 'count' not in line:
                    line['count'] = line.pop('quantity')
                for inv_field in ('unitPriceExcludingVatCurrency', 'unitPrice', 'price',
                                   'project', 'projectId_obj', 'supplier'):
                    line.pop(inv_field, None)

    logging.disable(logging.NOTSET)
    return path, method, payload, query_params


# ============================================================================
# REAL LOG REPLAY TESTS — every scenario from the GCloud evaluation logs
# ============================================================================

print("=" * 70)
print("REAL LOG REPLAY: Testing every failure scenario from GCloud logs")
print("=" * 70)

# --------------------------------------------------------------------------
# LOG CASE A: Spanish invoice — "Crea y envía una factura al cliente Montaña SL"
# LLM generated: POST /invoice with top-level orderLines + wrong field names
# Error: 422 "orders - Listen kan ikke være tom / Kan ikke være null"
# Root cause: orders[] missing, orderLines at top level
# --------------------------------------------------------------------------
print("\n--- CASE A: Spanish invoice (orderLines at top level) ---")
print("    Log: POST /invoice -> 422 orders Listen kan ikke være tom")

path, method, payload, qp = simulate_preprocess('/invoice', 'POST', {
    "customer": {"id": 2147483647},
    "invoiceDate": "2026-03-22",
    "invoiceDueDate": "2026-04-22",
    "orderLines": [
        {
            "product": {"id": 1},
            "description": "Licencia de software",
            "quantity": 1,
            "unitPrice": 48600
        }
    ]
})
test("orders[] created (was missing)", 'orders' in payload)
test("orderLines NOT at invoice top level", 'orderLines' not in payload)
test("orderLines inside orders[0]", len(payload['orders'][0].get('orderLines', [])) == 1)
test("customer moved into order", payload['orders'][0].get('customer', {}).get('id') == 2147483647)
test("quantity -> count in line", payload['orders'][0]['orderLines'][0].get('count') == 1)
test("unitPrice -> unitPriceExcludingVatCurrency", payload['orders'][0]['orderLines'][0].get('unitPriceExcludingVatCurrency') == 48600)

# --------------------------------------------------------------------------
# LOG CASE B: German employee — "Erstellen Sie den Mitarbeiter in Tripletex"
# LLM generated: POST /employee with startDate, yearlySalary, occupationCode, etc.
# Error: 422 "employmentDetails - Feltet eksisterer ikke"
#        422 "occupationCode - Feltet eksisterer ikke"
#        422 "yearlySalary - Feltet eksisterer ikke"
#        422 "employmentPercentage - Feltet eksisterer ikke"
#        422 "startDate - Feltet eksisterer ikke"
# --------------------------------------------------------------------------
print("\n--- CASE B: German employee (invalid sub-resource fields) ---")
print("    Log: POST /employee -> 422 employmentDetails/occupationCode/yearlySalary/startDate Feltet eksisterer ikke")

path, method, payload, qp = simulate_preprocess('/employee', 'POST', {
    "firstName": "Hans",
    "lastName": "Müller",
    "email": "hans.mueller@example.de",
    "dateOfBirth": "1985-06-15",
    "fodselsnummer": "15068512345",
    "department": {"id": 1},
    "employeeNumber": "EMP042",
    "userType": "employee",
    "employmentDetails": {
        "startDate": "2026-01-01",
        "percentage": 100
    },
    "startDate": "2026-01-01",
    "yearlySalary": 650000,
    "occupationCode": "2511",
    "employmentPercentage": 100,
    "salary": 54167,
    "position": "Developer",
    "jobTitle": "Senior Engineer",
    "annualSalary": 650000
})
test("firstName preserved", payload.get('firstName') == 'Hans')
test("lastName preserved", payload.get('lastName') == 'Müller')
test("email preserved", payload.get('email') == 'hans.mueller@example.de')
test("department preserved", payload.get('department') == {"id": 1})
test("employeeNumber preserved", payload.get('employeeNumber') == 'EMP042')
test("fodselsnummer -> nationalIdentityNumber", payload.get('nationalIdentityNumber') == '15068512345')
test("userType fixed to STANDARD", payload.get('userType') == 'STANDARD')
test("employmentDetails stripped", 'employmentDetails' not in payload)
test("startDate stripped", 'startDate' not in payload)
test("yearlySalary stripped", 'yearlySalary' not in payload)
test("occupationCode stripped", 'occupationCode' not in payload)
test("employmentPercentage stripped", 'employmentPercentage' not in payload)
test("salary stripped", 'salary' not in payload)
test("position stripped", 'position' not in payload)
test("jobTitle stripped", 'jobTitle' not in payload)
test("annualSalary stripped", 'annualSalary' not in payload)

# Extra employee tests: postalAddress→address, missing userType
print("\n--- CASE B2: Employee with postalAddress and no userType ---")
path, method, payload, qp = simulate_preprocess('/employee', 'POST', {
    "firstName": "Ola",
    "lastName": "Nordmann",
    "email": "ola@example.no",
    "postalAddress": {"addressLine1": "Storgata 1", "city": "Oslo", "zipCode": "0001"},
    "department": {"id": 1}
})
test("postalAddress -> address", 'postalAddress' not in payload and payload.get('address') == {"addressLine1": "Storgata 1", "city": "Oslo", "zipCode": "0001"})
test("userType defaults to STANDARD", payload.get('userType') == 'STANDARD')

path, method, payload, qp = simulate_preprocess('/employee', 'POST', {
    "firstName": "Kari",
    "lastName": "Hansen",
    "homeAddress": {"addressLine1": "Lillegata 5"},
    "department": {"id": 2}
})
test("homeAddress -> address", 'homeAddress' not in payload and payload.get('address') == {"addressLine1": "Lillegata 5"})

# --------------------------------------------------------------------------
# LOG CASE C: German month-end — "Führen Sie den Monatsabschluss"
# LLM tried: POST /voucher (404), /vouchers (404), /journal-entry (404), /journal (404)
# Never found /ledger/voucher
# --------------------------------------------------------------------------
print("\n--- CASE C: Month-end journal entries (wrong voucher paths) ---")
print("    Log: POST /voucher -> 404, /vouchers -> 404, /journal-entry -> 404, /journal -> 404")

for wrong_path, label in [
    ('/voucher', '/voucher'),
    ('/vouchers', '/vouchers'),
    ('/journal', '/journal'),
    ('/journal-entry', '/journal-entry'),
    ('/journalEntry', '/journalEntry'),
    ('/accounting/journalEntry', '/accounting/journalEntry'),
    ('/accounting/journalEntries', '/accounting/journalEntries'),
    ('/accounting/journal', '/accounting/journal'),
]:
    path, _, _, _ = simulate_preprocess(wrong_path, 'POST', {
        "date": "2026-03-22",
        "description": "Monatsabschluss",
        "postings": [
            {"account": 1720, "amount": 2200, "description": "Rechnungsabgrenzung"},
            {"account": 6020, "amount": -2200, "description": "Aufwand"}
        ]
    })
    test(f"{label} redirected to /ledger/voucher", path == '/ledger/voucher', f"got {path}")

# --------------------------------------------------------------------------
# LOG CASE D: Voucher postings stripped (old bug) — causing
# "Et bilag kan ikke registreres uten posteringer" (voucher can't be registered without postings)
# AND "Posteringene på rad 0 er systemgenererte" (row 0 postings are system-generated)
# --------------------------------------------------------------------------
print("\n--- CASE D: Voucher postings must be kept + cleaned ---")
print("    Log: POST /ledger/voucher -> 422 Et bilag kan ikke registreres uten posteringer")

path, method, payload, qp = simulate_preprocess('/ledger/voucher', 'POST', {
    "date": "2026-03-22",
    "description": "Depreciation entry",
    "postings": [
        {"account": 6020, "amount": 4861.67, "description": "Abschreibung", "row": 0, "systemGenerated": False, "id": 1, "version": 0},
        {"account": 1090, "amount": -4861.67, "description": "Anlage", "row": 1, "systemGenerated": False, "id": 2, "version": 0}
    ]
})
test("postings NOT stripped (was the old bug)", len(payload.get('postings', [])) == 2)
test("debit posting account fixed", payload['postings'][0]['account'] == {"id": 6020})
test("credit posting account fixed", payload['postings'][1]['account'] == {"id": 1090})
test("debit amount correct", payload['postings'][0]['amount'] == 4861.67)
test("credit amount correct", payload['postings'][1]['amount'] == -4861.67)
test("row stripped from posting[0]", 'row' not in payload['postings'][0])
test("systemGenerated stripped from posting[0]", 'systemGenerated' not in payload['postings'][0])
test("id stripped from posting[0]", 'id' not in payload['postings'][0])
test("version stripped from posting[0]", 'version' not in payload['postings'][0])
test("description kept", payload['postings'][0].get('description') == 'Abschreibung')

# also test redirect + postings fix together
print("\n    Subtest: redirect + postings fix combined")
path, method, payload, qp = simulate_preprocess('/voucher', 'POST', {
    "date": "2026-03-01",
    "description": "Gehaltsrückstellung",
    "postings": [
        {"account": 5000, "amount": 50000, "row": 0},
        {"account": 2900, "amount": -50000, "row": 1}
    ]
})
test("path redirected from /voucher", path == '/ledger/voucher')
test("postings preserved after redirect", len(payload.get('postings', [])) == 2)
test("account fixed to {id: 5000}", payload['postings'][0]['account'] == {"id": 5000})
test("account fixed to {id: 2900}", payload['postings'][1]['account'] == {"id": 2900})
test("row stripped after redirect", 'row' not in payload['postings'][0])

# --------------------------------------------------------------------------
# LOG CASE E: Invoice payment — PUT /invoice/{id}/:payment
# LLM put paymentDate, paymentTypeId, paidAmount in body instead of query params
# Error: 422 "paidAmount/paymentDate/paymentTypeId - Kan ikke være null"
# --------------------------------------------------------------------------
print("\n--- CASE E: Invoice payment (body -> query params) ---")
print("    Log: PUT /invoice/2147676325/:payment -> 422 paidAmount/paymentDate/paymentTypeId Kan ikke være null")

path, method, payload, qp = simulate_preprocess('/invoice/2147676325/:payment', 'PUT', {
    "paymentDate": "2026-03-22",
    "paymentTypeId": 1,
    "paidAmount": 8900,
    "paidAmountCurrency": 8900
})
test("paymentDate moved to query params", qp.get('paymentDate') == '2026-03-22')
test("paymentTypeId moved to query params", qp.get('paymentTypeId') == 1)
test("paidAmount moved to query params", qp.get('paidAmount') == 8900)
test("paidAmountCurrency moved to query params", qp.get('paidAmountCurrency') == 8900)
test("payload is empty after move", len(payload) == 0, f"still has: {list(payload.keys())}")
test("path unchanged", path == '/invoice/2147676325/:payment')

# --------------------------------------------------------------------------
# LOG CASE F: Invoice payment with {id} placeholder
# LLM used literal {id} in path
# --------------------------------------------------------------------------
print("\n--- CASE F: Invoice payment with {id} placeholder ---")
print("    Log: PUT /invoice/{id}/:payment with invoiceId in body")

path, method, payload, qp = simulate_preprocess('/invoice/{id}/:payment', 'PUT', {
    "invoiceId": 2147676325,
    "paymentDate": "2026-03-22",
    "paymentTypeId": 1,
    "paidAmount": 5000
})
test("{id} resolved to actual ID", path == '/invoice/2147676325/:payment', f"got {path}")
test("invoiceId removed from payload", 'invoiceId' not in payload)
test("payment fields in query params", all(k in qp for k in ['paymentDate', 'paymentTypeId', 'paidAmount']))

# --------------------------------------------------------------------------
# LOG CASE G: Invoice payment with wrong HTTP method and plural path
# LLM tried: DELETE /invoice/{id}/:payment (405)
#            POST /invoice/{id}/:payment (405)
#            PUT /invoices/{id}/:payment (404 - plural)
# --------------------------------------------------------------------------
print("\n--- CASE G: Invoice payment with plural /invoices/ path ---")
print("    Log: PUT /invoices/2147677971/:payment -> 404")

path, method, payload, qp = simulate_preprocess('/invoices/2147677971/:payment', 'PUT', {
    "paymentDate": "2026-03-22",
    "paymentTypeId": 1,
    "paidAmount": 5000
})
test("plural /invoices/ fixed to /invoice/", path == '/invoice/2147677971/:payment', f"got {path}")
test("payment fields moved to query params", 'paymentDate' in qp)

# --------------------------------------------------------------------------
# LOG CASE H: Supplier invoice — German "Lieferantenrechnung"
# POST /supplierInvoice doesn't exist -> redirect to POST /incomingInvoice
# with payload transformation to {invoiceHeader: {...}, orderLines: [...]}
# --------------------------------------------------------------------------
print("\n--- CASE H: German supplier invoice (redirect to /incomingInvoice) ---")
print("    Log: POST /supplierInvoice -> redirect to POST /incomingInvoice")

path, method, payload, qp = simulate_preprocess('/supplierInvoice', 'POST', {
    "invoiceNumber": "INV-2026-8172",
    "invoiceDate": "2026-03-15",
    "dueDate": "2026-04-15",
    "supplier": {"id": 789},
    "account": 6500,
    "expenseAccount": 6500,
    "accountNumber": "6500",
    "amount": 19100,
    "amountCurrency": 19100,
    "postings": [
        {"account": {"id": 6500}, "amount": 15280},
        {"account": {"id": 2710}, "amount": 3820}
    ],
    "orderLines": [
        {"description": "Bürodienstleistungen", "count": 1, "unitPriceExcludingVatCurrency": 15280}
    ]
})
test("redirected to /incomingInvoice", path == '/incomingInvoice')
test("has invoiceHeader", 'invoiceHeader' in payload)
test("vendorId from supplier", payload.get('invoiceHeader', {}).get('vendorId') == 789)
test("invoiceNumber in header", payload.get('invoiceHeader', {}).get('invoiceNumber') == 'INV-2026-8172')
test("invoiceDate in header", payload.get('invoiceHeader', {}).get('invoiceDate') == '2026-03-15')
test("dueDate in header", payload.get('invoiceHeader', {}).get('dueDate') == '2026-04-15')
test("invoiceAmount in header", payload.get('invoiceHeader', {}).get('invoiceAmount') == 19100)
test("orderLines preserved", len(payload.get('orderLines', [])) == 1)
# account, postings etc should NOT be in transformed payload
test("no account in payload", 'account' not in payload)
test("no postings in payload", 'postings' not in payload)

# Also test supplier as plain int
print("\n    Subtest: supplier as plain integer")
path, method, payload, qp = simulate_preprocess('/supplierInvoice', 'POST', {
    "invoiceNumber": "INV-2026-2118",
    "invoiceDate": "2026-03-20",
    "supplier": 456,
    "totalAmount": 70400,
    "account": 6590,
    "lines": [
        {"description": "Bürodienstleistungen", "quantity": 1, "unitPrice": 56320}
    ]
})
test("redirected to /incomingInvoice", path == '/incomingInvoice')
test("supplier int -> vendorId", payload.get('invoiceHeader', {}).get('vendorId') == 456)
test("totalAmount -> invoiceAmount", payload.get('invoiceHeader', {}).get('invoiceAmount') == 70400)
test("lines -> orderLines", 'orderLines' in payload)

# --------------------------------------------------------------------------
# LOG CASE I: GET /invoice missing required date params
# Error: 460 "query.invoiceDateFrom Required query parameter must be provided"
# Agent kept hitting this on every turn, wasting time
# --------------------------------------------------------------------------
print("\n--- CASE I: GET /invoice missing date params ---")
print("    Log: GET /invoice -> 460 invoiceDateFrom Required query parameter must be provided")

path, method, payload, qp = simulate_preprocess('/invoice', 'GET', {})
test("invoiceDateFrom auto-added", qp.get('invoiceDateFrom') == '2020-01-01')
test("invoiceDateTo auto-added", qp.get('invoiceDateTo') == '2026-12-31')

# With partial params
path, method, payload, qp = simulate_preprocess('/invoice', 'GET', {},
    query_params={"invoiceDateFrom": "2026-01-01", "customerId": 123})
test("existing invoiceDateFrom preserved", qp.get('invoiceDateFrom') == '2026-01-01')
test("invoiceDateTo auto-added when missing", qp.get('invoiceDateTo') == '2026-12-31')
test("customerId preserved", qp.get('customerId') == 123)

# --------------------------------------------------------------------------
# LOG CASE J: French order-to-invoice — "Créez une commande...Convertissez en facture"
# LLM tried: POST /invoice with orderIds field
# Error: 422 "orderIds - Feltet eksisterer ikke i objektet"
# --------------------------------------------------------------------------
print("\n--- CASE J: French order-to-invoice (orderIds field) ---")
print("    Log: POST /invoice -> 422 orderIds Feltet eksisterer ikke")

path, method, payload, qp = simulate_preprocess('/invoice', 'POST', {
    "invoiceDate": "2026-03-22",
    "invoiceDueDate": "2026-04-22",
    "orderIds": [2147483647],
    "orderId": 2147483647,
    "orders": [{"id": 2147483647}]
})
test("orderIds stripped", 'orderIds' not in payload)
test("orderId stripped", 'orderId' not in payload)
test("orders preserved (valid field)", 'orders' in payload)

# --------------------------------------------------------------------------
# LOG CASE K: Bank reconciliation looping
# Agent kept doing GET /invoice every turn without date params
# Max cascading turns reached
# --------------------------------------------------------------------------
print("\n--- CASE K: Bank reconciliation GET /invoice loop ---")
print("    Log: GET /invoice -> 460 (8 times, max turns reached)")

# Simulate 3 consecutive GET /invoice calls - all should have date params
for i in range(3):
    path, method, payload, qp = simulate_preprocess('/invoice', 'GET', {},
        query_params={"customerName": f"Customer {i}"})
    test(f"GET /invoice iteration {i+1}: dates auto-added",
         'invoiceDateFrom' in qp and 'invoiceDateTo' in qp)

# --------------------------------------------------------------------------
# LOG CASE L: Project creation — projectManager.id invalid
# Error: 422 "projectManager.id - ID-en må referere til et gyldig object"
# (Can't fix invalid ID, but make sure valid payloads pass through)
# --------------------------------------------------------------------------
print("\n--- CASE L: Project creation (valid payload passthrough) ---")
print("    Log: POST /project -> 422 projectManager.id invalid (data issue, not preprocessing)")

path, method, payload, qp = simulate_preprocess('/project', 'POST', {
    "name": "Website Development",
    "projectManager": {"id": 1},
    "customer": {"id": 123},
    "startDate": "2026-03-01"
})
test("project payload passes through unchanged", payload.get('name') == 'Website Development')
test("projectManager preserved", payload.get('projectManager') == {"id": 1})
test("customer preserved", payload.get('customer') == {"id": 123})
test("startDate preserved (valid for project)", payload.get('startDate') == '2026-03-01')

# --------------------------------------------------------------------------
# LOG CASE M2: Travel expense with wrong field names
# Log: POST /travelExpense -> 422 numberOfDays/description/text Feltet eksisterer ikke
# Root cause: LLM uses wrong field names for travel expense
# --------------------------------------------------------------------------
print("\n--- CASE M2: Travel expense wrong field names ---")
print("    Log: POST /travelExpense -> 422 numberOfDays/description/text Feltet eksisterer ikke")

# Test 1: Top-level description -> title
path, method, payload, qp = simulate_preprocess('/travelExpense', 'POST', {
    "employee": {"id": 42},
    "description": "Kundebesøk Stavanger for Arne Vik",
    "numberOfDays": 2,
    "perDiem": 800,
    "costs": [
        {"description": "Flybillett", "amount": 3600},
        {"description": "Taxi", "amount": 550}
    ],
    "perDiemCompensations": [
        {"numberOfDays": 2, "dailyRate": 800}
    ]
})
test("description -> title", payload.get('title') == "Kundebesøk Stavanger for Arne Vik")
test("'description' removed from top", 'description' not in payload)
test("numberOfDays stripped from top", 'numberOfDays' not in payload)
test("perDiem stripped from top", 'perDiem' not in payload)
test("cost[0].description -> comments", payload['costs'][0].get('comments') == "Flybillett")
test("cost[0].amount -> amountCurrencyIncVat", payload['costs'][0].get('amountCurrencyIncVat') == 3600)
test("cost[1].comments set", payload['costs'][1].get('comments') == "Taxi")
test("pdc[0].numberOfDays -> count", payload['perDiemCompensations'][0].get('count') == 2)
test("pdc[0].dailyRate -> rateAmount", payload['perDiemCompensations'][0].get('rateAmount') == 800)

# Test 2: text -> title
path, method, payload, qp = simulate_preprocess('/travelExpense', 'POST', {
    "employee": 55,
    "text": "Kundenbesuch Oslo",
    "costs": [{"text": "Flugticket", "amountNOK": 4150}]
})
test("text -> title", payload.get('title') == "Kundenbesuch Oslo")
test("employee int -> {id}", payload.get('employee') == {"id": 55})
test("cost text -> comments", payload['costs'][0].get('comments') == "Flugticket")
test("cost amountNOK -> amountCurrencyIncVat", payload['costs'][0].get('amountCurrencyIncVat') == 4150)

# Test 3: name -> title, days -> count
path, method, payload, qp = simulate_preprocess('/travelExpense', 'POST', {
    "employee": {"id": 10},
    "name": "Business trip",
    "perDiemCompensations": [{"days": 3, "rate": 800}],
    "diett": 800,
    "dailyAllowance": 800,
    "duration": 3
})
test("name -> title", payload.get('title') == "Business trip")
test("diett stripped", 'diett' not in payload)
test("dailyAllowance stripped", 'dailyAllowance' not in payload)
test("duration stripped", 'duration' not in payload)
test("pdc days -> count", payload['perDiemCompensations'][0].get('count') == 3)
test("pdc rate -> rateAmount", payload['perDiemCompensations'][0].get('rateAmount') == 800)

# --------------------------------------------------------------------------
# LOG CASE M3: Voucher with guiRow and type fields (system-generated error)
# Log: POST /ledger/voucher -> 422 "Posteringene på rad 0 (guiRow 0) er systemgenererte"
# Root cause: guiRow and type fields mark postings as system-generated
# --------------------------------------------------------------------------
print("\n--- CASE M3: Voucher with guiRow/type (system-generated error) ---")
print("    Log: POST /ledger/voucher -> 422 systemgenererte")

path, method, payload, qp = simulate_preprocess('/ledger/voucher', 'POST', {
    "date": "2025-12-31",
    "description": "Årleg avskriving IT-utstyr",
    "postings": [
        {"account": 6010, "amount": 94540, "description": "Debit", "row": 0, "guiRow": 0, "type": "DEBIT", "systemGenerated": False},
        {"account": 1209, "amount": -94540, "description": "Credit", "row": 1, "guiRow": 1, "type": "CREDIT"}
    ]
})
test("guiRow stripped from posting[0]", 'guiRow' not in payload['postings'][0])
test("guiRow stripped from posting[1]", 'guiRow' not in payload['postings'][1])
test("type stripped from posting[0]", 'type' not in payload['postings'][0])
test("type stripped from posting[1]", 'type' not in payload['postings'][1])
test("row stripped from posting[0]", 'row' not in payload['postings'][0])
test("systemGenerated stripped", 'systemGenerated' not in payload['postings'][0])
test("account fixed to {id: 6010}", payload['postings'][0].get('account') == {"id": 6010})
test("amount preserved", payload['postings'][0].get('amount') == 94540)

# --------------------------------------------------------------------------
# LOG CASE M: Invoice orderLine with "project" field
# Log: POST /invoice -> 422 project Feltet eksisterer ikke i objektet
# Root cause: "project" is not a valid field on invoice order lines
# --------------------------------------------------------------------------
print("\n--- CASE M: Invoice orderLine with invalid 'project' field ---")
print("    Log: POST /invoice -> 422 project Feltet eksisterer ikke")

path, method, payload, qp = simulate_preprocess('/invoice', 'POST', {
    "invoiceDate": "2026-03-22",
    "orders": [{
        "customer": {"id": 123},
        "orderDate": "2026-03-22",
        "deliveryDate": "2026-03-22",
        "orderLines": [{
            "product": {"id": 1},
            "description": "Consulting",
            "count": 10,
            "unitPriceExcludingVatCurrency": 1000,
            "project": {"id": 402070991}
        }]
    }]
})
line = payload['orders'][0]['orderLines'][0]
test("project stripped from orderLine", 'project' not in line)
test("projectId stripped from orderLine", 'projectId' not in line)
test("product preserved", line.get('product') == {"id": 1})
test("description preserved", line.get('description') == "Consulting")
test("count preserved", line.get('count') == 10)

# Also test with top-level orderLines that get wrapped
path, method, payload, qp = simulate_preprocess('/invoice', 'POST', {
    "invoiceDate": "2026-03-22",
    "customer": {"id": 123},
    "orderLines": [{
        "description": "Item",
        "quantity": 5,
        "unitPrice": 200,
        "project": {"id": 999}
    }]
})
line = payload['orders'][0]['orderLines'][0]
test("project stripped after wrapping", 'project' not in line)
test("quantity->count after wrapping", line.get('count') == 5)
test("unitPrice->unitPriceExcludingVatCurrency after wrapping",
     line.get('unitPriceExcludingVatCurrency') == 200)

# --------------------------------------------------------------------------
# LOG CASE N: PUT /company with bankAccountNumber
# Log: PUT /company -> 422 bankAccountNumber Feltet eksisterer ikke
# Root cause: bankAccountNumber is not a valid field on Company
# --------------------------------------------------------------------------
print("\n--- CASE N: PUT /company with invalid bankAccountNumber ---")
print("    Log: PUT /company -> 422 bankAccountNumber Feltet eksisterer ikke")

path, method, payload, qp = simulate_preprocess('/company', 'PUT', {
    "name": "Test Company AS",
    "bankAccountNumber": "1234.56.78901",
    "organizationNumber": "912345678"
})
test("bankAccountNumber stripped", 'bankAccountNumber' not in payload)
test("name preserved", payload.get('name') == 'Test Company AS')
test("organizationNumber preserved", payload.get('organizationNumber') == '912345678')

# Also test other invalid bank fields
path, method, payload, qp = simulate_preprocess('/company', 'PUT', {
    "name": "Test",
    "bankAccount": "123",
    "iban": "NO123456",
    "bic": "SPSONO22"
})
test("bankAccount stripped", 'bankAccount' not in payload)
test("iban stripped", 'iban' not in payload)
test("bic stripped", 'bic' not in payload)

# --------------------------------------------------------------------------
# LOG CASE O: {id} resolution from query_params (endpoint switch scenario)
# Log: PUT /invoice/{id}/:payment -> 404 (LLM switched to this after 404 on real path)
# The LLM sometimes puts the ID in query_params instead of path
# --------------------------------------------------------------------------
print("\n--- CASE O: {id} resolution from query_params ---")
print("    Log: PUT /invoice/{id}/:payment -> 404 after endpoint switch")

path, method, payload, qp = simulate_preprocess('/invoice/{id}/:payment', 'PUT',
    {"paymentDate": "2026-03-22", "paymentTypeId": 1, "paidAmount": 5000},
    query_params={"invoiceId": 2147686451})
test("{id} resolved from query_params", '/invoice/2147686451/:payment' in path)
test("payment fields in query params", 'paymentDate' in qp)

# --------------------------------------------------------------------------
# LOG CASE P: Time logging wrong endpoint /project/{id}/time
# --------------------------------------------------------------------------
print("\n--- CASE P: Time logging redirect /project/{id}/time -> /timesheet/entry ---")
path, method, payload, qp = simulate_preprocess('/project/402072857/time', 'POST', {
    "employee": {"id": 10},
    "hours": 73,
    "date": "2026-03-22",
    "activity": {"id": 1}
})
test("redirected to /timesheet/entry", path == '/timesheet/entry')
test("project ID injected", payload.get('project') == {"id": 402072857})
test("employee preserved", payload.get('employee') == {"id": 10})
test("hours preserved", payload.get('hours') == 73)

# Already has project in payload - should not overwrite
path, method, payload, qp = simulate_preprocess('/project/999/time', 'POST', {
    "employee": {"id": 5},
    "project": {"id": 123},
    "hours": 8
})
test("redirected with existing project", path == '/timesheet/entry')
test("existing project preserved", payload.get('project') == {"id": 123})

# --------------------------------------------------------------------------
# LOG CASE Q: /project/list expects array of Project objects
# --------------------------------------------------------------------------
print("\n--- CASE Q: /project/list expects array body ---")
path, method, payload, qp = simulate_preprocess('/project/list', 'POST', {
    "name": "Test Project", "startDate": "2026-01-01"
})
test("payload is array", isinstance(payload, list))
test("payload contains project dict", payload == [{"name": "Test Project", "startDate": "2026-01-01"}])

# --------------------------------------------------------------------------
# LOG CASE R: Credit note /:credit -> /:createCreditNote
# --------------------------------------------------------------------------
print("\n--- CASE R: Credit note path fix ---")
path, method, payload, qp = simulate_preprocess('/invoice/2147688799/:credit', 'PUT', {
    "creditNoteDate": "2026-03-25", "comment": "Full reversal"
})
test("/:credit -> /:createCreditNote", path == '/invoice/2147688799/:createCreditNote')
test("method forced to PUT", method == 'PUT')
test("date moved to query param", qp.get('date') == '2026-03-25')
test("comment moved to query param", qp.get('comment') == 'Full reversal')

path, method, payload, qp = simulate_preprocess('/invoice/123/:credit', 'POST', {
    "date": "2026-03-20"
})
test("POST /:credit fixed to PUT", method == 'PUT')
test("path fixed", path == '/invoice/123/:createCreditNote')
test("date in query params", qp.get('date') == '2026-03-20')

# --------------------------------------------------------------------------
# LOG CASE S: Supplier invoice POST redirect to /incomingInvoice
# --------------------------------------------------------------------------
print("\n--- CASE S: POST /supplierInvoice -> /incomingInvoice ---")
path, method, payload, qp = simulate_preprocess('/supplierInvoice', 'POST', {
    "supplier": {"id": 456},
    "invoiceNumber": "INV-2026-001",
    "invoiceDate": "2026-03-22",
    "dueDate": "2026-04-22",
    "amount": 50000,
    "description": "Consulting services",
    "orderLines": [{"product": {"id": 1}, "count": 1}]
})
test("redirected to /incomingInvoice", path == '/incomingInvoice')
test("has invoiceHeader", 'invoiceHeader' in payload)
test("vendorId from supplier", payload['invoiceHeader'].get('vendorId') == 456)
test("invoiceNumber in header", payload['invoiceHeader'].get('invoiceNumber') == 'INV-2026-001')
test("invoiceDate in header", payload['invoiceHeader'].get('invoiceDate') == '2026-03-22')
test("dueDate in header", payload['invoiceHeader'].get('dueDate') == '2026-04-22')
test("invoiceAmount in header", payload['invoiceHeader'].get('invoiceAmount') == 50000)
test("orderLines preserved", len(payload.get('orderLines', [])) == 1)

# --------------------------------------------------------------------------
# Knowledge graph validation
# --------------------------------------------------------------------------
print("\n--- Knowledge Graph Rules ---")
from core.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
test(f"loaded {len(kg.rules)} rules (>= 11 expected)", len(kg.rules) >= 11)

# Test rule retrieval for each task type we've seen
task_queries = [
    ("Create invoice for customer", "invoice"),
    ("Create journal entry voucher", "voucher"),
    ("Create employee with salary", "employee"),
    ("Register invoice payment", "payment"),
    ("Supplier invoice", "supplier"),
    ("Erstellen Sie den Mitarbeiter", "employee (German)"),
    ("Créez une facture", "invoice (French)"),
    ("Führen Sie den Monatsabschluss durch", "voucher (German)"),
    ("Registre um pagamento", "payment (Portuguese)"),
    ("Registrer ei reiserekning", "travel expense (Norwegian)"),
    ("Reisekostenabrechnung", "travel expense (German)"),
    ("bankkontonummer registrert", "bank account blocker"),
]
for query, label in task_queries:
    rules = kg.get_applicable_rules(query)
    test(f"rules found for '{label}'", len(rules) > 0, f"0 rules for: {query}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 70)
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
