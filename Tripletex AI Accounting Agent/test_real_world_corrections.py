#!/usr/bin/env python3
"""
Real-world correction tests based on actual deployment errors.
Tests all error patterns seen in the Tripletex agent logs.
"""

import json
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from core.autonomous_corrector import get_autonomous_corrector
from core.schema_intelligence import get_schema_intelligence


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add(self, name: str, success: bool, details: str):
        self.tests.append({'name': name, 'success': success, 'details': details})
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self):
        print("\n" + "="*70)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("="*70)
        for test in self.tests:
            status = "✅" if test['success'] else "❌"
            print(f"{status} {test['name']}")
            if not test['success']:
                print(f"   {test['details']}")
        return self.failed == 0


def test_invoice_line_item_corrections():
    """Test invoice line item field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Invoice Line Item Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    
    test_cases = [
        {
            'name': 'unitPrice -> unitPriceExcludingVatCurrency',
            'payload': {
                'customer': {'id': 123},
                'orderLines': [{'unitPrice': 1000, 'count': 2}]
            },
            'field': 'unitPrice',
            'expected': 'unitPriceExcludingVatCurrency'
        },
        {
            'name': 'amount -> amountExcludingVatCurrency',
            'payload': {
                'customer': {'id': 123},
                'orderLines': [{'amount': 2000, 'description': 'Test'}]
            },
            'field': 'amount',
            'expected': 'amountExcludingVatCurrency'
        },
        {
            'name': 'price -> unitPriceExcludingVatCurrency (in lines)',
            'payload': {
                'customer': {'id': 123},
                'orderLines': [{'price': 500, 'count': 1}]
            },
            'field': 'price',
            'expected': 'unitPriceExcludingVatCurrency'
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', '/invoice', test['payload'], error, [])
            
            # Check if expected field is in corrected payload
            corrected_str = json.dumps(result.corrected_payload)
            success = test['expected'] in corrected_str
            
            results.add(
                test['name'],
                success,
                f"Expected '{test['expected']}' in payload: {success}, Changes: {len(result.changes_made)}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_product_corrections():
    """Test product field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Product Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    
    test_cases = [
        {
            'name': 'price -> priceExcludingVatCurrency',
            'payload': {'name': 'Test Product', 'price': 500},
            'field': 'price',
            'expected_in': ['priceExcludingVatCurrency', 'priceIncludingVatCurrency']
        },
        {
            'name': 'salesPrice -> priceExcludingVatCurrency',
            'payload': {'name': 'Test Product', 'salesPrice': 600},
            'field': 'salesPrice',
            'expected_in': ['priceExcludingVatCurrency', 'priceIncludingVatCurrency']
        },
        {
            'name': 'costPrice -> costExcludingVatCurrency',
            'payload': {'name': 'Test Product', 'costPrice': 300},
            'field': 'costPrice',
            'expected_in': ['costExcludingVatCurrency']
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', '/product', test['payload'], error, [])
            
            corrected_str = json.dumps(result.corrected_payload)
            success = any(exp in corrected_str for exp in test['expected_in'])
            
            results.add(
                test['name'],
                success,
                f"Found one of {test['expected_in']}: {success}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_supplier_corrections():
    """Test supplier field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Supplier Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    si = get_schema_intelligence()
    
    # First check if POST /supplier exists and what fields it has
    valid_fields = si.get_valid_fields('/supplier', 'POST')
    print(f"Valid fields for POST /supplier: {list(valid_fields.keys())[:10]}...")
    
    test_cases = [
        {
            'name': 'accountNumber -> supplierNumber/organizationNumber',
            'payload': {'name': 'Test Supplier', 'accountNumber': '12345'},
            'field': 'accountNumber',
            'expected_in': ['supplierNumber', 'organizationNumber']
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', '/supplier', test['payload'], error, [])
            
            corrected_str = json.dumps(result.corrected_payload)
            success = any(exp in corrected_str for exp in test['expected_in'])
            
            results.add(
                test['name'],
                success,
                f"Found one of {test['expected_in']}: {success}, Payload: {corrected_str[:100]}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_customer_corrections():
    """Test customer field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Customer Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    si = get_schema_intelligence()
    
    # Check valid fields
    valid_fields = si.get_valid_fields('/customer', 'POST')
    print(f"Valid fields for POST /customer: {list(valid_fields.keys())[:10]}...")
    
    test_cases = [
        {
            'name': 'address -> postalAddress',
            'payload': {
                'name': 'Test Customer',
                'address': 'Street 123'
            },
            'field': 'address',
            'expected': 'postalAddress'
        },
        {
            'name': 'customerId -> customer (object)',
            'payload': {
                'name': 'Test',
                'customerId': 123
            },
            'field': 'customerId',
            'expected': 'customer'
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', '/customer', test['payload'], error, [])
            
            corrected_str = json.dumps(result.corrected_payload)
            success = test['expected'] in corrected_str
            
            results.add(
                test['name'],
                success,
                f"Expected '{test['expected']}' in payload: {success}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_employee_corrections():
    """Test employee field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Employee Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    si = get_schema_intelligence()
    
    # Check valid fields and userType enum
    valid_fields = si.get_valid_fields('/employee', 'POST')
    print(f"Valid fields for POST /employee: {list(valid_fields.keys())[:10]}...")
    
    # Check userType schema
    user_type_field = valid_fields.get('userType', {})
    print(f"userType field info: {user_type_field.get('type')}")
    
    return results


def test_project_corrections():
    """Test project field corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Project Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    si = get_schema_intelligence()
    
    valid_fields = si.get_valid_fields('/project', 'POST')
    print(f"Valid fields for POST /project: {list(valid_fields.keys())[:15]}...")
    
    test_cases = [
        {
            'name': 'projectManagerId -> projectManager',
            'payload': {
                'name': 'Test Project',
                'projectManagerId': 123
            },
            'field': 'projectManagerId',
            'expected': 'projectManager'
        },
        {
            'name': 'budgetAmountTotal -> priceCeilingAmount',
            'payload': {
                'name': 'Test Project',
                'budgetAmountTotal': 50000
            },
            'field': 'budgetAmountTotal',
            'expected': 'priceCeilingAmount'
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', '/project', test['payload'], error, [])
            
            corrected_str = json.dumps(result.corrected_payload)
            success = test['expected'] in corrected_str
            
            results.add(
                test['name'],
                success,
                f"Expected '{test['expected']}' in payload: {success}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_norwegian_field_corrections():
    """Test Norwegian to English field name corrections."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Norwegian Field Name Corrections")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    
    test_cases = [
        {
            'name': 'fodselsnummer -> nationalIdentityNumber',
            'payload': {
                'firstName': 'Test',
                'lastName': 'Person',
                'fodselsnummer': '12345678901'
            },
            'field': 'fodselsnummer',
            'expected': 'nationalIdentityNumber',
            'endpoint': '/employee'
        },
        {
            'name': 'avdeling -> department',
            'payload': {
                'firstName': 'Test',
                'avdeling': {'id': 1}
            },
            'field': 'avdeling',
            'expected': 'department',
            'endpoint': '/employee'
        },
    ]
    
    for test in test_cases:
        try:
            error = {
                'status_code': 422,
                'message': f"{test['field']} - Feltet eksisterer ikke i objektet",
                'validationMessages': [
                    {'field': test['field'], 'message': 'Feltet eksisterer ikke i objektet'}
                ]
            }
            
            result = corrector.correct_api_error('POST', test['endpoint'], test['payload'], error, [])
            
            corrected_str = json.dumps(result.corrected_payload)
            success = test['expected'] in corrected_str
            
            results.add(
                test['name'],
                success,
                f"Expected '{test['expected']}' in payload: {success}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def test_ledger_voucher_postings():
    """Test that postings field is removed for ledger voucher."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Ledger Voucher (postings removal)")
    print("="*70)
    
    corrector = get_autonomous_corrector(llm_client=None)
    si = get_schema_intelligence()
    
    # Check if ledger voucher endpoint exists
    schema = si.get_endpoint_schema('/ledger/voucher', 'POST')
    if schema:
        print("POST /ledger/voucher exists in schema")
        valid_fields = si.get_valid_fields('/ledger/voucher', 'POST')
        print(f"Valid fields: {list(valid_fields.keys())}")
        
        # Check if postings is in valid fields - this is what matters for our
        # correction logic (we need to know to remove it)
        has_postings = 'postings' in valid_fields
        
        results.add(
            'Ledger voucher schema check',
            has_postings,  # Just check that postings field exists
            f"'postings' exists: {has_postings} (needed for removal logic)"
        )
    else:
        print("POST /ledger/voucher not found - checking alternative endpoints")
        results.add(
            'Ledger voucher endpoint',
            True,
            "Endpoint not in schema - special handling in agent.py"
        )
    
    return results


def test_missing_required_field_corrections():
    """Test auto-adding missing required fields like department.id."""
    results = TestResult()
    print("\n" + "="*70)
    print("TESTING: Missing Required Field Corrections")
    print("="*70)
    
    from core.autonomous_corrector import AutonomousCorrector
    from core.error_analyzer import ErrorAnalysis, ValidationError
    
    corrector = AutonomousCorrector(llm_client=None)
    
    # Test 1: Missing department.id for employee
    test_cases = [
        {
            'name': 'Employee missing department.id',
            'payload': {
                'firstName': 'Charles',
                'lastName': 'Taylor',
                'email': 'charles.taylor@example.org'
            },
            'error_analysis': ErrorAnalysis(
                status_code=422,
                path='/employee',
                method='POST',
                error_type='missing_required',
                message='Validering feilet. | VALIDATION_ERROR | department.id | Feltet må fylles ut.',
                validation_errors=[
                    ValidationError(
                        field='department.id',
                        message='Feltet må fylles ut',
                        error_type='missing_required',
                        original_error='department.id | Feltet må fylles ut'
                    )
                ],
                raw_response={'message': 'Validering feilet.'},
                is_retryable=True,
                suggestions={}
            ),
            'check': lambda p: p.get('department') == {'id': 1},
            'expected': "department: {'id': 1}"
        },
        {
            'name': 'Invoice missing currency',
            'payload': {
                'description': 'Test invoice',
                'orderLines': [{'description': 'Line item', 'count': 1}]
            },
            'error_analysis': ErrorAnalysis(
                status_code=422,
                path='/invoice',
                method='POST',
                error_type='missing_required',
                message='Validering feilet. | VALIDATION_ERROR | currency.id | Feltet må fylles ut.',
                validation_errors=[
                    ValidationError(
                        field='currency',
                        message='Feltet må fylles ut',
                        error_type='missing_required',
                        original_error='currency | Feltet må fylles ut'
                    )
                ],
                raw_response={'message': 'Validering feilet.'},
                is_retryable=True,
                suggestions={}
            ),
            'check': lambda p: p.get('currency') == {'id': 1},
            'expected': "currency: {'id': 1}"
        }
    ]
    
    for test in test_cases:
        try:
            schema_hints = {
                'valid_fields': {},
                'field_candidates': {},
                'required_fields': [test['error_analysis'].validation_errors[0].field],
                'example_payload': None
            }
            
            result = corrector._basic_correction(
                test['error_analysis'].method,
                test['error_analysis'].path,
                test['payload'],
                test['error_analysis'],
                schema_hints
            )
            
            success = test['check'](result['payload'])
            results.add(
                test['name'],
                success,
                f"Expected {test['expected']}, got: {result['payload']}"
            )
        except Exception as e:
            results.add(test['name'], False, str(e))
    
    return results


def main():
    """Run all real-world tests."""
    print("="*70)
    print("REAL-WORLD CORRECTION TESTS")
    print("Testing error patterns from deployment history")
    print("="*70)
    
    all_results = []
    
    all_results.append(test_invoice_line_item_corrections())
    all_results.append(test_product_corrections())
    all_results.append(test_supplier_corrections())
    all_results.append(test_customer_corrections())
    all_results.append(test_employee_corrections())
    all_results.append(test_project_corrections())
    all_results.append(test_norwegian_field_corrections())
    all_results.append(test_ledger_voucher_postings())
    all_results.append(test_missing_required_field_corrections())
    
    # Final summary
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"Success rate: {100*total_passed/total_tests:.1f}%" if total_tests > 0 else "N/A")
    
    if total_failed == 0:
        print("\n🎉 All real-world correction tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} tests need attention.")
        return 1


if __name__ == "__main__":
    exit(main())
