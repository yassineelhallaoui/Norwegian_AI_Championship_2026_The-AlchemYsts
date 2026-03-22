import pytest
import os
from unittest.mock import MagicMock
from tempfile import TemporaryDirectory

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.agent import V3Agent
from test_offline import FakeTripletexClient

def test_create_customer():
    client = FakeTripletexClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_customer.json")
        
        # Mock LLM strictly synthesizing the correct payload
        agent.llm.synthesize_api_call = MagicMock(side_effect=[{
            "method": "POST",
            "path": "/customer",
            "payload": {
                "name": "Acme AS",
                "email": "post@acme.no",
                "invoiceSendMethod": "EMAIL"
            }
        }, {"status": "completed"}])
        
        result = agent.process_task("Create customer Acme AS with email post@acme.no")
        
        assert result["status"] == "success"
        assert len(client.calls) == 1
        assert client.calls[0][0] == "POST"
        assert client.calls[0][1] == "/customer"
        assert client.calls[0][2]["data"]["invoiceSendMethod"] == "EMAIL"


def test_create_product():
    client = FakeTripletexClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_product.json")
        
        agent.llm.synthesize_api_call = MagicMock(side_effect=[{
            "method": "POST",
            "path": "/product",
            "payload": {
                "name": "Beratungsstunden",
                "number": "3512",
                "priceExcludingVatCurrency": 42600.0,
                "vatType": {"id": 1}
            }
        }, {"status": "completed"}])
        
        result = agent.process_task("Create product Beratungsstunden, number 3512, price 42600 NOK ex vat")
        
        assert result["status"] == "success"
        assert len(client.calls) == 1
        assert client.calls[0][0] == "POST"
        assert client.calls[0][1] == "/product"
        assert client.calls[0][2]["data"]["priceExcludingVatCurrency"] == 42600.0


def test_create_invoice():
    client = FakeTripletexClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_invoice.json")
        
        agent.llm.synthesize_api_call = MagicMock(side_effect=[{
            "method": "POST",
            "path": "/invoice",
            "payload": {
                "invoiceDate": "2026-03-20",
                "invoiceDueDate": "2026-03-27",
                "customer": {"id": 10}
            }
        }, {"status": "completed"}])
        
        result = agent.process_task("Create invoice for customer 10")
        
        assert result["status"] == "success"
        assert client.calls[-1][0] == "POST"
        assert client.calls[-1][1] == "/invoice"


def test_register_payment():
    class InvoicePaymentClient(FakeTripletexClient):
        def put(self, path: str, **kwargs):
            self.calls.append(("PUT", path, kwargs))
            return {"value": {"id": 1}}
            
    client = InvoicePaymentClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_payment.json")
        
        agent.llm.synthesize_api_call = MagicMock(side_effect=[{
            "method": "PUT",
            "path": "/invoice/99/:payment",
            "payload": {
                "paymentDate": "2026-03-20",
                "paymentTypeId": 333,
                "paidAmountCurrency": 1250.0
            }
        }, {"status": "completed"}])
        
        result = agent.process_task("Register payment of 1250 for invoice 99")
        
        assert result["status"] == "success"
        assert client.calls[-1][0] == "PUT"
        assert client.calls[-1][1] == "/invoice/99/:payment"
        assert client.calls[-1][2]["data"]["paidAmountCurrency"] == 1250.0


def test_reverse_payment():
    class ReversePaymentClient(FakeTripletexClient):
        def put(self, path: str, **kwargs):
            self.calls.append(("PUT", path, kwargs))
            if path.endswith("/:reverse"):
                return {"value": {"id": 888}}
            return {"value": {"id": 1}}
            
    client = ReversePaymentClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_reverse.json")
        
        agent.llm.synthesize_api_call = MagicMock(side_effect=[{
            "method": "PUT",
            "path": "/ledger/voucher/777/:reverse",
            "payload": {
                "date": "2026-03-20"
            }
        }, {"status": "completed"}])
        
        result = agent.process_task("Reverse payment voucher 777")
        
        assert result["status"] == "success"
        assert client.calls[-1][0] == "PUT"
        assert client.calls[-1][1] == "/ledger/voucher/777/:reverse"

def test_cascading_tasks():
    class CascadingClient(FakeTripletexClient):
        def post(self, path: str, **kwargs):
            self.calls.append(("POST", path, kwargs))
            if path == "/customer":
                return {"value": {"id": 10}}
            if path == "/order":
                return {"value": {"id": 20}}
            if path == "/invoice":
                return {"value": {"id": 30}}
            return {"value": {"id": 1}}

    client = CascadingClient()
    agent = V3Agent(client)
    
    with TemporaryDirectory() as tmpdir:
        agent.knowledge.graph_file = os.path.join(tmpdir, "test_kg_cascade.json")
        
        # Mock LLM synthesizing 3 sequential calls, then completion
        agent.llm.synthesize_api_call = MagicMock(side_effect=[
            {
                "method": "POST",
                "path": "/customer",
                "payload": {"name": "Cascade Corp"}
            },
            {
                "method": "POST",
                "path": "/order",
                "payload": {"customer": {"id": 10}}
            },
            {
                "method": "POST",
                "path": "/invoice",
                "payload": {"order": {"id": 20}}
            },
            {"status": "completed"}
        ])
        
        result = agent.process_task("Create customer Cascade Corp, create order for them, then invoice them.")
        
        assert result["status"] == "success"
        assert result["attempts"] == 3 # 3 successful steps
        
        paths_called = [call[1] for call in client.calls]
        assert paths_called == ["/customer", "/order", "/invoice"]
