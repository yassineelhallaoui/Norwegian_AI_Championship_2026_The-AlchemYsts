from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TaskFile(StrictBaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(StrictBaseModel):
    base_url: str
    session_token: str


class SolveRequest(StrictBaseModel):
    prompt: str
    files: list[TaskFile] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials


class AddressSpec(StrictBaseModel):
    address_line_1: str | None = None
    address_line_2: str | None = None
    postal_code: str | None = None
    city: str | None = None
    country_id: int | None = None


class LookupSpec(StrictBaseModel):
    email: str | None = None
    customer_name: str | None = None
    full_name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    employee_number: str | None = None
    invoice_number: str | None = None
    organization_number: str | None = None


class DepartmentSpec(StrictBaseModel):
    name: str | None = None
    department_number: str | None = None


class EmployeeSpec(StrictBaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    employee_number: str | None = None
    phone_number_mobile: str | None = None
    phone_number_home: str | None = None
    phone_number_work: str | None = None
    date_of_birth: str | None = None
    start_date: str | None = None
    national_identity_number: str | None = None
    occupation_code: str | None = None
    annual_salary: float | None = None
    employment_percentage: float | None = None
    user_type: Literal["STANDARD", "EXTENDED", "NO_ACCESS"] | None = None
    address: AddressSpec | None = None
    department: DepartmentSpec | None = None


class CustomerSpec(StrictBaseModel):
    name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    phone_number_mobile: str | None = None
    organization_number: str | None = None
    description: str | None = None
    invoice_send_method: Literal["EMAIL", "EHF", "EFAKTURA", "AVTALEGIRO", "VIPPS", "PAPER", "MANUAL"] | None = None
    is_supplier: bool | None = None
    is_private_individual: bool | None = None
    postal_address: AddressSpec | None = None
    physical_address: AddressSpec | None = None


class SupplierSpec(StrictBaseModel):
    name: str | None = None
    email: str | None = None
    invoice_email: str | None = None
    phone_number: str | None = None
    organization_number: str | None = None
    description: str | None = None
    is_private_individual: bool | None = None
    postal_address: AddressSpec | None = None
    physical_address: AddressSpec | None = None


class ProductSpec(StrictBaseModel):
    name: str | None = None
    number: str | None = None
    description: str | None = None
    price_excluding_vat: float | None = None
    price_including_vat: float | None = None
    cost_excluding_vat: float | None = None
    vat_type_id: int | None = None
    vat_rate: float | None = None


class OrderLineSpec(StrictBaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    price_mode: Literal["excluding_vat", "including_vat", "unknown"] | None = None
    vat_type_id: int | None = None
    vat_rate: float | None = None
    account_number: str | None = None
    product_name: str | None = None
    product_number: str | None = None


class ProjectSpec(StrictBaseModel):
    name: str | None = None
    number: str | None = None
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    reference: str | None = None
    is_offer: bool | None = None
    is_internal: bool | None = None
    is_fixed_price: bool | None = None


class InvoiceSpec(StrictBaseModel):
    invoice_date: str | None = None
    due_date: str | None = None
    send_to_customer: bool | None = None
    invoice_number: str | None = None
    payment_date: str | None = None
    paid_amount: float | None = None
    payment_type_id: int | None = None
    credit_note_comment: str | None = None
    price_mode: Literal["excluding_vat", "including_vat", "unknown"] | None = None
    comment: str | None = None
    gross_amount: float | None = None


class TaskInterpretation(StrictBaseModel):
    task_type: Literal[
        "create_employee",
        "update_employee",
        "create_customer",
        "update_customer",
        "create_department",
        "create_product",
        "create_project",
        "create_invoice",
        "register_supplier_invoice",
        "record_payroll_voucher",
        "record_travel_expense_voucher",
        "record_accounting_dimension_voucher",
        "register_payment",
        "reverse_payment",
        "create_credit_note",
        "correct_ledger_errors",
        "record_year_end_closing",
        "bank_reconciliation",
        "unknown",
    ]
    language: str | None = None
    confidence: float | None = None
    attachments_summary: str | None = None
    lookup: LookupSpec | None = None
    employee: EmployeeSpec | None = None
    customer: CustomerSpec | None = None
    supplier: SupplierSpec | None = None
    department: DepartmentSpec | None = None
    product: ProductSpec | None = None
    project: ProjectSpec | None = None
    invoice: InvoiceSpec | None = None
    order_lines: list[OrderLineSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    thoughts: str | None = Field(None, description="Use this field for any reasoning or scratchpad thoughts before filling out the rest of the JSON. Do not put thoughts in other fields.")
