"""
Fact collection models for document generation.

Each model mirrors the required_fields from corresponding templates,
providing structured input validation for legal document drafting.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BailApplicationFacts(BaseModel):
    """
    Fact collection for bail applications.

    Mirrors required_fields from bail_application_*.json templates.
    All fields accept plain language input - LLM converts to formal legal language.
    """

    applicant_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the accused/applicant seeking bail",
        examples=["Shri Rajesh Kumar Singh"]
    )

    applicant_address: str = Field(
        ...,
        min_length=10,
        description="Complete residential address of the applicant",
        examples=["123, Vasant Vihar, New Delhi - 110057"]
    )

    fir_number: str = Field(
        ...,
        pattern=r"^FIR No\. \d+/\d{4}$",
        description="First Information Report number with year in format 'FIR No. XXX/YYYY'",
        examples=["FIR No. 123/2024"]
    )

    police_station: str = Field(
        ...,
        description="Name and district of the police station where FIR is registered",
        examples=["Saket Police Station, South Delhi"]
    )

    sections_charged: str = Field(
        ...,
        description="Penal sections under which the accused is charged",
        examples=["Sections 302, 120B of BNS"]
    )

    date_of_arrest: date = Field(
        ...,
        description="Date when the applicant was taken into custody (YYYY-MM-DD format)",
        examples=["2024-01-15"]
    )

    grounds_for_bail: str = Field(
        ...,
        min_length=50,
        description="Legal grounds and circumstances justifying the grant of bail. Plain language - LLM converts to formal legal arguments.",
        examples=["The applicant has deep roots in society, is not a flight risk, and investigation is complete."]
    )

    relief_sought: str = Field(
        ...,
        description="Specific bail relief being requested from the Court",
        examples=["Regular bail pending trial with conditions as deemed fit by this Hon'ble Court"]
    )

    @field_validator('date_of_arrest')
    @classmethod
    def validate_date_of_arrest(cls, v: date) -> date:
        """Ensure arrest date is not in the future."""
        if v > date.today():
            raise ValueError("Date of arrest cannot be in the future")
        return v


class LegalNoticeFacts(BaseModel):
    """
    Fact collection for legal notices.

    Mirrors required_fields from legal_notice_*.json templates.
    Used for pre-litigation notices.
    """

    sender_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the person or entity sending the notice",
        examples=["M/s ABC Enterprises Pvt. Ltd."]
    )

    sender_address: str = Field(
        ...,
        min_length=10,
        description="Complete postal address of the sender for service of reply",
        examples=["101, Corporate Tower, Connaught Place, New Delhi - 110001"]
    )

    recipient_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the person or entity receiving the notice",
        examples=["M/s XYZ Corporation Ltd."]
    )

    recipient_address: str = Field(
        ...,
        min_length=10,
        description="Complete postal address for service of the notice",
        examples=["502, Business Park, Sector 44, Gurugram - 122003"]
    )

    subject_matter: str = Field(
        ...,
        description="Brief description of the matter for which notice is being sent",
        examples=["Breach of Contract and Recovery of Dues"]
    )

    demand_details: str = Field(
        ...,
        min_length=20,
        description="Specific demands, claims, or reliefs sought from the recipient",
        examples=["Payment of Rs. 50,00,000/- with interest @ 18% p.a. from 01.01.2024"]
    )

    compliance_period: str = Field(
        ...,
        description="Time period within which the recipient must comply or respond",
        examples=["15 days from receipt of this notice"]
    )


class AffidavitFacts(BaseModel):
    """
    Fact collection for affidavits.

    Mirrors required_fields from affidavit_*.json templates.
    Used for sworn statements before courts.
    """

    deponent_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the person making the affidavit",
        examples=["Shri Vikram Singh Chauhan"]
    )

    deponent_address: str = Field(
        ...,
        min_length=10,
        description="Complete residential address of the deponent",
        examples=["A-12, Vasant Kunj, New Delhi - 110070"]
    )

    deponent_occupation: str = Field(
        ...,
        description="Profession or occupation of the deponent",
        examples=["Businessman"]
    )

    statement_facts: str = Field(
        ...,
        min_length=20,
        description="The facts and statements being affirmed under oath. Can be numbered paragraphs.",
        examples=["1. I am the petitioner in the above matter. 2. The facts stated in the petition are true to my knowledge."]
    )

    verification_place: str = Field(
        ...,
        description="Place where the affidavit is verified and sworn",
        examples=["New Delhi"]
    )

    @field_validator('statement_facts')
    @classmethod
    def validate_statements(cls, v: str) -> str:
        """Ensure statement_facts is not empty or too short."""
        if not v or len(v.strip()) < 20:
            raise ValueError("Statement of facts must contain at least 20 characters of meaningful content")
        return v


class PetitionFacts(BaseModel):
    """
    Fact collection for petitions.

    Mirrors required_fields from petition_*.json templates.
    Used for filing petitions before courts under various jurisdictions.
    """

    petitioner_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the petitioner filing the petition",
        examples=["ABC Public Interest Foundation"]
    )

    petitioner_address: str = Field(
        ...,
        min_length=10,
        description="Complete address of the petitioner for service",
        examples=["10, Janpath, New Delhi - 110001"]
    )

    respondent_name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Full legal name of the respondent against whom petition is filed",
        examples=["Union of India through Secretary, Ministry of Home Affairs"]
    )

    respondent_address: str = Field(
        ...,
        min_length=10,
        description="Complete address of the respondent for service",
        examples=["North Block, Central Secretariat, New Delhi - 110001"]
    )

    cause_of_action: str = Field(
        ...,
        min_length=30,
        description="Facts giving rise to the petition and when the cause of action arose",
        examples=["The impugned notification dated 01.01.2024 violates Article 14 and 21 of the Constitution. Cause of action arose on 01.01.2024 at New Delhi."]
    )

    relief_sought: str = Field(
        ...,
        min_length=20,
        description="Specific reliefs and prayers being sought from the Court",
        examples=["a) Declare the impugned notification unconstitutional; b) Issue writ of mandamus; c) Grant costs"]
    )

    jurisdiction_ground: str = Field(
        ...,
        min_length=20,
        description="Legal basis for the Supreme Court's jurisdiction",
        examples=["Article 32 of the Constitution of India for enforcement of fundamental rights"]
    )
