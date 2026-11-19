"""Validated model decisions shared by retrieval, generation and correction."""
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator
from policy_rag.calculation import CalculationRequest

Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=10000)]
ShortText=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=400)]
Slot=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=80)]

def parse_model_bool(value: Any) -> bool:
    if isinstance(value,bool):return value
    if isinstance(value,str):
        lowered=value.strip().lower()
        if lowered in {"true","1","yes"}:return True
        if lowered in {"false","0","no"}:return False
    raise ValueError("Invalid model boolean verdict")

class EvidenceGrade(BaseModel):
    sufficient: bool
    missing_type: Literal["none","user_facts","policy_evidence","both"] | None = None
    missing_aspect: str = Field(default="",max_length=800)
    missing_slots: list[Slot] = Field(default_factory=list,max_length=6)
    clarification_questions: list[ShortText] = Field(default_factory=list,max_length=6)
    rationale: str = Field(default="",max_length=1600)

    @field_validator("sufficient",mode="before")
    @classmethod
    def parse_sufficient(cls,value):return parse_model_bool(value)

    @model_validator(mode="after")
    def consistent_decision(self):
        self.missing_aspect=self.missing_aspect.strip()
        if self.sufficient:
            if self.missing_type not in {None,"none"} or self.missing_aspect or self.missing_slots or self.clarification_questions:
                raise ValueError("Sufficient evidence cannot contain a blocking gap")
            self.missing_type="none"
        else:
            if self.missing_type not in {"user_facts","policy_evidence","both"} or not self.missing_aspect:
                raise ValueError("Insufficient verdict requires an explicit gap type and description")
            if self.missing_type in {"user_facts","both"} and (not self.missing_slots or not self.clarification_questions):
                raise ValueError("Missing user facts require slots and actual questions")
        return self

class RewritePayload(BaseModel):
    queries: list[ShortText] = Field(min_length=1,max_length=2)

class AnswerPayload(BaseModel):
    action: Literal["answer","clarify","abstain"] = "answer"
    answer: Text
    calculations: list[CalculationRequest] = Field(default_factory=list,max_length=6)
    citation_chunk_ids: list[Slot] = Field(default_factory=list,max_length=20)
    missing_slots: list[Slot] = Field(default_factory=list,max_length=6)
    clarification_questions: list[ShortText] = Field(default_factory=list,max_length=6)

    @model_validator(mode="after")
    def clarification_is_actionable(self):
        if self.action=="clarify" and (not self.missing_slots or not self.clarification_questions):
            raise ValueError("Clarification needs at least one missing slot and question")
        return self
