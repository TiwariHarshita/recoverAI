from pydantic import BaseModel, Field

from .actions import RecoveryAction
from .enums import FailureClass


class CandidateActionSet(BaseModel):
    """
    Deterministic output from the candidate-action generator.

    This layer does NOT rank actions.
    It only decides which actions are eligible to be scored later.
    """

    case_id: str

    failure_class: FailureClass

    actions: list[RecoveryAction] = Field(
        default_factory=list
    )

    generation_notes: list[str] = Field(
        default_factory=list
    )