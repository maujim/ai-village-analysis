"""Reusable DSPy signatures for conversational-act and task-contract distillation.

Install the project's DSPy dependency before importing this module. The
machine-readable taxonomy and signature metadata live in taxonomy.json, so
consumers that do not need DSPy can read that file directly.

Model outputs are hypotheses with source spans. They are not calibrated
probabilities, verified outcomes, or automatic evidence of task completion.
"""
from typing import List, Literal
import dspy

ActLabel = Literal[
    "instruction_request",
    "question",
    "proposal_plan",
    "commitment_assignment",
    "offer_help",
    "status_report",
    "check_report",
    "assertion",
    "endorsement",
    "qualification",
    "challenge_correction",
    "acknowledgment",
    "uncertain",
]
ContractStatus = Literal[
    "none", "requested", "accepted", "proposed", "reported_complete", "blocked", "unknown"
]
TaskTemplate = Literal[
    "none_or_uncertain",
    "requested_action",
    "verification_task",
    "proposal_plan",
    "status_update",
    "artifact_submission_or_review",
    "coordination_plan",
]


class ClassifyMessageAct(dspy.Signature):
    """Classify the communicative function of one message segment.

    Use only the segment and supplied local context. Return one primary act,
    optional independently expressed secondary acts, and exact supporting
    spans. If the act is unclear, use uncertain and abstain.
    """
    message: str = dspy.InputField(desc="One raw utterance or already segmented clause.")
    local_context: str = dspy.InputField(desc="Relevant nearby turns, or 'none provided'.")
    primary_act: ActLabel = dspy.OutputField(desc="One taxonomy label, or uncertain.")
    secondary_acts: List[ActLabel] = dspy.OutputField(desc="Other separately expressed acts; empty when none.")
    evidence_spans: List[str] = dspy.OutputField(desc="Verbatim substrings from message that support the act labels.")
    abstain: bool = dspy.OutputField(desc="True when the message is too ambiguous or context-poor to label.")
    uncertainty_reason: str = dspy.OutputField(desc="Brief reason, or empty string when not uncertain.")


class SegmentMultiActMessage(dspy.Signature):
    """Split a mixed message into coherent clauses and classify each clause.

    Do not segment merely because the message is long. Split when its
    communicative function changes, such as from a check report to a plan.
    """
    message: str = dspy.InputField(desc="Raw message text.")
    local_context: str = dspy.InputField(desc="Relevant nearby turns, or 'none provided'.")
    segments: List[str] = dspy.OutputField(desc="Ordered verbatim segments; preserve source wording.")
    act_labels: List[ActLabel] = dspy.OutputField(desc="One primary act label per segment, in matching order.")
    evidence_spans: List[str] = dspy.OutputField(desc="Verbatim source substrings supporting the corresponding labels.")
    uncertain_segments: List[str] = dspy.OutputField(desc="Segments whose function remains ambiguous; empty when none.")


class ExtractExecutableTaskContract(dspy.Signature):
    """Extract only explicit, executable requests, assignments, or commitments.

    This is separate from conversational-act classification. Leave unknown
    slots empty. A plan is not a completed action; a reported result is not an
    independently verified outcome.
    """
    message: str = dspy.InputField(desc="Raw message or act segment.")
    act_labels: List[ActLabel] = dspy.InputField(desc="Previously inferred labels; do not treat them as proof.")
    executable: bool = dspy.OutputField(desc="True only for an explicit executable request, assignment, accepted commitment, or actionable proposal.")
    contract_status: ContractStatus = dspy.OutputField(desc="none/requested/accepted/proposed/reported_complete/blocked/unknown.")
    actor: str = dspy.OutputField(desc="Explicit assignee or committing speaker; empty if unstated.")
    action: str = dspy.OutputField(desc="Explicit action phrase; empty if unstated.")
    target: str = dspy.OutputField(desc="Explicit target or artifact; empty if unstated.")
    preconditions: List[str] = dspy.OutputField(desc="Only explicit prerequisites or triggers.")
    constraints: List[str] = dspy.OutputField(desc="Only explicit scope, policy, or method restrictions.")
    success_condition: str = dspy.OutputField(desc="Explicit observable completion condition; empty if unstated.")
    deadline: str = dspy.OutputField(desc="Explicit time/date constraint; empty if unstated.")
    required_evidence: List[str] = dspy.OutputField(desc="Explicit requested proof/report; do not invent evidence requirements.")
    source_spans: List[str] = dspy.OutputField(desc="Verbatim source quotes supporting populated fields, formatted field=quote.")


class CheckArtifactSignature(dspy.Signature):
    """Reusable DSPy contract for checking an artifact or a reported state."""
    artifact_or_subject: str = dspy.InputField(desc="Object, URL, repository, file, or state to inspect.")
    check_criteria: str = dspy.InputField(desc="Explicit property to check; empty if absent in source.")
    scope: str = dspy.InputField(desc="Explicit environment/branch/version/access scope; empty if absent.")
    constraints: List[str] = dspy.InputField(desc="Allowed methods and boundaries stated in the source.")
    check_method: str = dspy.OutputField(desc="Method explicitly requested or described; empty if none.")
    reported_result: str = dspy.OutputField(desc="What the source says the check found; label as reported.")
    evidence_span: str = dspy.OutputField(desc="Verbatim text supporting method or result; empty if none.")
    coverage_limit: str = dspy.OutputField(desc="What the check does not establish; do not infer a broader outcome.")


class SubmitReviewArtifactSignature(dspy.Signature):
    """Reusable contract for submitting or reviewing a versioned artifact."""
    artifact: str = dspy.InputField(desc="Artifact explicitly named in the source.")
    target_repository_or_location: str = dspy.InputField(desc="Explicit repository, branch, path, or destination.")
    required_components: List[str] = dspy.InputField(desc="Explicit required files or contents; empty when unspecified.")
    review_criteria: List[str] = dspy.InputField(desc="Explicit checklist or acceptance criteria; empty when unspecified.")
    action: str = dspy.OutputField(desc="Requested, committed, or reported artifact operation.")
    status: ContractStatus = dspy.OutputField(desc="Status as stated; use reported_complete only for a completion report.")
    version_or_reference: str = dspy.OutputField(desc="Explicit version, commit, MR/PR, or URL; empty if absent.")
    evidence_span: str = dspy.OutputField(desc="Verbatim source span supporting this contract.")
    limitations: List[str] = dspy.OutputField(desc="Unspecified state or limits on what the source establishes.")


class CoordinateActionSignature(dspy.Signature):
    """Reusable contract for distributing work among named participants."""
    goal: str = dspy.InputField(desc="Explicit shared objective, or empty if absent.")
    participants: List[str] = dspy.InputField(desc="Participants explicitly named in the source.")
    constraints: List[str] = dspy.InputField(desc="Explicit coordination rules and restrictions.")
    deadline: str = dspy.InputField(desc="Explicit deadline; empty if absent.")
    assignments: List[str] = dspy.OutputField(desc="Only source-stated actor/action assignments.")
    dependencies: List[str] = dspy.OutputField(desc="Explicit dependencies between assignments; empty if absent.")
    completion_conditions: List[str] = dspy.OutputField(desc="Explicit done conditions; empty if absent.")
    evidence_requirements: List[str] = dspy.OutputField(desc="Explicit reporting or proof requirements; empty if absent.")
