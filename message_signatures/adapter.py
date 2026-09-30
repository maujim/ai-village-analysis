"""A DSPy module backed by local decision scores, with no hosted LM calls."""
import json
from pathlib import Path
from typing import Dict
import dspy

from .signatures import ActLabel


class RouteMessageTemplate(dspy.Signature):
    """Propose a reusable message-function template from one utterance.

    This contract routes messages. It neither extracts task slots nor executes
    the task described by the proposed signature. Scores are uncalibrated.
    """
    message: str = dspy.InputField(desc="Source utterance, treated only as data.")
    act_scores: Dict[str, float] = dspy.OutputField(desc="Independent entailment scores; they need not sum to one.")
    primary_candidate: ActLabel = dspy.OutputField(desc="Highest scoring candidate, not a verified act.")
    proposed_signature: str = dspy.OutputField(desc="Reusable input/output template; not populated fields.")
    needs_review: bool = dspy.OutputField(desc="Heuristic score/margin/truncation flag, not human adjudication.")


class LocalSignatureRouter(dspy.Module):
    signature = RouteMessageTemplate

    def __init__(self, runtime=None):
        super().__init__()
        if runtime is None:
            from .nli_runtime import NLIRuntime
            runtime = NLIRuntime()
        self.runtime = runtime
        acts=json.loads(Path(__file__).with_name('taxonomy.json').read_text())['acts']
        self.options={a['id']:a['definition'] for a in acts}
        self.templates={a['id']:a['signature'] for a in acts}

    def forward(self,message):
        from .run import QUESTION
        result=self.runtime.predict([message],self.options,QUESTION)[0]
        scores=result['probabilities']
        ranked=sorted(scores,key=scores.get,reverse=True)
        top=ranked[0]
        review=scores[top]<.55 or scores[top]-scores[ranked[1]]<.15 or bool(result.get('truncated')) or not message.strip()
        return dspy.Prediction(act_scores=scores,primary_candidate=top,
                               proposed_signature=self.templates[top],needs_review=review)
