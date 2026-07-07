import sys
import os
sys.path.append(os.getcwd())

from src.shared.models import Narrative, Caption
from src.shared.providers import MockLLMProvider
from src.validation.semantic_validator import SemanticValidator

narrative = Narrative(text="A person starts the coffee maker.", key_events=[], salience_scores={}, evidence_mapping={})
captions = {
    "formal": Caption(text="A person starts the coffee maker.", style="formal", word_count=6)
}

mock_llm = MockLLMProvider()
validator = SemanticValidator(llm_provider=mock_llm)

# Let's inspect class attributes or run validate
report = validator.validate(captions, narrative)
print("REPORT DETAIL:", report)
