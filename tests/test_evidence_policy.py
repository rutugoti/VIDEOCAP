import pytest
from src.shared.models import Observation, Confidence
from src.fusion.evidence_policy import EvidenceResolutionPolicy


def test_evidence_policy_missing_conf():
    policy = EvidenceResolutionPolicy()
    obs = [
        Observation(
            content="Person enters room",
            timestamp=1.0,
            confidence=0.0,
            source="visual",
            observation_type="action",
            id=1
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 1
    kept_obs = resolved.kept[0]
    assert kept_obs.confidence == 0.5
    assert kept_obs.confidence_meta is not None
    assert kept_obs.confidence_meta.value == 0.5
    assert kept_obs.confidence_meta.source == "default"


def test_evidence_policy_discard_low_confidence():
    policy = EvidenceResolutionPolicy(min_confidence=0.3)
    obs = [
        Observation(
            content="Low confidence object",
            timestamp=1.0,
            confidence=0.2,
            source="visual",
            observation_type="object",
            id=1,
            confidence_meta=Confidence(value=0.2, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 0
    assert len(resolved.dropped) == 1
    assert resolved.dropped[0][0].id == 1
    assert "below min_confidence" in resolved.dropped[0][1]


def test_evidence_policy_agree():
    policy = EvidenceResolutionPolicy()
    # Similar content (Jaccard distance < 0.6)
    obs = [
        Observation(
            content="A dog is barking loudly in the garden",
            timestamp=2.0,
            confidence=0.8,
            source="visual",
            observation_type="action",
            id=1,
            confidence_meta=Confidence(value=0.8, source="measured")
        ),
        Observation(
            content="dog barking in garden",
            timestamp=2.1,
            confidence=0.7,
            source="audio",
            observation_type="speech",
            id=2,
            confidence_meta=Confidence(value=0.7, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 2
    # Confidences should be boosted
    assert resolved.kept[0].confidence == pytest.approx(0.85)
    assert resolved.kept[1].confidence == pytest.approx(0.75)
    assert any(c["type"] == "modality_agreement" for c in resolved.conflicts)


def test_evidence_policy_vision_only():
    policy = EvidenceResolutionPolicy()
    obs = [
        Observation(
            content="Person enters room",
            timestamp=1.0,
            confidence=0.8,
            source="visual",
            observation_type="action",
            id=1,
            confidence_meta=Confidence(value=0.8, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 1
    assert resolved.kept[0].confidence == 0.8


def test_evidence_policy_speech_only_high():
    policy = EvidenceResolutionPolicy()
    # Speech only, confidence >= 0.6 -> kept, downweighted by 0.5
    obs = [
        Observation(
            content="Someone says hello",
            timestamp=1.0,
            confidence=0.8,
            source="audio",
            observation_type="speech",
            id=1,
            confidence_meta=Confidence(value=0.8, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 1
    assert resolved.kept[0].confidence == 0.4
    assert any(c["type"] == "audio_only_downweighted" for c in resolved.conflicts)


def test_evidence_policy_speech_only_low():
    policy = EvidenceResolutionPolicy()
    # Speech only, confidence < 0.6 -> dropped
    obs = [
        Observation(
            content="Faint whisper",
            timestamp=1.0,
            confidence=0.5,
            source="audio",
            observation_type="speech",
            id=1,
            confidence_meta=Confidence(value=0.5, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 0
    assert len(resolved.dropped) == 1
    assert "Speech only observation" in resolved.dropped[0][1]


def test_evidence_policy_contradiction():
    policy = EvidenceResolutionPolicy(escalate_conflict_min_conf=0.7)
    # Different content (Jaccard distance >= 0.6)
    obs = [
        Observation(
            content="A cat climbs a tree",
            timestamp=2.0,
            confidence=0.8,
            source="visual",
            observation_type="action",
            id=1,
            confidence_meta=Confidence(value=0.8, source="measured")
        ),
        Observation(
            content="loud sound of car crash",
            timestamp=2.1,
            confidence=0.7,
            source="audio",
            observation_type="sound",
            id=2,
            confidence_meta=Confidence(value=0.7, source="measured")
        )
    ]
    resolved = policy.resolve(obs)
    assert len(resolved.kept) == 2
    # Both kept, marked as contested, each other's ID in alternatives
    o1 = next(o for o in resolved.kept if o.id == 1)
    o2 = next(o for o in resolved.kept if o.id == 2)
    assert o1.contested is True
    assert o2.contested is True
    assert 2 in o1.alternatives
    assert 1 in o2.alternatives
    assert len(resolved.contested_pairs) == 1
    assert any(c["type"] == "modality_contradiction" and c["escalate_to_llm"] for c in resolved.conflicts)
