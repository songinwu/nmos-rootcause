"""섹션 16 시연 문구 — 시나리오·원인 id가 엔진·데이터와 어긋나지 않는지 (streamlit 없이)."""
import json

from src.dashboard import plain_cmos as pc
from src.inference import rootcause_kb as kb
from src.inference import rootcause_kb_moscap as mk


def test_cmos_plain_ids_match_engine_and_scenarios():
    ids = {s["id"] for s in json.load(open("data/demo/rootcause_scenarios.json", encoding="utf-8"))["scenarios"]}
    assert {s["id"] for s in pc.SCENARIOS} <= ids
    assert set(pc.CARD) <= set(kb.CAUSE_IDS)
    assert set(pc.CAUSE_STEP) == set(pc.CARD)
    flow_keys = {k for k, _, _ in pc.FLOW}
    assert all(v is None or v in flow_keys for v in pc.CAUSE_STEP.values())
    assert all({"suspect", "normal", "checks"} <= set(c) for c in pc.CARD.values())


def test_moscap_cards_have_engine_and_answer_parts():
    # 엔진이 말하는 것(suspect)과 넣은 사고(this)가 나뉘어 있어야 한다 — 바운더리
    assert all({"suspect", "this", "normal", "checks"} <= set(c) for c in mk.CARD.values())
    d = json.load(open("data/demo/moscap_scenarios.json", encoding="utf-8"))
    assert set(d["card"]) == set(mk.CAUSE_IDS)
    assert all(r["cause_id"] in mk.CAUSE_INDEX for s in d["scenarios"] for r in s["ranking"])


def test_nmos_scenarios_match_engine():
    """NMOS 시연 — 카드가 원인 40개 전부이고, 넣은 사고가 엔진의 1순위(공동 포함) 안에 있다."""
    from src.inference import rootcause_kb_nmos as nk
    d = json.load(open("data/demo/nmos_scenarios.json", encoding="utf-8"))
    assert set(d["card"]) == set(nk.CAUSE_IDS)
    assert d["bin_types"] == nk.BIN_TYPES and len(d["colors"]) == nk.B
    assert all({"suspect", "this", "normal", "checks"} <= set(c) for c in d["card"].values())
    flow = {k for k, _ in d["flow"]}
    for s in d["scenarios"]:
        assert all(set(r["steps"]) <= flow for r in s["ranking"])
        top = {r["cause_id"] for r in s["ranking"] if r["tied"]} or {s["ranking"][0]["cause_id"]}
        assert s["set_cause"] in top


def test_nmos_magnitude_scenarios():
    """「얼마나」 시나리오 — 넣은 파라미터 하나만 「밖이고 맞는」 줄이고, 엔진 1순위 카드에 크기가 붙는다."""
    d = json.load(open("data/demo/nmos_scenarios.json", encoding="utf-8"))
    planted = [s for s in d["scenarios"] if s.get("planted")]
    assert len(planted) >= 4
    for s in planted:
        assert [r["param"] for r in s["fingerprint"] if r["explains"]] == [s["planted"]["param"]]
        e = s["magnitude"][s["set_cause"]]
        assert e["out_of_range"] and e["consistent"]
    assert all(c.get("kind") in ("전기 값", "여유", "결함 밀도", "장비 변수") for c in d["card"].values())
