"""
섹션 16 — 시연 화면 (2026-10-16). 발표 시연은 이 섹션에서 「우리가 만든 MOSCAP」 ↔ 「CMOS 문헌 사례」를 전환해 보인다.

본인 피드백과 교수님 시선 점검(wiki 종합설계-피드백대응 §5-0)에서 정한 것:
- **3단계로 넘긴다** *(2026-10-16 — 옛 섹션 15 「시연 3단계」를 이리로 합치고 15는 뺐다)* — ① 모양(CNN) → ② 검사 결과 →
  ③ 먼저 볼 공정. 단계에 들어올 때 「CNN이 모양 판단 중…」처럼 모델이 하는 일을 한 줄씩 보여 준다(본인: 「모델이
  생각하는 것처럼 보여주게」). 계산은 웨이퍼를 고를 때 한 번이고(MOSCAP은 JSON, CMOS는 진단 API) 단계는 그 결과를
  차례로 펼치는 연출이다 — 단계마다 따로 계산하면 숫자가 어긋난다
- ③은 한 화면에 맵 · 검사 결과 · 만드는 순서(의심 공정에 불) · 엔진이 말하는 것 — 스크롤 없이 읽히게
- **엔진이 말하는 것**(어느 단계 · 어느 방향 · 확인할 것)과 **이 웨이퍼에 넣은 사고**(정답 공개 — 「이번에 100 °C」 같은
  조건값)를 나눈다. 엔진은 조건값을 맞히지 않는다(바운더리, 피드백대응 §3-1)
- 결론은 판결이 아니라 「~했을 가능성」 말투
- 긴 설명(무슨 일이 · 왜 이 모양 · 왜 이 검사 결과 · C-V 개념도)은 「왜 그런가」 새 창

MOSCAP은 data/demo/moscap_scenarios.json(scripts/make_moscap_scenarios.py — 실제 맵 · CNN 실제 추론 · 합성 bin ·
MOSCAP 지식표 순위 · 화면 문구)을, CMOS는 진단 API(/demo/rootcause/{id} — 섹션 14와 같은 엔진)를 읽는다.
NMOS *(2026-10-28)* 는 data/demo/nmos_scenarios.json(scripts/make_nmos_scenarios.py — 41단계(38 + SAB 3) · 측정 12칸 · 원인 40개)을
읽는다. 칸이 11개라 색을 시나리오 파일에서 받고, 만드는 순서는 38줄 대신 공정 묶음 7개에 단계 칩으로 보인다.
*(2026-10-28 보강 — 「얼마나」)* NMOS는 ② 단계에 웨이퍼 WAT 값(Vt · SS · on 전류 · C-V 산화막)을 같이 보이고, ③ 엔진 칸에
「얼마나 — 어느 파라미터가 얼마나 · 90% 범위 · 정상 범위 · 다른 측정과 맞나」를 붙인다. 크기를 못 재는 원인은 왜 못 재는지
(유형)를 보인다. 계산은 scripts/make_nmos_scenarios.py가 src/inference/nmos_device.py로 미리 해 JSON에 담는다.
대시보드는 src.inference를 import하지 않는다.
"""
import json
import os
import time

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from src.dashboard import plain_cmos as pc
    from src.dashboard import ui
except ImportError:          # app.py와 같은 폴백 — streamlit을 src/dashboard에서 띄운 경우
    import plain_cmos as pc
    import ui

API_URL = os.getenv("API_URL", "http://localhost:8000")
MOSCAP_PATH = "data/demo/moscap_scenarios.json"
NMOS_PATH = "data/demo/nmos_scenarios.json"
WORLDS = {"moscap": "우리가 만든 MOSCAP", "cmos": "CMOS 문헌 사례", "nmos": "NMOS 41단계"}
JSON_PATH = {"moscap": MOSCAP_PATH, "nmos": NMOS_PATH}
PAT_KO = {"none": "정상", "center": "가운데", "donut": "도넛", "edge-loc": "가장자리 일부",
          "edge-ring": "가장자리 링", "loc": "한 곳에 뭉침", "near-full": "거의 전체",
          "random": "흩어짐", "scratch": "선"}
BIN_COLORS = ["#8F7EF2", "#E85D5D", "#C98500", "#2E9FD0"]      # 검사 순서 1~4
RANK_COLOR = ["#E85D5D", "#C98500", "#93A1B0"]


# ── 데이터 ───────────────────────────────────────

@st.cache_data
def _json_at(path: str, mtime: float):   # mtime은 캐시 키 — JSON을 다시 만들면 새로 읽는다
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _moscap():
    return _json_at(MOSCAP_PATH, os.path.getmtime(MOSCAP_PATH))


def _nmos():
    return _json_at(NMOS_PATH, os.path.getmtime(NMOS_PATH))


@st.cache_data(ttl=600)
def _cmos_fetch(sid: str):
    r = requests.get(f"{API_URL}/demo/rootcause/{sid}", timeout=30)
    r.raise_for_status()
    return r.json()


def _ranked(items):
    """[(키, 사후확률, 공동 여부)] → 순위 숫자(공동이면 같은 숫자)."""
    out, rank = [], 0
    for k, it in enumerate(items):
        if k == 0 or not (it["tied"] and items[k - 1]["tied"]):
            rank = k + 1
        out.append(rank)
    return out


def _view_moscap(sid):
    d = _moscap()
    sc = next(s for s in d["scenarios"] if s["id"] == sid)
    card = d["card"]
    step_name = {f[0]: f[1] for f in d["flow"]}
    ranks = _ranked(sc["ranking"])
    ranking = [dict(rank=rk, tied=r["tied"], step=r["step"], step_name=step_name[r["step"]], **card[r["cause_id"]])
               for rk, r in zip(ranks, sc["ranking"])]
    if len(sc["truth"]) > 1:
        injected = "실습에서 빼먹은 단계 셋을 그대로 넣었어요 — " + " · ".join(
            f"{step_name[next(r['step'] for r in sc['ranking'] if r['cause_id'] == t)]}: {card[t]['this']}"
            for t in sc["truth"] if any(r["cause_id"] == t for r in sc["ranking"]))
    else:
        c = card[sc["set_cause"]]
        injected = f"{c['title']} — 기준 「{c['normal']}」 → 이번에 「{c['this']}」"
    return dict(
        world="moscap", id=sid, title=sc["title"], story=sc["story"],
        banner=("우리가 포항공대에서 만든 MOSCAP — 4인치 웨이퍼 한 장 가득",
                "W 전극 50 nm / Al₂O₃ 절연막 5 nm / p형 Si · 맵의 칸 하나 = 커패시터 한 묶음 · 칸마다 C-V 검사"),
        flow=[(k, n, cond, done) for k, n, _eq, cond, done in d["flow"]],
        die_map=sc["die_map"], bins=d["bin_types"], bin_plain=d["bin_plain"], bin_test=d["bin_plain_test"],
        counts=sc["bin_counts"],
        cnn=(sc["cnn"]["top"], sc["cnn"]["conf_pct"], [(t["pattern"], t["pct"]) for t in sc["cnn"]["top3"][1:]]),
        ranking=ranking, note=sc.get("note"), injected=injected, lot=sc["source"]["lot"],
        hit=sc["ranking"][0]["cause_id"] in sc["truth"],
        chips=[("맵", "WM-811K 실측", True), ("모양 판정", "CNN 실제 추론", True),
               ("검사 결과", "합성", False), ("공정 조건", "우리 실습 기록", True), ("배경 먼지", "대학 팹 가정", False)])


def _view_cmos(sid):
    meta = next(s for s in pc.SCENARIOS if s["id"] == sid)
    r = _cmos_fetch(sid)
    dist = sorted(r["pattern"]["distribution_pct"].items(), key=lambda kv: -kv[1])
    cands = r["candidates"][:3]
    step_name = {k: n for k, n, _ in pc.FLOW}
    ranking = []
    for c in cands:
        card = pc.CARD.get(c["cause_id"], dict(title=c["name"], suspect=c["name"], normal="—", what="", shape="",
                                                test="", checks=c["checks"], source=c.get("source", "")))
        step = pc.CAUSE_STEP.get(c["cause_id"])
        ranking.append(dict(rank=c["rank"], tied=c.get("tied", False), step=step,
                            step_name=step_name.get(step, "전 공정 공통") if step else "전 공정 공통", **card))
    set_cause = r.get("scenario", {}).get("set_cause")
    return dict(
        world="cmos", id=sid, title=meta["title"], story=meta["story"],
        banner=("45nm급 로직 칩 — 문헌 사례로 짠 가상 공정",
                "트랜지스터 위에 Cu 배선을 여러 층 쌓은 칩 · 맵은 실제 공장 웨이퍼(WM-811K) · 칸 하나 = 칩 하나 · 칸마다 EDS 검사"),
        flow=[(k, n, desc, True) for k, n, desc in pc.FLOW],
        die_map=r["die_map"], bins=pc.BIN_TYPES, bin_plain=pc.BIN_PLAIN, bin_test=pc.BIN_PLAIN_TEST,
        counts=r["bins"]["counts"],
        cnn=(dist[0][0], dist[0][1], [(p, v) for p, v in dist[1:3]]),
        ranking=ranking, note=meta.get("note"), injected=meta["injected"],
        lot=r.get("scenario", {}).get("source", {}).get("lot", "—"),
        hit=bool(cands) and any(c["cause_id"] == set_cause and (c["rank"] == 1) for c in cands),
        chips=[("맵", "WM-811K 실측", True), ("모양 판정", "CNN 실제 추론", True),
               ("검사 결과", "합성 (문헌 비율)", False), ("공정 사슬", "문헌 원문", True)])


def _view_nmos(sid):
    d = _nmos()
    sc = next(s for s in d["scenarios"] if s["id"] == sid)
    card = d["card"]
    step_name = dict(d["flow"])
    ranks = _ranked(sc["ranking"])
    ranking = []
    for rk, r in zip(ranks, sc["ranking"]):
        steps = r["steps"]     # 공통 원인(CMP 긁힘)은 여러 단계, 로봇 · 클린룸은 단계 없음
        ranking.append(dict(rank=rk, tied=r["tied"], step=steps[0] if steps else None, steps=steps,
                            step_name=" · ".join(f"{_n(k)}. {step_name[k]}" for k in steps) or "전 공정 공통",
                            cause_id=r["cause_id"], mag=sc.get("magnitude", {}).get(r["cause_id"]),
                            **card[r["cause_id"]]))
    ev = [c["evidence"] for c in card.values()]
    n_src, n_moved = sum(e in ("원문", "일부") for e in ev), ev.count("옮김")
    return dict(
        world="nmos", id=sid, title=sc["title"], story=sc["story"],
        banner=("NMOS 트랜지스터 — 41단계(38 + SAB 3) · 마스크 6장으로 짠 공정",
                "P형 기판 · STI 분리 · LDD 없음 · salicide(SAB로 시험 패턴만 막음) · W contact · Al 배선 1층 · "
                "맵은 실제 공장 웨이퍼(WM-811K) · 칸 하나 = 칩 하나 · 칩마다 측정 9가지(Vt는 높음 · 낮음, 면저항은 "
                "패턴 셋으로 나눠 12칸)"),
        flow=[(k, n, "", True) for k, n in d["flow"]], groups=d["groups"],
        die_map=sc["die_map"], bins=d["bin_types"], bin_plain=d["bin_plain"], bin_test=d["bin_plain_test"],
        colors=d["colors"], counts=sc["bin_counts"],
        cnn=(sc["cnn"]["top"], sc["cnn"]["conf_pct"], [(t["pattern"], t["pct"]) for t in sc["cnn"]["top3"][1:]]),
        ranking=ranking, note=sc.get("note"), lot=sc["source"]["lot"],
        injected=" + ".join(card[t]["this"] for t in sc.get("truth", [sc["set_cause"]])),   # 사고를 둘 넣은 웨이퍼도 있다
        hit=any(rk == 1 and r["cause_id"] in sc.get("truth", [sc["set_cause"]]) for rk, r in zip(ranks, sc["ranking"])),
        warning=sc.get("warning"),
        n_cause=len(card), bin_rule="가장 크게 벗어난 측정",
        wat=sc.get("wat"), wat_spec=d.get("wat_spec"), planted=sc.get("planted"),
        magnitude=sc.get("magnitude", {}), fingerprint=sc.get("fingerprint", []), set_cause=sc["set_cause"],
        chips=[("맵", "WM-811K 실측", True), ("모양 판정", "CNN 실제 추론", True),
               ("측정 결과", "합성 (지식표 방향)", False), ("WAT 값", "합성 (MOS 식)", False),
               ("공정 순서", "IDEC 0.5 µm + KSIA STI", True),
               ("원인 표", f"{len(card)}개 — 문헌 원문 {n_src} · 옮김 {n_moved} · 강의 · 원리 {len(card) - n_src - n_moved}",
                False)])


# ── 부품 ─────────────────────────────────────────

def _colors(v):
    return v.get("colors", BIN_COLORS)


WAT_ORDER = ["Vt", "SS", "ion", "tox_cv", "rs_sd", "rs_poly", "rs_sil"]


def _n(k):
    """단계 번호 — 「03」 → 「3」, 끼운 단계 「26a」는 그대로."""
    return k.lstrip("0") or k


def _wat_cells(v):
    """WAT 값 네 줄 — (이름, 이번, 기준, 합격선, 벗어났나)."""
    w, ws = v["wat"]["meas"], v["wat_spec"]
    base, spec = ws["base"], ws["spec"]
    out = []
    for k in WAT_ORDER:
        if k not in w:
            continue
        x = w[k]
        if k == "Vt":
            now, ref, rng_ = f"{x:.2f} V", f"{base[k]:.2f} V", f"{spec[k][0]:.2f} ~ {spec[k][1]:.2f} V"
            bad = not v["wat"]["in_spec"][k]
        elif k == "SS":
            now, ref, rng_ = f"{x:.0f} mV/dec", f"{base[k]:.0f}", f"{spec[k][1]:.0f} 이하"
            bad = not v["wat"]["in_spec"][k]
        elif k == "ion":
            now, ref, rng_ = f"{(x - 1) * 100:+.0f}%", "기준", f"{(spec[k][0] - 1) * 100:+.0f} ~ {(spec[k][1] - 1) * 100:+.0f}%"
            bad = not v["wat"]["in_spec"][k]
        elif k == "tox_cv":
            lo, hi = ws["tox_range"]
            now, ref, rng_ = f"{x:.0f} Å", f"{base[k]:.0f} Å", f"정상 {lo:.0f} ~ {hi:.0f} Å"
            bad = not (lo <= x <= hi)
        else:   # 면저항 셋 — silicide 덮인 곳은 값이 작아 소수 둘째 자리까지
            lo, hi = spec[k]
            d = 2 if base[k] < 10 else 0
            now, ref = f"{x:.{d}f} Ω/sq", f"{base[k]:.{d}f}"
            rng_ = f"{lo:.{d}f} ~ {hi:.{d}f}" if lo is not None else f"{hi:.{d}f} 이하"
            bad = not v["wat"]["in_spec"][k]
        out.append((ws["label"][k], now, ref, rng_, bad))
    return out


def _wat_table(v):
    rows = "".join(
        f'<tr><td style="padding:3px 8px;color:{ui.INK_MUTED};">{n}</td>'
        f'<td style="padding:3px 8px;font-weight:700;color:{"#E85D5D" if bad else ui.INK};">{now}{" ⚠" if bad else ""}</td>'
        f'<td style="padding:3px 8px;color:{ui.INK_MUTED};">{ref}</td><td style="padding:3px 8px;color:{ui.INK_MUTED};">{r}</td></tr>'
        for n, now, ref, r, bad in _wat_cells(v))
    st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:13px;border:1px solid {ui.BORDER};">'
                f'<tr style="color:{ui.INK_MUTED};font-size:11.5px;"><td style="padding:3px 8px;">측정</td>'
                f'<td style="padding:3px 8px;">이번</td><td style="padding:3px 8px;">기준</td>'
                f'<td style="padding:3px 8px;">합격선</td></tr>{rows}</table>', unsafe_allow_html=True)


def _wat_line(v):
    """③ 왼쪽 — WAT 값 중 합격선 밖인 것만 (일곱 개를 다 늘어놓으면 줄이 넘친다)."""
    cells = _wat_cells(v)
    bad = [f'<span style="color:#E85D5D;">{n} {now}</span>' for n, now, _, _, b in cells if b]
    txt = (" · ".join(bad) + f" · 나머지 {len(cells) - len(bad)}개 정상") if bad else "전부 합격선 안"
    st.markdown(f'<div style="font-size:12.5px;color:{ui.INK_MUTED};margin-top:2px;">WAT — {txt}</div>',
                unsafe_allow_html=True)


def _mag_line(r):
    """③ 엔진 칸의 「얼마나」 한 줄 — 크기를 재는 원인이면 크기 · 90% 범위 · 정상 범위 · 다른 측정과 맞나, 아니면 왜 못 재나."""
    mg = r.get("mag")
    if mg and mg.get("kind") == "density":   # 결함 밀도 — 맵의 불량 칩 비율에서 칩당 결함 수
        return (f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-top:3px;">얼마나 — '
                f'<b style="color:{ui.ACCENT};font-size:14px;">{mg["label"]} 약 {mg["value"]}</b> '
                f'<span>(90% {mg["lo"]} ~ {mg["hi"]})</span><br>웨이퍼 {mg["normal"]} · {mg["from"]}으로 계산</div>')
    if not mg:
        return (f'<div style="color:{ui.INK_MUTED};font-size:12px;margin-top:3px;">{r.get("kind_note", "")} '
                f'<span style="font-size:11px;">(유형: {r.get("kind", "—")})</span></div>')
    if not mg["out_of_range"]:
        head = f'{mg["label"]} {mg["value"]} — 정상 범위({mg["normal"]}) 안이라 이 파라미터는 아니에요'
        return f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-top:3px;">얼마나 — {head}</div>'
    cons = ("다른 측정도 맞아요 ✓" if mg["consistent"] else f'안 맞는 측정 — {" · ".join(mg["mismatch"])} ✗')
    return (f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-top:3px;">얼마나 — '
            f'<b style="color:{ui.ACCENT};font-size:14px;">{mg["label"]} 약 {mg["value"]}</b> '
            f'<span>(90% {mg["lo"]} ~ {mg["hi"]})</span><br>정상 {mg["normal"]} 밖 · {mg["from"]} 값으로 계산 · {cons}</div>')


def _rank_label(r):
    return ("공동 " if r["tied"] else "") + f"{r['rank']}순위"


def _banner(v):
    head, sub = v["banner"]
    st.markdown(
        f'<div style="background:{ui.SURFACE};border:1px solid {ui.BORDER};border-radius:10px;padding:8px 14px;'
        f'margin:2px 0 8px;"><span style="color:{ui.INK_MUTED};font-size:12px;">이 웨이퍼는 무엇을 만드나 · </span>'
        f'<span style="color:{ui.INK};font-size:16px;font-weight:700;">{head}</span>'
        f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-top:2px;">{sub}</div></div>',
        unsafe_allow_html=True)


def _map(v, key, binned=True):
    """binned=False면 ① 단계용 — 검사 결과를 감추고 합격/불합격만(CNN이 보는 그대로)."""
    die = np.asarray(v["die_map"])
    if binned:
        cols = [ui.BG, ui.STEEL] + list(_colors(v))
        names = {0: "웨이퍼 밖", 1: "합격"} | {i + 2: v["bin_plain"][b] for i, b in enumerate(v["bins"])}
    else:
        die = np.minimum(die, 2)
        cols = [ui.BG, ui.STEEL, ui.ACCENT]
        names = {0: "웨이퍼 밖", 1: "합격", 2: "불합격"}
    n = len(cols)
    scale = []
    for i, c in enumerate(cols):
        scale += [[i / n, c], [(i + 1) / n, c]]
    text = [[names.get(int(x), "") for x in row] for row in die]
    fig = go.Figure(go.Heatmap(z=die, text=text, colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False,
                               xgap=1.2, ygap=1.2, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(**ui._base_layout(height=285, xaxis=dict(visible=False, scaleanchor="y"),
                                        yaxis=dict(visible=False, autorange="reversed"),
                                        margin=dict(l=2, r=2, t=2, b=2)))
    st.plotly_chart(fig, width="stretch", config=ui._PLOTLY_CONFIG, key=key)


def _bins(v, key, top=None):
    """칸이 4개면 검사 순서대로, 많으면(NMOS 12칸) 많은 순으로 — top을 주면 상위 top개 + 「그 밖」 한 줄."""
    cnt = v["counts"]
    tot = max(sum(cnt.values()), 1)
    col = dict(zip(v["bins"], _colors(v)))
    rows = [(v["bin_plain"][b], cnt[b], col[b]) for b in v["bins"]]
    if len(rows) > 4:
        rows.sort(key=lambda t: -t[1])
        if top and len(rows) > top:
            rest = rows[top:]
            rows = rows[:top] + [(f"그 밖 {len(rest)}가지", sum(t[1] for t in rest), ui.STEEL)]
    rows = list(reversed(rows))
    fig = go.Figure(go.Bar(
        x=[n / tot * 100 for _, n, _ in rows], y=[lab for lab, _, _ in rows], orientation="h",
        marker_color=[c for _, _, c in rows],
        text=[f"{n / tot * 100:.0f}%" for _, n, _ in rows], textposition="outside", hoverinfo="skip"))
    fig.update_layout(**ui._base_layout(height=max(128, 23 * len(rows) + 10), margin=dict(l=2, r=6, t=2, b=2),
                                        xaxis=dict(visible=False, range=[0, 125]),
                                        yaxis=dict(tickfont=dict(size=12.5)), bargap=0.25))
    st.plotly_chart(fig, width="stretch", config=ui._PLOTLY_CONFIG, key=key)
    dom = max(v["bins"], key=lambda b: cnt[b])
    st.markdown(f'<div style="font-size:12.5px;color:{ui.INK_MUTED};margin-top:-6px;">불량 {tot}칸 중 {_most(cnt[dom], tot)} '
                f'<b style="color:{col[dom]};">{v["bin_plain"][dom]}</b> · '
                f'{v["bin_test"][dom]}</div>', unsafe_allow_html=True)


def _most(n, tot):
    return "대부분" if n * 2 > tot else "가장 많은 것은"


def _prob(v, key):
    """CNN 확률 상위 셋 — 1위만 강조."""
    top, conf, nxt = v["cnn"]
    items = list(reversed([(top, conf)] + list(nxt)))
    fig = go.Figure(go.Bar(
        x=[x for _, x in items], y=[PAT_KO.get(p, p) for p, _ in items], orientation="h",
        marker_color=[ui.ACCENT if p == top else ui.STEEL for p, _ in items],
        text=[f"{x:.1f}%" for _, x in items], textposition="outside", hoverinfo="skip"))
    fig.update_layout(**ui._base_layout(height=120, margin=dict(l=2, r=6, t=2, b=2),
                                        xaxis=dict(visible=False, range=[0, 125]),
                                        yaxis=dict(tickfont=dict(size=12.5)), bargap=0.3))
    st.plotly_chart(fig, width="stretch", config=ui._PLOTLY_CONFIG, key=key)


def _flow_groups(v):
    """NMOS 38단계 — 공정 묶음 7개, 묶음마다 단계 칩. 의심 단계 칩에 불, 의심이 없는 묶음은 흐리게."""
    hi = {}
    for r in v["ranking"]:
        for k in r["steps"]:
            hi.setdefault(k, r)
    name = {k: n for k, n, _, _ in v["flow"]}
    out = []
    for g, a, b in v["groups"]:
        keys = [k for k in name if a <= k <= b]
        lit = any(k in hi for k in keys)
        chips = []
        for k in keys:
            r = hi.get(k)
            if r:
                col = RANK_COLOR[min(r["rank"] - 1, 2)]
                chips.append(f'<span style="border:2px solid {col};background:#1E2632;color:{ui.INK};font-weight:700;'
                             f'font-size:12.5px;padding:1px 7px;border-radius:7px;margin:2px 3px 2px 0;display:inline-block;">'
                             f'{_n(k)}. {name[k]} <span style="background:{col};color:#0D1319;font-size:10.5px;'
                             f'padding:0 5px;border-radius:8px;">{_rank_label(r)}</span></span>')
            else:
                chips.append(f'<span style="color:{ui.INK_MUTED};font-size:11.5px;padding:0 3px;margin:1px 2px 1px 0;'
                             f'display:inline-block;">{_n(k)} {name[k]}</span>')
        dim = "" if lit else "opacity:.55;"
        out.append(f'<div style="border-left:2px solid {RANK_COLOR[0] if lit else ui.BORDER};padding:2px 0 3px 8px;'
                   f'margin:3px 0;{dim}"><div style="color:{ui.INK};font-size:12.5px;font-weight:600;">{g} '
                   f'<span style="color:{ui.INK_MUTED};font-weight:400;font-size:11px;">{_n(a)}{f"–{_n(b)}" if b != a else ""}</span></div>'
                   f'<div style="line-height:1.55;">{"".join(chips)}</div></div>')
    if not v["ranking"][0]["steps"]:
        out.append(f'<div style="color:{ui.INK_MUTED};font-size:12px;">1순위는 특정 단계가 아니라 전 공정 공통이에요.</div>')
    st.markdown(f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-bottom:2px;">만드는 순서 {len(v["flow"])}단계 — '
                f'의심 단계에 불이 들어와요</div>' + "".join(out), unsafe_allow_html=True)


def _flow(v):
    if v["world"] == "nmos":
        _flow_groups(v)
        return
    hi = {}
    for r in v["ranking"]:
        if r["step"] and r["step"] not in hi:
            hi[r["step"]] = r
    rows = []
    for i, (key, name, desc, done) in enumerate(v["flow"]):
        r = hi.get(key)
        num = i if v["world"] == "moscap" else i + 1
        if r:
            col = RANK_COLOR[min(r["rank"] - 1, 2)]
            skip = "" if (done or "안 함" in desc) else " · 실습에서 안 함"
            rows.append(
                f'<div style="border:2px solid {col};background:#1E2632;border-radius:7px;padding:4px 9px;margin:3px 0;">'
                f'<div style="color:{ui.INK};font-size:13.5px;font-weight:700;">{num}. {name} '
                f'<span style="background:{col};color:#0D1319;font-size:11px;padding:0 6px;border-radius:9px;">'
                f'{_rank_label(r)}</span></div>'
                f'<div style="color:{ui.INK_MUTED};font-size:11.5px;line-height:1.35;">{desc}{skip}</div></div>')
        else:
            style = "" if done else "opacity:.5;"
            tag = "" if done else f' <span style="font-size:10.5px;border:1px dashed {ui.INK_MUTED};padding:0 4px;border-radius:5px;">실습에서 안 함</span>'
            rows.append(f'<div title="{desc}" style="color:{ui.INK};font-size:13px;padding:3px 9px;margin:1px 0;'
                        f'border-left:2px solid {ui.BORDER};{style}">{num}. {name}{tag}</div>')
    st.markdown(f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-bottom:2px;">만드는 순서 — 의심 공정에 불이 들어와요</div>'
                + "".join(rows), unsafe_allow_html=True)


@st.dialog("왜 그런가 — 원리 · 모양 · 검사 결과", width="large")
def _why_dialog(world, sid):
    v = _view(world, sid)
    for k, r in enumerate(v["ranking"][:2]):
        if k == 1:
            st.markdown("---")
        st.markdown(f'<div style="color:{RANK_COLOR[min(r["rank"] - 1, 2)]};font-weight:700;">{_rank_label(r)} · '
                    f'{r["step_name"]}</div><div style="font-size:18px;font-weight:700;margin-bottom:6px;">'
                    f'{r["title"]}</div>', unsafe_allow_html=True)
        if r.get("what"):
            st.markdown(f"**무슨 일이** — {r['what']}")
        if r.get("shape"):
            st.markdown(f"**왜 이 모양** — {r['shape']}")
        if r.get("test"):
            st.markdown(f"**왜 이 검사 결과** — {r['test']}")
        if r.get("source"):
            st.caption(f"근거 — {r['source']}")
        if k == 0 and world == "moscap":
            _cv_plot(v)
    if world == "nmos" and v.get("fingerprint"):
        st.markdown("---")
        st.markdown("**파라미터를 하나씩 의심해 보면** — 이번 WAT 값으로 계산한 크기, 그리고 그 크기면 나와야 할 다른 측정이 "
                    "이번과 맞나. 「정상 범위 밖이고 다른 측정도 맞는」 줄이 이번 일을 설명해요.")
        rows = "".join(
            f'<tr style="{"background:#1E2A22;" if f["explains"] else ""}">'
            f'<td style="padding:4px 8px;">{f["label"]}</td>'
            f'<td style="padding:4px 8px;"><b>{f["value"]}</b> <span style="color:{ui.INK_MUTED};font-size:11.5px;">'
            f'({f["lo"]} ~ {f["hi"]})</span></td>'
            f'<td style="padding:4px 8px;color:{ui.INK_MUTED};">{f["normal"]}</td>'
            f'<td style="padding:4px 8px;">{"밖" if f["out_of_range"] else "안"}</td>'
            f'<td style="padding:4px 8px;">{"✓ 맞음" if f["consistent"] else "✗ " + " · ".join(f["mismatch"])}</td></tr>'
            for f in v["fingerprint"])
        st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                    f'<tr style="color:{ui.INK_MUTED};font-size:11.5px;"><td style="padding:4px 8px;">파라미터</td>'
                    f'<td style="padding:4px 8px;">이번 값으로 계산한 크기 (90%)</td><td style="padding:4px 8px;">정상 범위</td>'
                    f'<td style="padding:4px 8px;">범위</td><td style="padding:4px 8px;">다른 측정과</td></tr>{rows}</table>',
                    unsafe_allow_html=True)
        st.caption("산화막은 C-V 값, 채널 · poly 도핑은 Vt, 계면 트랩은 SS에서 거꾸로 풀었어요. 식은 장채널 MOS 식(교과서)이고, "
                   "같은 식으로 합성한 값이라 범위는 「식이 맞다면」의 상한이에요.")


@st.dialog("정답 공개 — 이 웨이퍼에 넣은 사고", width="large")
def _answer_dialog(world, sid):
    v = _view(world, sid)
    st.markdown(f"**시나리오에서 넣은 사고** — {v['injected']}")
    group = [r for r in v["ranking"] if r["rank"] == v["ranking"][0]["rank"]]
    label = "공동 1순위" if len(group) > 1 else "1순위"
    st.markdown(f"**엔진의 {label}**")
    for r in group:
        st.markdown(f"- {r['step_name']} — {r['suspect']}")
    if v["hit"]:
        st.success("넣은 사고가 " + ("공동 1순위 안에 있어요 — 엔진은 둘을 가르지 못하고, 「먼저 확인할 것」으로 가르라고 해요."
                                  if len(group) > 1 else "엔진의 1순위와 같아요."), icon="✅")
    else:
        st.warning("넣은 사고가 엔진의 1순위에 없어요.", icon="⚠️")
    pl = v.get("planted")
    if pl and pl.get("multi"):
        st.markdown(f"**넣은 크기** — {pl['label']} {pl['value']} (사고 둘을 같이 넣음)")
        if v.get("warning"):
            st.success("엔진이 「파라미터 하나로는 설명이 안 된다」고 경고했어요 — 사고가 겹쳤을 때 맞는 반응이에요. "
                       "1순위의 「얼마나」만 믿었다면 크기를 잘못 말했을 거예요.", icon="✅")
        pl = None
    if pl:
        e = v["magnitude"].get(v["set_cause"])
        st.markdown(f"**넣은 크기** — {pl['label']} {pl['value']} (정상 {pl['normal']})")
        if e:
            st.markdown(f"**엔진이 계산한 크기** — {e['label']} 약 {e['value']} (90% {e['lo']} ~ {e['hi']})")
        if pl.get("inside") is False:
            st.info(f"넣은 크기가 이번엔 90% 범위 밖이에요. 90% 범위는 10번에 1번꼴로 놓치는데(웨이퍼 150장 시험에서 적중 "
                    f"90 ~ 92%), 이번 웨이퍼는 {pl['nuisance']}가 정상 산포의 끝(±{pl['nuisance_z']:.1f}σ)에 있었어요.",
                    icon="🎲")
    if world == "nmos":
        st.info("엔진은 「어느 단계의 · 어느 파라미터가 · 얼마나」(산화막 몇 Å 같은 소자 값)까지 말해요. 「산화로가 몇 °C · 몇 분"
                "이었나」 같은 장비 설정은 같은 두께를 만드는 조합이 여러 개라 정하지 않고, 장비 기록으로 넘겨요.", icon="ℹ️")
    else:
        st.info("엔진은 「어느 단계에서 · 어느 방향으로」까지만 말해요. 「이번에 몇 도였다」 같은 조건값은 맵과 검사 결과에 "
                "들어 있지 않아서, 장비 기록을 봐야 알 수 있어요 — 그래서 엔진 결론과 따로 보여요.", icon="ℹ️")
    st.caption("검사 결과(bin)는 이 사고에서 합성했어요 — WM-811K에는 검사 결과가 없어요.")


def _cv_plot(v):
    dom = max(v["bins"], key=lambda b: v["counts"][b])
    x = np.linspace(-3, 3, 241)
    sig = lambda z, w: 1.0 / (1.0 + np.exp(z / w))        # noqa: E731
    normal = 0.25 + 0.75 * sig(x - 0.2, 0.22)
    rng = np.random.default_rng(0)
    if dom == "open":
        bads, note = [("이번 웨이퍼", 0.02 + 0.01 * rng.standard_normal(x.size))], "전극이 없어 용량이 안 잡혀요"
    elif dom == "leak_bd":
        bad = normal * (1 - 0.65 * sig(x + 1.6, 0.12)) + np.where(x < -1.6, 0.05 * rng.standard_normal(x.size), 0)
        bads, note = [("이번 웨이퍼", bad)], "−1.6 V부터 막이 뚫려 곡선이 무너져요"
    elif dom == "cv_shape":
        bads = [("이번 — 갈 때", 0.25 + 0.75 * sig(x - 0.2, 0.6)), ("이번 — 올 때", 0.25 + 0.75 * sig(x - 0.55, 0.6))]
        note = "완만해지고, 갈 때와 올 때가 어긋나요"
    else:
        bads, note = [("이번 웨이퍼", 0.21 + 0.54 * sig(x - 0.45, 0.22))], "용량이 낮고 곡선이 옆으로 밀려요"
    col = BIN_COLORS[v["bins"].index(dom)]
    fig = go.Figure()
    fig.add_scatter(x=x, y=normal, name="정상 (기준)", line=dict(color=ui.INK_MUTED, dash="dash", width=2))
    for i, (nm, y) in enumerate(bads):
        fig.add_scatter(x=x, y=y, name=nm, line=dict(color=col, width=2.5, dash="solid" if i == 0 else "dot"))
    fig.update_layout(**ui._base_layout(
        height=260, margin=dict(l=40, r=8, t=30, b=70), showlegend=True,
        title=dict(text=f"C-V 곡선 개념도 — {note}", font=dict(size=13)),
        xaxis=dict(title="게이트 전압 (V)"), yaxis=dict(title="C / Cox", range=[-0.05, 1.1]),
        legend=dict(orientation="h", y=-0.38, yanchor="top")))
    st.plotly_chart(fig, width="stretch", config=ui._PLOTLY_CONFIG, key="d16_cv")
    st.caption("개념도예요 — 실제 측정값이 아니라, 이 검사 결과가 곡선에서 어떻게 보이는지 그린 그림이에요.")


def _engine(v):
    """③ 엔진이 말하는 것 — 공동 1위면 같은 크기로 나란히 보인다(어느 쪽을 먼저 쓸지 엔진이 정하지 않았으므로)."""
    rk = v["ranking"]
    top_group = [r for r in rk if r["rank"] == rk[0]["rank"]]
    rest = [r for r in rk if r["rank"] != rk[0]["rank"]][:2]
    col = RANK_COLOR[0]
    blocks = []
    for r in top_group:
        blocks.append(
            f'<div style="color:{col};font-size:13.5px;font-weight:700;margin-top:6px;">{_rank_label(r)} · 공정 「{r["step_name"]}」</div>'
            f'<div style="color:{ui.INK};font-size:{17 if len(top_group) > 1 else 18}px;font-weight:700;line-height:1.35;margin:2px 0 4px;">'
            f'{r["suspect"]}</div>'
            f'<div style="color:{ui.INK_MUTED};font-size:12.5px;">정상이면 — <span style="color:{ui.INK};">{r["normal"]}</span></div>'
            f'<div style="color:{ui.INK_MUTED};font-size:12.5px;margin-top:2px;">먼저 확인할 것 — '
            f'<span style="color:{ui.INK};">{" · ".join(r["checks"])}</span></div>'
            + (_mag_line(r) if v["world"] == "nmos" else ""))
    st.markdown(
        f'<div style="border:1px solid {ui.BORDER};background:{ui.SURFACE};border-radius:10px;padding:8px 14px 10px;">'
        f'<div style="color:{ui.INK_MUTED};font-size:12.5px;">③ 엔진이 말하는 것</div>'
        + f'<div style="border-top:1px dashed {ui.BORDER};margin-top:8px;"></div>'.join(blocks) + '</div>',
        unsafe_allow_html=True)
    st.markdown("".join(
        f'<div style="border-left:3px solid {RANK_COLOR[min(r["rank"] - 1, 2)]};padding:2px 9px;margin:5px 0;font-size:13px;">'
        f'<b style="color:{RANK_COLOR[min(r["rank"] - 1, 2)]};">{_rank_label(r)}</b>'
        f'<span style="color:{ui.INK_MUTED};"> · {r["step_name"]} — </span>'
        f'<span style="color:{ui.INK};">{r["suspect"]}</span></div>' for r in rest), unsafe_allow_html=True)
    if v.get("note"):
        kind, text = v["note"]
        (st.success if kind == "good" else st.info)(text, icon="🎯" if kind == "good" else "🔎")
    if v.get("warning"):
        st.warning(v["warning"], icon="⚠️")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("왜 그런가 ↗", width="stretch", key="d16_why"):
            _why_dialog(v["world"], v["id"])
    with b2:
        if st.button("정답 공개 ↗", width="stretch", key="d16_ans"):
            _answer_dialog(v["world"], v["id"])


def _view(world, sid):
    return {"moscap": _view_moscap, "nmos": _view_nmos}.get(world, _view_cmos)(sid)


# ── 3단계 진행 ───────────────────────────────────
# 계산은 순식간이라 그냥 두면 결과가 툭 나타나 「모델이 무엇을 했는지」가 안 보인다. 단계에 **앞으로 들어올 때
# 한 번만** 모델이 하는 일을 한 줄씩 찍으며 짧게 멈추고(줄당 PAUSE초), 다시 그릴 때는 끝난 상태로 접어 둔다.
# 줄의 내용은 실제로 일어난 계산 순서다 — 꾸민 말이 아니다.

STEPS = ["① 모양 판정 (CNN)", "② 검사 결과", "③ 먼저 볼 공정"]
PAUSE = 0.45


def _stage_lines(v, step):
    n_bad = sum(v["counts"].values())
    if step == 0:
        return "CNN이 모양을 판단하는 중…", [
            "맵을 26×26 칸으로 — 칸마다 합격 / 불합격만 입력",
            "작은 돋보기(3×3)로 훑으며 무늬 찾기 — 합성곱 3층 (32 → 64 → 128가지 무늬)",
            "무늬를 모아 큰 모양으로 — 정상 + 불량 8종, 9종 각각의 확률 계산"]
    if step == 1:
        return "검사 결과를 읽는 중…", [
            (f"불량 {n_bad}칸마다 측정 9가지 중 「가장 크게 벗어난 측정」 읽기" if v["world"] == "nmos" else
             f"불량 {n_bad}칸마다 「처음 걸린 검사」 읽기 (검사는 순서대로, 처음 걸리면 멈춤)"),
            "검사 종류별로 세기" + (f" ({len(v['bins'])}칸)" if v["world"] == "nmos" else ""),
            "한쪽으로 쏠렸나 — 고르게 섞였으면 검사 결과는 버리고 모양만 씀"]
    n_cause = v.get("n_cause") or (14 if v["world"] == "moscap" else 17)
    lines = [f"원인 후보 {n_cause}개 중 이 모양을 만드는 것만 남기기 (문헌·실습 기록)",
             "후보마다 「이 공정이 틀어지면 나올 검사 결과」와 이번 결과를 비교",
             "가장 잘 맞는 순서로 줄 세우기 — 검사 결과로 안 갈리면 공동 순위"]
    if v["world"] == "nmos" and v.get("wat"):
        lines.append("1순위의 파라미터가 얼마나 벗어났나 — WAT 값을 MOS 식으로 거꾸로 계산, 정상 산포로 90% 범위")
    return "먼저 볼 공정을 고르는 중…", lines


def _done_label(v, step):
    if step == 0:
        top, conf, _ = v["cnn"]
        return f"CNN 판정 끝 — {PAT_KO.get(top, top)} {conf:.0f}%"
    if step == 1:
        cnt = v["counts"]
        dom = max(v["bins"], key=lambda b: cnt[b])
        tot = max(sum(cnt.values()), 1)
        return f"검사 결과 끝 — {_most(cnt[dom], tot)} {v['bin_plain'][dom]} ({cnt[dom] / tot * 100:.0f}%)"
    r = v["ranking"][0]
    return f"순위 끝 — {_rank_label(r)} 공정 「{r['step_name']}」"


def _go(step, animate=False):
    st.session_state["d16_step"] = step
    st.session_state["d16_anim"] = step if animate else None
    if animate:
        st.session_state["d16_ran"] = True


def _stage(v, step):
    """진행 상자 — 앞으로 들어온 첫 번만 줄마다 멈추며 펼치고, 끝나면 결과 한 줄로 접는다."""
    animate = st.session_state.get("d16_anim") == step
    label, lines = _stage_lines(v, step)
    with st.status(label if animate else _done_label(v, step), expanded=animate,
                   state="running" if animate else "complete") as box:
        for t in lines:
            st.write(f"· {t}")
            if animate:
                time.sleep(PAUSE)
        box.update(label=_done_label(v, step), state="complete", expanded=False)
    if animate:
        st.session_state["d16_anim"] = None


def _step1(v):
    left, right = st.columns([1.0, 1.1], gap="medium")
    with left:
        _map(v, key="d16_map1", binned=False)
        st.caption(f"WM-811K `{v['lot']}`의 실제 맵 — 패턴 CNN이 학습 때 한 장도 보지 않은 lot이에요. "
                   "지금은 합격 / 불합격만 보여요.")
    with right:
        if not st.session_state.get("d16_ran"):
            st.markdown("#### 이 맵은 어떤 불량 모양인가")
            st.markdown("칸마다 **합격 / 불합격**만 있는 맵이에요. CNN이 불합격 칸이 늘어선 **모양**을 보고 "
                        "9종(정상 + 불량 8종) 중 하나로 판정해요.")
            st.button("🔎 CNN으로 판정", type="primary", key="d16_run", on_click=_go, args=(0, True))
            return
        _stage(v, 0)
        top, conf, nxt = v["cnn"]
        st.markdown(f"#### CNN 판정 — {PAT_KO.get(top, top)} {conf:.1f}%")
        _prob(v, key="d16_prob")
        st.caption("CNN — 합성곱 3층 · lot 단위로 나눈 시험(처음 보는 lot 1,615개)에서 정확도 90.7% · macro F1 0.741")


def _step2(v):
    left, right = st.columns([1.0, 1.1], gap="medium")
    with left:
        _map(v, key="d16_map2")
    with right:
        _stage(v, 1)
        st.markdown(f"#### 불량 {sum(v['counts'].values())}칸은 어느 검사에서 걸렸나")
        _bins(v, key="d16_bins2")
        if v["world"] == "nmos":
            st.caption("칩마다 측정 9가지를 다 재고, **가장 크게 벗어난 측정 하나가 그 칩의 결과**가 돼요. "
                       f"모양이 후보를 좁히고, 이 결과가 그중에서 골라요 — 칸이 {len(v['bins'])}개라 기존 4칸보다 잘게 갈라요.")
            if v.get("wat"):
                st.markdown("#### 웨이퍼 WAT 값 — 「얼마나」를 잴 재료")
                _wat_table(v)
                st.caption(f"scribe line 시험 패턴 — {v['wat'].get('where', '웨이퍼 중앙값')}이에요. 면저항은 SAB로 "
                           "silicide를 막은 패턴과 덮인 패턴을 따로 재요. 칩 결과(위)가 「어느 단계」를 고르고, 이 값이 "
                           "「얼마나」를 말해요. 이번 값은 MOS 식으로 합성했어요.")
        else:
            st.caption("검사는 위에서부터 순서대로 하고, **처음 걸린 검사가 그 칸의 결과**가 돼요. "
                       "모양이 후보를 좁히고, 이 결과가 그중에서 골라요.")


def _step3(v):
    _stage(v, 2)
    left, mid, right = st.columns([1.0, 0.82, 1.18], gap="medium")
    with left:
        top, conf, nxt = v["cnn"]
        st.markdown(f'<div style="font-size:13px;color:{ui.INK_MUTED};">① 모양 — CNN 판정 '
                    f'<b style="color:{ui.INK};font-size:16px;">{PAT_KO.get(top, top)} {conf:.0f}%</b>'
                    f' · 그다음 {" · ".join(f"{PAT_KO.get(p, p)} {x:.0f}%" for p, x in nxt)}</div>',
                    unsafe_allow_html=True)
        _map(v, key="d16_map")
        rule = v.get("bin_rule", "검사 순서대로 처음 걸린 것")
        st.markdown(f'<div style="font-size:13px;color:{ui.INK_MUTED};margin-top:-4px;">② 무엇으로 떨어졌나 '
                    f'<span style="font-size:11.5px;">({rule})</span></div>', unsafe_allow_html=True)
        _bins(v, key="d16_bins", top=4)
        if v["world"] == "nmos" and v.get("wat"):
            _wat_line(v)
    with mid:
        _flow(v)
    with right:
        _engine(v)


# ── 섹션 ─────────────────────────────────────────

def render_demo16():
    ui.section_header(16, "시연 — 맵에서 공정 단계까지", badge="맵 실측 · CNN 실제 · 검사 결과 합성")
    q_world = str(st.query_params.get("w", "moscap")).lower()
    c1, c2 = st.columns([1, 1.6])
    with c1:
        world = st.radio("공정", list(WORLDS), index=list(WORLDS).index(q_world) if q_world in WORLDS else 0,
                         format_func=lambda k: WORLDS[k], horizontal=True, key="d16_world")
    if world in JSON_PATH:
        path = JSON_PATH[world]
        if not os.path.exists(path):
            st.error(f"{path}가 없습니다 — `python -m scripts.make_{world}_scenarios`로 만드세요.")
            return
        items = [(s["id"], s["title"]) for s in (_moscap() if world == "moscap" else _nmos())["scenarios"]]
    else:
        items = [(s["id"], s["title"]) for s in pc.SCENARIOS]
    ids = [i for i, _ in items]
    q_sid = str(st.query_params.get("ms", "")).upper()
    with c2:
        sid = st.selectbox("시연할 웨이퍼", ids, index=ids.index(q_sid) if q_sid in ids else 0, key=f"d16_pick_{world}",
                           format_func=lambda k: f"{k} · {dict(items)[k]}")
    try:
        v = _view(world, sid)
    except Exception as e:
        st.error(f"시나리오를 불러오지 못했습니다: {e}")
        return
    if st.session_state.get("d16_key") != (world, sid):     # 웨이퍼를 바꾸면 ①부터
        try:
            first = "d16_key" not in st.session_state      # 링크 `&step=3`은 처음 열 때만 따른다
            step0 = max(0, min(2, int(st.query_params.get("step", 1)) - 1)) if first else 0
        except (TypeError, ValueError):
            step0 = 0
        st.session_state.update(d16_key=(world, sid), d16_step=step0, d16_anim=None,
                                d16_ran=step0 > 0)
    step = st.session_state["d16_step"]

    st.markdown(f'<div style="color:{ui.INK_MUTED};font-size:13px;margin:-8px 0 4px;">{v["story"]}</div>',
                unsafe_allow_html=True)
    _banner(v)
    # 이전 · 다음은 단계 표시줄과 한 줄에 둔다 — ③은 화면이 길어 아래에 두면 발표 중에 스크롤해야 한다
    b1, mid_, b3 = st.columns([1, 4, 1], vertical_alignment="center")
    with b1:
        st.button("◀ 이전", key="d16_prev", disabled=step == 0, on_click=_go, args=(step - 1,), width="stretch")
    with mid_:
        ui.step_chips(STEPS, step)
    with b3:
        st.button("다음 ▶", key="d16_next", type="primary", width="stretch",
                  disabled=step == 2 or not st.session_state.get("d16_ran"), on_click=_go, args=(step + 1, True))
    if step == 0:
        _step1(v)
    elif step == 1:
        _step2(v)
    else:
        _step3(v)
    ui.source_chips(v["chips"])
