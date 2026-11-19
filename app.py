from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MONTHLY_FILE = DATA_DIR / "monthly_splits.csv"
HALF_FILE = DATA_DIR / "half_splits.csv"
REGISTRY_FILE = DATA_DIR / "player_registry.csv"
PITCH_FILE = DATA_DIR / "statcast_pitch_data.csv"

st.set_page_config(
    page_title="The Anatomy of Dominance | Pitcher Lab",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
      .hero {padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
      .hero-kicker {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .7;}
      .hero-title {font-size: 2.05rem; font-weight: 750; line-height: 1.15; margin: .25rem 0 .4rem 0;}
      .hero-sub {font-size: 1rem; opacity: .8;}
      .note {font-size: .86rem; opacity: .78;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20); padding: .8rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(MONTHLY_FILE)
    half = pd.read_csv(HALF_FILE)
    registry = pd.read_csv(REGISTRY_FILE)
    return monthly, half, registry


def as_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def _swing_mask(description: pd.Series) -> pd.Series:
    # Statcast description values for recognizable batter swings.
    pattern = (
        r"swinging_strike|foul|hit_into_play|missed_bunt|"
        r"bunt_foul|swinging_pitchout"
    )
    return description.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def summarize_pitch_window(raw_window: pd.DataFrame) -> dict[str, float]:
    d = raw_window.copy()
    out: dict[str, float] = {}

    def num(col: str) -> pd.Series:
        if col not in d.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(d[col], errors="coerce")

    out["avg_release_speed"] = _safe_mean(num("release_speed"))
    out["avg_spin"] = _safe_mean(num("release_spin"))
    pfx_x = num("pfx_x").dropna()
    pfx_z = num("pfx_z").dropna()
    out["horizontal_break_in"] = float(pfx_x.abs().mean() * 12) if not pfx_x.empty else math.nan
    out["vertical_movement_in"] = float(pfx_z.mean() * 12) if not pfx_z.empty else math.nan

    zone = num("zone")
    zone = zone.dropna()
    out["zone_rate"] = float(zone.between(1, 9).mean()) if not zone.empty else math.nan

    if "type" in d.columns:
        pitch_type = d["type"].fillna("").astype(str)
        valid = pitch_type.isin(["B", "S", "X"])
        out["strike_result_rate"] = float(pitch_type[valid].eq("S").mean()) if valid.any() else math.nan
    else:
        out["strike_result_rate"] = math.nan

    if "description" in d.columns:
        desc = d["description"].fillna("").astype(str)
        swings = _swing_mask(desc)
        whiffs = desc.str.contains("swinging_strike", case=False, regex=False)
        out["whiff_rate"] = float(whiffs[swings].mean()) if swings.any() else math.nan
    else:
        out["whiff_rate"] = math.nan

    launch = num("launch_speed").dropna()
    out["hard_hit_rate"] = float((launch >= 95).mean()) if not launch.empty else math.nan

    if "bb_type" in d.columns:
        bb = d["bb_type"].dropna().astype(str)
        out["groundball_rate"] = float(bb.eq("ground_ball").mean()) if not bb.empty else math.nan
    else:
        out["groundball_rate"] = math.nan

    if "pitch_type" in d.columns:
        pt = d["pitch_type"].fillna("").astype(str)
        out["fastball_usage"] = float(pt.isin(["FF", "FA", "FT", "SI", "FC"]).mean())
    else:
        out["fastball_usage"] = math.nan

    out["pitch_count"] = float(len(d))
    return out


@st.cache_data(show_spinner=False)
def build_pitch_windows(raw: pd.DataFrame, appearances_per_window: int = 3) -> pd.DataFrame:
    """Build overlapping 3-appearance windows and label each pitcher's extremes.

    Performance is defined by wOBA allowed on completed plate appearances.
    Within each pitcher, the lowest quartile of window wOBA is '호조' and
    the highest quartile is '부진'. This is an exploratory within-player label.
    """
    if raw.empty:
        return pd.DataFrame()

    d = raw.copy()
    player_col = "project_player" if "project_player" in d.columns else "player_name"
    required = {player_col, "game_pk", "game_date"}
    if not required.issubset(d.columns):
        return pd.DataFrame()

    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d.dropna(subset=[player_col, "game_pk", "game_date"]).copy()
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).eq("R")].copy()
    if "game_year" in d.columns:
        d = d[pd.to_numeric(d["game_year"], errors="coerce").eq(2025)].copy()
    if d.empty:
        return pd.DataFrame()

    d["game_pk"] = d["game_pk"].astype(str)
    if not {"woba_value", "woba_denom", "events"}.issubset(d.columns):
        return pd.DataFrame()
    d["_woba_value"] = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0)
    d["_woba_denom"] = pd.to_numeric(d["woba_denom"], errors="coerce").fillna(0)

    # Each pitcher's appearance/game, with plate-appearance-level wOBA allowed.
    game_rows = []
    for (player, game_pk, game_date), grp in d.groupby([player_col, "game_pk", "game_date"], sort=True):
        if "inning" in grp.columns:
            first_inning = pd.to_numeric(grp["inning"], errors="coerce").min()
            if pd.notna(first_inning) and first_inning != 1:
                continue
        pa = grp[grp["events"].notna()].copy()
        denom = float(pa["_woba_denom"].sum()) if not pa.empty else 0.0
        num = float(pa["_woba_value"].sum()) if not pa.empty else 0.0
        pitch_count = len(grp)
        # Exclude very short relief appearances; the chosen case studies are starters.
        if pitch_count < 25 or denom < 12:
            continue
        game_rows.append({
            "player": str(player),
            "game_pk": game_pk,
            "game_date": game_date,
            "pitch_count": pitch_count,
            "pa_count": denom,
            "woba_against": num / denom if denom else math.nan,
        })

    games = pd.DataFrame(game_rows)
    if games.empty:
        return pd.DataFrame()

    windows: list[dict] = []
    for player, pg in games.groupby("player"):
        pg = pg.sort_values("game_date").reset_index(drop=True)
        if len(pg) < appearances_per_window + 3:
            continue
        for start in range(0, len(pg) - appearances_per_window + 1):
            section = pg.iloc[start:start + appearances_per_window]
            pa_total = float(section["pa_count"].sum())
            if pa_total < 45:
                continue
            ids = section["game_pk"].tolist()
            rw = d[(d[player_col].astype(str) == str(player)) & d["game_pk"].isin(ids)]
            pa = rw[rw["events"].notna()] if "events" in rw.columns else rw.iloc[0:0]
            den = float(pa["_woba_denom"].sum()) if not pa.empty else 0.0
            num = float(pa["_woba_value"].sum()) if not pa.empty else 0.0
            if den <= 0:
                continue
            stats = summarize_pitch_window(rw)
            windows.append({
                "player": str(player),
                "start_date": section["game_date"].min(),
                "end_date": section["game_date"].max(),
                "game_count": len(section),
                "pa_count": den,
                "woba_against": num / den,
                **stats,
            })

    result = pd.DataFrame(windows)
    if result.empty:
        return result

    result["phase"] = "중간"
    for player, idx in result.groupby("player").groups.items():
        vals = result.loc[idx, "woba_against"]
        if len(vals) < 6:
            continue
        ranks = vals.rank(method="average", pct=True)
        result.loc[idx[ranks <= 0.25], "phase"] = "호조"
        result.loc[idx[ranks >= 0.75], "phase"] = "부진"
    return result


def make_phase_comparison(windows: pd.DataFrame, player_filter: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if windows.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = windows.copy()
    if player_filter and player_filter != "전체 선수":
        data = data[data["player"].eq(player_filter)]
    phases = data[data["phase"].isin(["호조", "부진"])]
    metrics = [
        "avg_release_speed", "avg_spin", "horizontal_break_in", "vertical_movement_in",
        "zone_rate", "strike_result_rate", "whiff_rate", "hard_hit_rate",
        "groundball_rate", "fastball_usage", "woba_against",
    ]
    long = phases.melt(
        id_vars=["player", "phase"],
        value_vars=[m for m in metrics if m in phases.columns],
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    avg = long.groupby(["player", "phase", "metric"], as_index=False)["value"].mean()
    pivot = avg.pivot_table(index=["player", "metric"], columns="phase", values="value").reset_index()
    if "호조" in pivot.columns and "부진" in pivot.columns:
        pivot["호조-부진"] = pivot["호조"] - pivot["부진"]
    else:
        pivot["호조-부진"] = math.nan
    return avg, pivot


monthly, half, registry = load_reference_data()
monthly = as_numeric(monthly, ["month_order", "ip_decimal", "era"])
half = as_numeric(half, ["era", "avg", "obp", "slg", "woba"])

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">MLB Statcast · 2025 Regular Season</div>
      <div class="hero-title">The Anatomy of Dominance</div>
      <div class="hero-sub">잘 던질 때와 부진할 때, 투수의 투구 데이터는 무엇이 달라질까?</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "이 대시보드는 먼저 공개된 월별·전후반기 성적 자료로 전환 사례를 확인하고, "
    "Statcast 투구별 원자료를 불러오면 각 투수의 호조·부진 구간에서 구속·회전·무브먼트 등을 비교합니다."
)

with st.sidebar:
    st.header("분석 조건")
    available_teams = sorted(monthly["team"].dropna().unique().tolist())
    team_selection = st.multiselect("팀", available_teams, default=available_teams, format_func=lambda x: {"PHI":"필라델피아","ARI":"애리조나"}.get(x, x))
    available_players = sorted(monthly[monthly["team"].isin(team_selection)]["player"].unique().tolist()) if team_selection else []
    player_selection = st.multiselect("투수", available_players, default=available_players)
    st.divider()
    st.caption("분석 시즌: 2025 정규시즌")
    st.caption("성과 분할 자료: CBS Sports / FanGraphs")
    st.caption("투구 단위 자료: MLB Baseball Savant Statcast")

filtered_monthly = monthly[monthly["team"].isin(team_selection) & monthly["player"].isin(player_selection)]
filtered_half = half[half["team"].isin(team_selection) & half["player"].isin(player_selection)]

# Candidate summary metrics based on published first/second-half splits.
if not filtered_half.empty:
    era_diff = filtered_half.pivot_table(index="player", columns="period", values="era")
    improved_count = int(((era_diff.get("후반기", pd.Series(dtype=float)) < era_diff.get("전반기", pd.Series(dtype=float)))).sum()) if {"전반기", "후반기"}.issubset(era_diff.columns) else 0
else:
    improved_count = 0

m1, m2, m3 = st.columns(3)
m1.metric("사례 투수", f"{len(set(filtered_monthly['player']).union(set(filtered_half['player'])))}명")
m2.metric("후반기 ERA 개선 사례", f"{improved_count}명")
m3.metric("투구 단위 원자료", "다운로드 전" if not PITCH_FILE.exists() else f"{PITCH_FILE.stat().st_size / 1024 / 1024:.1f} MB")

st.subheader("1. 먼저 확인한 성적 전환 사례")
st.caption("아래 월별 ERA는 실제 시즌 분할 통계입니다. ERA가 투구 특성의 원인을 증명하지는 않으며, 분석 대상 선수를 찾는 용도로 사용합니다.")

if filtered_monthly.empty:
    st.info("왼쪽 사이드바에서 팀과 투수를 선택해 주세요.")
else:
    fig_month = px.line(
        filtered_monthly.sort_values("month_order"),
        x="month", y="era", color="player", markers=True,
        hover_data={"ip_display": True, "source_name": True, "month_order": False, "era": ":.2f"},
        labels={"month":"월","era":"월별 ERA","player":"투수","ip_display":"이닝","source_name":"출처"},
        title="월별 평균자책점 변화 (낮을수록 좋음)",
    )
    fig_month.update_layout(legend_title_text="투수", height=420, hovermode="x unified")
    st.plotly_chart(fig_month, use_container_width=True)

if not filtered_half.empty:
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        fig_half = px.bar(
            filtered_half,
            x="player", y="era", color="period", barmode="group",
            text="era", hover_data=["ip_display", "woba"],
            labels={"player":"투수","era":"ERA","period":"기간","ip_display":"이닝","woba":"상대 wOBA"},
            title="전반기와 후반기 성적 비교",
        )
        fig_half.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_half.update_layout(height=390, legend_title_text="기간")
        st.plotly_chart(fig_half, use_container_width=True)
    with col_b:
        woba_available = filtered_half.dropna(subset=["woba"])
        if not woba_available.empty:
            fig_woba = px.bar(
                woba_available, x="player", y="woba", color="period", barmode="group",
                text="woba", labels={"player":"투수","woba":"상대 wOBA","period":"기간"},
                title="상대 타자의 wOBA 변화",
            )
            fig_woba.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_woba.update_layout(height=390, legend_title_text="기간")
            st.plotly_chart(fig_woba, use_container_width=True)
        else:
            st.info("선택한 투수의 전·후반기 wOBA 자료가 없습니다.")

st.markdown("#### 선정한 사례")
case_rows = [
    {"투수":"Jesús Luzardo", "팀":"PHI", "관찰된 전환":"4월 호조 → 5~7월 부진 → 8~9월 회복", "검증할 점":"직구 구속·구종 비율·타자 상대 결과가 구간별로 달라졌는가?"},
    {"투수":"Eduardo Rodríguez", "팀":"ARI", "관찰된 전환":"전반기 부진 → 후반기 개선", "검증할 점":"상대 wOBA 감소와 함께 투구 특성도 달라졌는가?"},
    {"투수":"Zac Gallen", "팀":"ARI", "관찰된 전환":"전반기 부진 → 후반기 개선", "검증할 점":"후반기 피안타 질과 구종별 성과가 개선됐는가?"},
]
case_df = pd.DataFrame(case_rows)
case_df = case_df[case_df["팀"].isin(team_selection)]
st.dataframe(case_df, use_container_width=True, hide_index=True)

with st.expander("집계 기준과 데이터 출처 보기"):
    st.markdown(
        """
        - **시즌:** 2025년 MLB 정규시즌.
        - **월별 ERA:** 각 출처의 월별 투수 성적 분할 표에 게시된 값. 이닝은 별도 표기하여 표본 크기를 함께 확인합니다.
        - **전·후반기:** 올스타 브레이크를 기준으로 FanGraphs의 split 기록을 사용합니다.
        - **해석:** 월별 ERA는 결과 지표입니다. 구속·회전수·무브먼트가 성적 변화의 원인이라고 말하려면 Statcast 투구별 자료를 별도로 비교해야 합니다.
        """
    )
    st.markdown("- Jesús Luzardo: [CBS Sports 2025 splits](https://www.cbssports.com/mlb/players/2910624/jesus-luzardo/splits/2025/), [MLB.com 2025 시즌 평가](https://www.mlb.com/)")
    st.markdown("- Eduardo Rodríguez: [FanGraphs 2025 pitching splits](https://www.fangraphs.com/players/eduardo-rodriguez/13164/splits?position=P&season=2025&split=)")
    st.markdown("- Zac Gallen: [FanGraphs 2025 pitching splits](https://www.fangraphs.com/players/zac-gallen/19291/splits?position=P&season=2025&split=)")
    st.markdown("- Statcast 변수 정의: [Baseball Savant CSV Documentation](https://baseballsavant.mlb.com/csv-docs)")

st.divider()
st.subheader("2. Statcast 투구 단위 분석")
st.write("여기부터는 월별 ERA가 아니라 **투구 하나당 한 행**인 원자료를 분석합니다. 선수별 연속 3경기 구간을 만들고, 본인 시즌 내 상대 wOBA가 낮은 구간을 호조, 높은 구간을 부진으로 분류합니다.")

if not PITCH_FILE.exists():
    st.info(
        "현재 프로젝트 폴더에는 투구 단위 CSV가 아직 없습니다. 아래 버튼으로 Baseball Savant 원자료를 가져오면 "
        "구속·회전수·무브먼트 비교 차트가 활성화됩니다. 인터넷 연결이 필요합니다."
    )
    st.code("pip install -r requirements.txt\npython fetch_data.py", language="bash")
    if st.button("Statcast 투구 데이터 가져오기", type="primary"):
        try:
            from fetch_data import download_all_pitch_data
            with st.spinner("선수별 2025년 투구 데이터를 Baseball Savant에서 가져오는 중입니다..."):
                fetched = download_all_pitch_data()
            st.success(f"{len(fetched):,}개 투구 행을 저장했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"데이터를 가져오지 못했습니다: {exc}")
            st.caption("Baseball Savant 접속 제한 또는 네트워크 상태에 따라 요청이 실패할 수 있습니다. 로컬에서 `python fetch_data.py`를 실행해 다시 시도해 주세요.")
else:
    try:
        raw = pd.read_csv(PITCH_FILE, low_memory=False)
        if "project_player" in raw.columns:
            raw_players = sorted(raw["project_player"].dropna().unique().tolist())
        else:
            raw_players = sorted(raw["player_name"].dropna().unique().tolist()) if "player_name" in raw.columns else []
        pitch_player = st.selectbox("Statcast 분석 투수", ["전체 선수"] + raw_players)
        with st.spinner("투수별 3경기 구간을 계산하고 있습니다..."):
            windows = build_pitch_windows(raw)
        if windows.empty:
            st.warning("유효한 3경기 분석 구간이 충분하지 않습니다. 원자료의 events, woba_value, woba_denom 및 게임 식별 열을 확인해 주세요.")
        else:
            selected_windows = windows if pitch_player == "전체 선수" else windows[windows["player"].eq(pitch_player)]
            hot_bad = selected_windows[selected_windows["phase"].isin(["호조", "부진"])]
            feature_labels = {
                "avg_release_speed":"평균 구속 (mph)",
                "avg_spin":"평균 회전수 (rpm)",
                "horizontal_break_in":"평균 수평 무브먼트 크기 (in)",
                "vertical_movement_in":"평균 수직 무브먼트 (in)",
                "zone_rate":"존 안 투구 비율",
                "strike_result_rate":"Statcast S 결과 비율",
                "whiff_rate":"헛스윙 / 스윙 비율",
                "hard_hit_rate":"하드히트 비율 (95mph+)",
                "groundball_rate":"땅볼 비율",
                "fastball_usage":"패스트볼 사용 비율",
            }
            selected_metric = st.selectbox("비교할 투구 지표", list(feature_labels.keys()), format_func=lambda x: feature_labels[x])
            phase_means = hot_bad.groupby("phase")[selected_metric].mean().reindex(["호조", "부진"])
            k1, k2, k3 = st.columns(3)
            k1.metric("분석 구간", f"{len(selected_windows):,}개")
            k2.metric("호조 구간", f"{int(selected_windows['phase'].eq('호조').sum()):,}개")
            k3.metric("부진 구간", f"{int(selected_windows['phase'].eq('부진').sum()):,}개")

            c1, c2 = st.columns(2)
            with c1:
                phase_plot = phase_means.rename_axis("phase").reset_index(name="value")
                fig_phase = px.bar(phase_plot, x="phase", y="value", color="phase", text="value", title=f"호조·부진 구간의 {feature_labels[selected_metric]}", labels={"phase":"성과 구간","value":feature_labels[selected_metric]})
                fig_phase.update_layout(showlegend=False, height=390)
                st.plotly_chart(fig_phase, use_container_width=True)
            with c2:
                player_phase = hot_bad.groupby(["player", "phase"], as_index=False)[selected_metric].mean()
                if not player_phase.empty:
                    fig_player = px.bar(player_phase, x="player", y=selected_metric, color="phase", barmode="group", title=f"투수별 {feature_labels[selected_metric]} 비교", labels={"player":"투수","phase":"성과 구간",selected_metric:feature_labels[selected_metric]})
                    fig_player.update_layout(height=390, legend_title_text="성과 구간")
                    st.plotly_chart(fig_player, use_container_width=True)

            # Per-pitcher paired deltas avoid giving more pitches from one player extra weight.
            feature_rows = []
            for feature, label in feature_labels.items():
                piv = hot_bad.groupby(["player", "phase"])[feature].mean().unstack()
                if not {"호조", "부진"}.issubset(piv.columns):
                    continue
                delta = (piv["호조"] - piv["부진"]).dropna()
                if not delta.empty:
                    feature_rows.append({
                        "지표": label,
                        "선수별 차이 평균 (호조-부진)": delta.mean(),
                        "호조에서 증가한 선수 비율": float((delta > 0).mean()),
                        "비교 가능한 선수 수": int(delta.size),
                    })
            if feature_rows:
                st.markdown("#### 선수별 차이를 먼저 계산한 뒤 평균한 결과")
                st.dataframe(pd.DataFrame(feature_rows).round(3), use_container_width=True, hide_index=True)
            st.markdown("#### 분석 구간 상세")
            show_cols = ["player", "start_date", "end_date", "game_count", "pa_count", "woba_against", "phase", "pitch_count", selected_metric]
            detail = selected_windows[show_cols].copy()
            detail["start_date"] = pd.to_datetime(detail["start_date"]).dt.strftime("%Y-%m-%d")
            detail["end_date"] = pd.to_datetime(detail["end_date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(detail.sort_values(["player", "start_date"]), use_container_width=True, hide_index=True)
            st.caption("주의: 3경기 이동 구간은 서로 겹칩니다. 호조·부진은 선수 개인의 2025시즌 내 상대적 성과 구분이며, 인과관계나 절대적인 실력 등급이 아닙니다.")
    except Exception as exc:
        st.error(f"Statcast 분석 중 오류가 발생했습니다: {exc}")

st.divider()
st.caption("데이터 출처: MLB Baseball Savant Statcast, FanGraphs, CBS Sports. 이 대시보드는 탐색적 분석용이며 선수의 부진 원인을 인과적으로 판정하지 않습니다.")
