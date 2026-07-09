import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = "artifacts"

st.set_page_config(
    page_title="Analityka cen diamentów",
    page_icon="\u25C8",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        -bg: #F7F8FA;
        -bg-elevated: #FFFFFF;
        -card-bg: #FFFFFF;
        -card-border: rgba(30, 40, 70, 0.10);
        -text-primary: #1B2130;
        -text-secondary: #666F84;
        -accent: #33618A;
        -accent-2: #6A5A9E;
        -accent-gold: #92702F;
        -positive: #2F7A4E;
        -negative: #A23B44;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(-text-primary); color-scheme: light; }
    .stApp { background: var(-bg); }
    section[data-testid="stSidebar"] { display: none; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    .stAppDeployButton { display: none !important; }
    a[href*="github.com/streamlit"] { display: none !important; }

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(-text-secondary) !important;
        opacity: 1 !important;
    }

    /* Wymuszenie jasnego kontrastu na WSZYSTKICH natywnych etykietach Streamlit -
       zabezpieczenie na wypadek, gdyby przegladarka odwiedzajacego zglaszala
       preferencje ciemnego motywu systemowego, ktora Streamlit moglby probowac
       zastosowac do wewnetrznych elementow (etykiety, opisy) niezaleznie od
       naszych zmiennych CSS powyzej. */
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
        color: var(-text-secondary) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(-text-secondary) !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: var(-text-primary) !important;
        opacity: 1 !important;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    label, .stRadio label, .stSlider label, .stSelectSlider label, .stTextInput label {
        color: var(-text-primary) !important;
        opacity: 1 !important;
    }
    div[data-testid="stWidgetLabel"] p {
        color: var(-text-primary) !important;
        opacity: 1 !important;
    }
    .stDataFrame, .stDataFrame * { color: var(-text-primary) !important; }

    .dv-hero {
        background: var(-bg-elevated);
        border: 1px solid var(-card-border);
        border-left: 3px solid var(-accent-gold);
        border-radius: 4px;
        padding: 1.7rem 2.1rem;
        margin-bottom: 1.3rem;
    }
    .dv-hero-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.85rem;
        color: var(-text-primary);
        margin: 0;
        line-height: 1.25;
    }
    .dv-hero-tag {
        font-size: 0.96rem;
        color: var(-text-secondary);
        margin: 0.55rem 0 0 0;
        max-width: 760px;
        line-height: 1.55;
    }

    .stButton button { border-radius: 3px !important; }

    div[data-testid="stMetric"] {
        background: var(-card-bg);
        border: 1px solid var(-card-border);
        border-radius: 4px;
        padding: 0.85rem 1rem 0.65rem 1rem;
    }
    div[data-testid="stMetricLabel"] { font-weight: 500; color: var(-text-secondary) !important; font-size: 0.82rem !important; }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        color: var(-text-primary);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
        background: var(-card-bg);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(-card-border) !important;
        border-radius: 4px !important;
        background: var(-card-bg);
    }

    .dv-section-num {
        display: inline-block;
        color: var(-accent-gold);
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-bottom: 0.1rem;
    }
    .dv-section-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: var(-text-primary);
        margin-bottom: 0.35rem;
    }
    .dv-story {
        color: var(-text-primary);
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 0.9rem;
    }
    .dv-explain { color: var(-text-secondary); font-size: 0.89rem; line-height: 1.55; margin-bottom: 0.6rem; }

    .dv-glossary {
        background: rgba(146,112,47,0.06);
        border: 1px solid rgba(146,112,47,0.25);
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.7rem 0 1rem 0;
    }
    .dv-glossary-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: var(-accent-gold);
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 0.4rem;
    }
    .dv-glossary dt { font-weight: 700; color: var(-text-primary); display: inline; }
    .dv-glossary dd { display: inline; margin: 0; color: var(-text-secondary); }
    .dv-glossary p { margin: 0.25rem 0; font-size: 0.88rem; line-height: 1.5; }

    .badge {
        padding: 3px 12px; border-radius: 3px; font-weight: 600; font-size: 0.74rem;
        display: inline-block; letter-spacing: 0.2px;
    }
    .badge-over { background: rgba(162,59,68,0.12); color: var(-negative); border: 1px solid rgba(162,59,68,0.3); }
    .badge-under { background: rgba(47,122,78,0.12); color: var(-positive); border: 1px solid rgba(47,122,78,0.3); }
    .badge-normal { background: rgba(100,110,140,0.10); color: var(-text-secondary); border: 1px solid var(-card-border); }

    .dv-price-value {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.7rem;
        color: var(-accent-gold);
        margin: 0.2rem 0;
    }
    .dv-price-label {
        color: var(-text-secondary);
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        font-size: 0.78rem;
    }

    .dv-footer {
        border-top: 1px solid var(-card-border);
        margin-top: 2rem;
        padding: 1.2rem 0 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .dv-footer-brand {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(-text-primary);
        font-size: 0.95rem;
    }
    .dv-footer-meta { color: var(-text-secondary); font-size: 0.82rem; }
    .dv-footer-meta a { color: var(-accent-gold); text-decoration: none; }
    .dv-footer-meta a:hover { text-decoration: underline; }

    @media (max-width: 640px) {
        .dv-hero-title { font-size: 1.35rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(name):
    with open(os.path.join(ART, name), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(ART, name))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ART, "model.pkl"))


eda = load_json("eda_overview.json")
model_metrics = load_json("model_metrics.json")
scale_params = load_json("scale_params.json")
encoding_maps = load_json("encoding_maps.json")
calc_defaults = load_json("calc_defaults.json")
feature_cols_model = joblib.load(os.path.join(ART, "feature_cols_model.pkl"))
model = load_model()

df_sample = load_csv("df_sample.csv")
anomaly_summary = load_csv("anomaly_summary.csv")
pca_scree = load_csv("pca_scree.csv")
pca_2d = load_csv("pca_2d.csv")
tsne_2d = load_csv("tsne_2d.csv")
predictions = load_csv("predictions.csv")
feature_importance = load_csv("feature_importance.csv")
diamonds_search = load_csv("diamonds_search.csv")

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]

ACCENT = "#33618A"
ACCENT2 = "#6A5A9E"
GOLD = "#92702F"
NEGATIVE = "#A23B44"
TEXT_SEC = "#666F84"
PLOT_BG = "#FFFFFF"
FONT_COLOR = "#1B2130"
GRID_COLOR = "rgba(30,40,70,0.09)"


def style_fig(fig, title=None):
    layout_kwargs = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Inter"),
        margin=dict(t=45 if title else 15, l=10, r=10, b=10),
        legend=dict(bgcolor=PLOT_BG),
    )
    if title:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def section(num, title):
    st.markdown(f"<div class='dv-section-num'>{num}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dv-section-title'>{title}</div>", unsafe_allow_html=True)


def story(text):
    st.markdown(f"<div class='dv-story'>{text}</div>", unsafe_allow_html=True)


def explain(text):
    st.markdown(f"<div class='dv-explain'>{text}</div>", unsafe_allow_html=True)


def glossary(title, items):
    rows = "".join(f"<p><dt>{k}</dt> - <dd>{v}</dd></p>" for k, v in items)
    st.markdown(
        f"<div class='dv-glossary'><div class='dv-glossary-title'>{title}</div>{rows}</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="dv-hero">
        <div class="dv-hero-title">Analityka cen diamentów na podstawie ich jakości, wagi oraz wymiarów</div>
        <p class="dv-hero-tag">Pełny potok przetwarzania danych - od surowego zbioru rynkowego, przez jego
        diagnozę i naprawę, redukcję wymiarowości i detekcję anomalii cenowych, po model wyceniający
        i interaktywny kalkulator.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    section("01", "Cel i źródło danych")
    story(
        "Celem analizy jest sprawdzenie, na ile cena diamentu daje się wyjaśnić jego mierzalnymi "
        "cechami - wagą, szlifem, kolorem, czystością i wymiarami fizycznymi - oraz zbudowanie na tej "
        "podstawie modelu wyceniającego."
    )
    story(
        "Źródłem jest prawdziwy, publicznie dostępny zbiór danych rynkowych o diamentach (biblioteka "
        "ggplot2/tidyverse)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Diamentów w zbiorze", f"{eda['n_rows']:,}".replace(",", " "))
    c2.metric("R² finalnego modelu", f"{model_metrics['r2']:.3f}")
    c3.metric("Korelacja waga↔cena", f"{eda['corr_price_carat']:.3f}")
    st.caption("Źródło danych: [github.com/tidyverse/ggplot2](https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv)")

st.write("")

with st.container(border=True):
    section("02", "Zbiór danych")
    story(
        "Poniższe liczby pochodzą wprost z artefaktów zapisanych przez notebook przetwarzania danych."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liczba diamentów", f"{eda['n_rows']:,}".replace(",", " "))
    c2.metric("Cech po przetworzeniu", eda["n_cols_final"])
    c3.metric("Przewartościowane", f"{eda['n_overpriced']:,}".replace(",", " "))
    c4.metric("Niedowartościowane (okazje)", f"{eda['n_undervalued']:,}".replace(",", " "))
    explain(
        f"Korelacja wagi (carat) z ceną wynosi **{eda['corr_price_carat']:.3f}** - silna, realna "
        "zależność rynkowa, nie losowy szum."
    )

    glossary(
        "Słowniczek pojęć",
        [
            ("Carat (waga)", "jednostka masy kamieni szlachetnych. 1 karat = 0,2 grama = 200 miligramów."),
            ("Table", "największa, płaska faseta na szczycie oszlifowanego diamentu. Wartość w danych to procent - stosunek szerokości tej fasety do całkowitej średnicy kamienia."),
            ("Depth (głębokość)", "wysokość diamentu (od fasety table do najniższego punktu) wyrażona jako procent średniej średnicy kamienia."),
            ("x, y, z", "wymiary fizyczne diamentu w milimetrach - długość, szerokość i wysokość."),
        ],
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Liczba diamentów wg szlifu**")
        cut_counts = df_sample["cut"].value_counts().reset_index()
        cut_counts.columns = ["cut", "count"]
        fig = px.bar(cut_counts, x="cut", y="count", category_orders={"cut": CUT_ORDER},
                     color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col_r:
        st.markdown("**Liczba diamentów wg czystości**")
        clarity_counts = df_sample["clarity"].value_counts().reset_index()
        clarity_counts.columns = ["clarity", "count"]
        fig = px.bar(clarity_counts, x="clarity", y="count", category_orders={"clarity": CLARITY_ORDER},
                     color_discrete_sequence=[ACCENT2])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("**Podgląd danych po pełnym przetworzeniu**")
    explain("Kolumny takie jak delta_price_vs_clarity czy price_residual_pct to cechy dodane w kolejnych sekcjach.")
    st.dataframe(df_sample.head(15), use_container_width=True)

st.write("")

with st.container(border=True):
    section("03", "Czyszczenie i integracja")
    story(
        "Zbiór źródłowy nie zawiera braków danych ani niespójnych formatów zapisu - kategorie są "
        "zapisane spójnie od początku. Dokładniejsza analiza ujawnia jednak trzy problemy, które są "
        "naturalnie obecne w oryginalnym pliku."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Pełne duplikaty wierszy", f"{eda['n_duplicates_removed']:,}".replace(",", " "))
    c2.metric("Zerowe wymiary fizyczne", eda["n_zero_dimension_rows"])
    c3.metric("Niespójność pomiaru głębokości", eda["n_inconsistent_measurement_rows"])
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown("<div class='dv-explain'>Identyczne rekordy we wszystkich kolumnach.</div>", unsafe_allow_html=True)
    cc2.markdown("<div class='dv-explain'>Wymiar x, y lub z równy zero jest fizycznie niemożliwy - diament nie może mieć zerowej długości.</div>", unsafe_allow_html=True)
    cc3.markdown("<div class='dv-explain'>Podana głębokość nie zgadza się z przeliczoną z x/y/z o ponad 5 punktów procentowych - ślad błędu transkrypcji, np. przestawionego przecinka.</div>", unsafe_allow_html=True)
    st.divider()
    story(
        "Wszystkie trzy problemy potraktowano jako brakujące dane i uzupełniono medianą liczoną w "
        "obrębie tego samego szlifu (cut) - różne szlify mają różne typowe proporcje geometryczne, "
        "więc mediana grupowa jest trafniejsza niż globalna. Mediana ta została policzona wyłącznie na "
        "zbiorze treningowym, aby uniknąć wycieku informacji ze zbioru testowego."
    )

    st.markdown("**Integracja: cena diamentu na tle jego grupy jakościowej**")
    explain(
        "Tabele podsumowujące (średnia cena wg czystości i wg koloru) są dołączane do tabeli głównej - "
        "powstają cechy pokazujące odchylenie ceny konkretnego diamentu od średniej dla jego grupy."
    )
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(df_sample, x="delta_price_vs_clarity", nbins=50, color_discrete_sequence=[ACCENT])
        fig.add_vline(x=0, line_dash="dash", line_color=TEXT_SEC)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col_r:
        fig = px.histogram(df_sample, x="delta_price_vs_color", nbins=50, color_discrete_sequence=[ACCENT2])
        fig.add_vline(x=0, line_dash="dash", line_color=TEXT_SEC)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.markdown("**Normalizacja: wspólna skala dla modelu**")
    story(
        "Waga diamentu mieści się w przedziale 0,2–5,0 karata, a table w przedziale 43–95% - dla "
        "modelu to dwie zupełnie różne skale. RobustScaler (mediana i rozstęp międzykwartylowy) "
        "sprowadza je do porównywalnego zakresu, zachowując odporność na wartości odstające. Parametry "
        "skalowania, podobnie jak przy imputacji, pochodzą wyłącznie ze zbioru treningowego."
    )
    choice = st.radio("Widok:", ["Przed normalizacją", "Po normalizacji"], horizontal=True)
    if choice == "Przed normalizacją":
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_sample["carat"], name="carat", opacity=0.8, marker_color=ACCENT))
        fig.add_trace(go.Histogram(x=df_sample["table"], name="table", opacity=0.8, marker_color=GOLD))
        fig.update_layout(barmode="overlay")
    else:
        carat_scaled = (df_sample["carat"] - scale_params["carat"]["center"]) / scale_params["carat"]["scale"]
        table_scaled = (df_sample["table"] - scale_params["table"]["center"]) / scale_params["table"]["scale"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=carat_scaled, name="carat", opacity=0.8, marker_color=ACCENT))
        fig.add_trace(go.Histogram(x=table_scaled, name="table", opacity=0.8, marker_color=GOLD))
        fig.update_layout(barmode="overlay")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    explain("Przypomnienie: carat to waga (1 ct = 0,2 g), table to procentowa szerokość górnej fasety względem średnicy kamienia - patrz słowniczek w sekcji 02.")

st.write("")

with st.container(border=True):
    section("04", "Anomalie cenowe")
    story(
        "Skoro waga silnie determinuje cenę, prosty model bazowy cena ≈ a × carat^b (model potęgowy, "
        "standardowy w wycenie diamentów i z definicji zawsze dodatni) pozwala wskazać diamenty, "
        "których rzeczywista cena istotnie odbiega od tego, co przewiduje sama waga - przepłacone albo "
        "okazje."
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Najbardziej przewartościowane**")
        top_over = df_sample[df_sample["Is_Overpriced"]].nlargest(10, "price_residual_pct")
        st.dataframe(top_over[["carat", "cut", "clarity", "price", "price_residual_pct"]],
                     use_container_width=True, hide_index=True)
    with col_r:
        st.markdown("**Największe okazje**")
        top_under = df_sample[df_sample["Is_Undervalued"]].nsmallest(10, "price_residual_pct")
        st.dataframe(top_under[["carat", "cut", "clarity", "price", "price_residual_pct"]],
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Metody nienadzorowane a reguła bazowa**")
    explain(
        "Isolation Forest i Local Outlier Factor wykrywają anomalie samodzielnie, bez znajomości "
        "reguły opartej na reszcie modelu bazowego. Tabela pokazuje zgodność obu metod z tą regułą."
    )
    st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)
    fig = px.bar(
        anomaly_summary.melt(id_vars="method", value_vars=["precision", "recall"]),
        x="method", y="value", color="variable", barmode="group",
        color_discrete_sequence=[ACCENT, GOLD],
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.markdown("**Próg przewartościowania na żywo**")
    explain("Suwak zmienia liczbę diamentów uznanych za przewartościowane w zależności od przyjętego progu percentylowego reszty ceny.")
    pct = st.slider("Percentyl reszty ceny", min_value=80, max_value=99, value=95, step=1)
    live_th = df_sample["price_residual_pct"].quantile(pct / 100)
    df_sample["is_over_live"] = df_sample["price_residual_pct"] > live_th

    fig = px.scatter(
        df_sample, x="carat", y="price",
        color=df_sample["is_over_live"].map({True: "Przewartościowany", False: "Cena rynkowa"}),
        color_discrete_map={"Przewartościowany": NEGATIVE, "Cena rynkowa": ACCENT},
        opacity=0.65, labels={"carat": "carat", "price": "price"}, render_mode="svg",
    )
    st.plotly_chart(style_fig(fig, "Waga a cena - anomalie na żywo"), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Diamentów powyżej progu", int(df_sample["is_over_live"].sum()))
    avg_overpay = (df_sample.loc[df_sample["is_over_live"], "price"] -
                   df_sample.loc[df_sample["is_over_live"], "price_baseline_pred"]).mean()
    c2.metric("Średnia nadpłata", f"${avg_overpay:,.0f}")

st.write("")

with st.container(border=True):
    section("05", "Redukcja wymiarowości: PCA i t-SNE")
    story(
        "Model korzysta z 13 cech wejściowych, jednak część z nich mierzy w praktyce tę samą "
        "własność - wagę, wymiary fizyczne i objętość diamentu determinuje głównie jego fizyczny "
        "rozmiar. PCA pozwala to policzyć wprost."
    )
    explain(
        f"Do zachowania 90% wariancji wystarczają **{eda['pca_n_components_90pct']}** komponenty "
        f"z {eda['n_cols_final']} cech - silny sygnał, że część informacji się powtarza."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pca_scree["n_components"], y=pca_scree["cumulative_variance"],
                              mode="lines+markers", line=dict(color=ACCENT, width=2.5), marker=dict(size=5)))
    fig.add_hline(y=0.9, line_dash="dash", line_color=GOLD)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.markdown("**Struktura danych na płaszczyźnie**")
    story("PCA (metoda liniowa) i t-SNE (metoda nieliniowa) rzutują te same diamenty na dwa wymiary. Widoczny gradient według wagi czy szlifu odzwierciedla rzeczywistą strukturę rynkową.")
    color_by = st.radio("Koloruj według:", ["cut", "color"], horizontal=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Rzut PCA (liniowy)**")
        fig = px.scatter(pca_2d, x="PC1", y="PC2", color=color_by, opacity=0.7, render_mode="svg",
                          category_orders={"cut": CUT_ORDER, "color": COLOR_ORDER},
                          color_discrete_sequence=px.colors.sequential.Tealgrn if color_by == "cut" else px.colors.sequential.Purp)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col_r:
        st.markdown("**Rzut t-SNE (nieliniowy)**")
        fig = px.scatter(tsne_2d, x="TSNE1", y="TSNE2", color=color_by, opacity=0.7, render_mode="svg",
                          category_orders={"cut": CUT_ORDER, "color": COLOR_ORDER},
                          color_discrete_sequence=px.colors.sequential.Tealgrn if color_by == "cut" else px.colors.sequential.Purp)
        st.plotly_chart(style_fig(fig), use_container_width=True)

st.write("")

with st.container(border=True):
    section("06", "Model predykcyjny")
    story(
        "Random Forest Regressor przewiduje logarytm ceny na podstawie wagi, szlifu, koloru, "
        "czystości i wymiarów fizycznych diamentu. Cechy będące pochodnymi samej ceny są świadomie "
        "wykluczone - ich użycie pozwoliłoby modelowi podejrzeć odpowiedź zamiast się jej nauczyć."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{model_metrics['r2']:.4f}")
    c2.metric("MAE", f"{model_metrics['mae']:.4f}")
    c3.metric("RMSE", f"{model_metrics['rmse']:.4f}")
    st.divider()
    explain(
        "Zbiór podzielono na 80% danych treningowych i 20% testowych. Model uczy się wyłącznie na "
        "części treningowej, a wszystkie metryki powyżej są liczone na części testowej - danych, "
        "których model nigdy nie widział podczas treningu."
    )

    st.divider()
    st.markdown("**Korelacja cech wejściowych z ceną**")
    corr_cols = ["carat", "depth", "table", "x", "y", "z", "price"]
    corr = df_sample[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=[[0, NEGATIVE], [0.5, "#E8EAF0"], [1, ACCENT]], zmin=-1, zmax=1)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Rzeczywiste a przewidywane wartości**")
        explain("Każdy punkt to jeden diament ze zbioru testowego. Im bliżej przerywanej linii, tym trafniejsza predykcja.")
        fig = px.scatter(predictions, x="y_true", y="y_pred", opacity=0.45, render_mode="svg",
                          labels={"y_true": "actual", "y_pred": "predicted"}, color_discrete_sequence=[ACCENT])
        mn, mx = predictions["y_true"].min(), predictions["y_true"].max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash", color=GOLD)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col_r:
        st.markdown("**Ranking ważności cech**")
        explain("Miara wpływu danej cechy na redukcję błędu przy podziałach w drzewach lasu losowego.")
        fig = px.bar(feature_importance.head(10).sort_values("importance"),
                     x="importance", y="feature", orientation="h", color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.markdown("**Wyszukiwarka diamentów**")
    search_term = st.text_input("Wpisz numer diamentu", "", label_visibility="collapsed", placeholder="Wpisz numer diamentu")
    if search_term:
        results = diamonds_search[diamonds_search["diamond_id"].astype(str).str.contains(search_term, na=False)]
    else:
        results = diamonds_search.head(0)

    if search_term and len(results) > 0:
        st.write(f"Znaleziono {len(results)} wyników")
        for _, row in results.head(10).iterrows():
            if row["Is_Overpriced"]:
                badge = '<span class="badge badge-over">Przewartościowany</span>'
            elif row["Is_Undervalued"]:
                badge = '<span class="badge badge-under">Okazja</span>'
            else:
                badge = '<span class="badge badge-normal">Cena rynkowa</span>'
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**#{row['diamond_id']}** - {row['cut']}, {row['color']}/{row['clarity']}")
                    st.markdown(badge, unsafe_allow_html=True)
                with c2:
                    st.metric("Carat", f"{row['carat']:.2f}")
                    st.metric("Cena", f"${row['price']:,.0f}")
                with c3:
                    st.metric("Szacowana cena rynkowa", f"${row['price_baseline_pred']:,.0f}")
    elif search_term:
        st.warning("Brak wyników dla podanego numeru.")
    else:
        st.info("Wpisz numer diamentu, aby zobaczyć jego wycenę i status anomalii.")

st.write("")

with st.container(border=True):
    section("07", "Kalkulator wyceny")
    story(
        "Ustawienie wagi i 4C (cut, color, clarity) pozwala wytrenowanemu modelowi obliczyć szacowaną "
        "cenę rynkową na żywo. Wymiary fizyczne są szacowane z zależności liniowej od wagi, "
        "obserwowanej w danych treningowych."
    )
    st.write("")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        carat = st.slider("Waga (carat)", min_value=float(round(calc_defaults["carat_min"], 1)),
                           max_value=float(round(calc_defaults["carat_max"], 1)), value=1.0, step=0.01)
        cut = st.select_slider("Szlif (cut)", options=CUT_ORDER, value="Ideal")
        color = st.select_slider("Kolor - D to bezbarwny, J to najsłabszy", options=COLOR_ORDER, value="G")
        clarity = st.select_slider("Czystość - IF to najlepsza, I1 to najsłabsza", options=CLARITY_ORDER, value="VS1")

    def scale_val(col_name, val):
        p = scale_params[col_name]
        return (val - p["center"]) / p["scale"]

    x = calc_defaults["x_by_carat_slope"] * carat + calc_defaults["x_by_carat_intercept"]
    y = calc_defaults["y_by_carat_slope"] * carat + calc_defaults["y_by_carat_intercept"]
    z = calc_defaults["z_by_carat_slope"] * carat + calc_defaults["z_by_carat_intercept"]
    depth = calc_defaults["depth_median"]
    table = calc_defaults["table_median"]
    volume = x * y * z
    depth_calculated = z / ((x + y) / 2) * 100
    depth_mismatch = abs(depth - depth_calculated)
    log_carat = np.log1p(carat)

    row = {c: 0 for c in feature_cols_model}
    row["carat"] = scale_val("carat", carat)
    row["depth"] = scale_val("depth", depth)
    row["table"] = scale_val("table", table)
    row["x"] = scale_val("x", x)
    row["y"] = scale_val("y", y)
    row["z"] = scale_val("z", z)
    row["volume"] = scale_val("volume", volume)
    row["depth_calculated"] = scale_val("depth_calculated", depth_calculated)
    row["depth_mismatch"] = scale_val("depth_mismatch", depth_mismatch)
    row["log_carat"] = scale_val("log_carat", log_carat)
    row["cut_encoded"] = scale_val("cut_encoded", encoding_maps["cut_order"][cut])
    row["color_encoded"] = scale_val("color_encoded", encoding_maps["color_order"][color])
    row["clarity_encoded"] = scale_val("clarity_encoded", encoding_maps["clarity_order"][clarity])

    X_calc = pd.DataFrame([row])[feature_cols_model]
    tree_preds = np.array([t.predict(X_calc.values)[0] for t in model.estimators_])
    pred_log = tree_preds.mean()
    pred_price = np.expm1(pred_log)
    pred_low = np.expm1(np.percentile(tree_preds, 10))
    pred_high = np.expm1(np.percentile(tree_preds, 90))

    baseline_pred = np.exp(calc_defaults["baseline_log_b0"] + calc_defaults["baseline_log_b1"] * np.log(carat))
    delta_vs_baseline = pred_price - baseline_pred

    with col_result:
        delta_color = "var(-positive)" if delta_vs_baseline >= 0 else "var(-negative)"
        with st.container(border=True):
            st.markdown("<div class='dv-price-label'>Szacowana cena rynkowa</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dv-price-value'>${pred_price:,.0f}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='dv-explain'>Przedział ufności: ${pred_low:,.0f} \u2013 ${pred_high:,.0f} "
                f"(wynika z rozrzutu predykcji poszczególnych drzew lasu losowego)</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='dv-explain' style='margin-top:0.3rem;'>"
                f"<span style='color:{delta_color}; font-weight:700;'>"
                f"{'+' if delta_vs_baseline >= 0 else ''}{delta_vs_baseline:,.0f} $</span> "
                f"- różnica względem modelu bazowego opartego tylko o wagę</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        with st.expander("Cechy użyte do wyceny"):
            display_feats = {
                "carat": str(round(carat, 2)), "cut": cut, "color": color, "clarity": clarity,
                "x (mm)": str(round(x, 2)), "y (mm)": str(round(y, 2)), "z (mm)": str(round(z, 2)),
                "volume (mm3)": str(round(volume, 1)), "depth (%)": str(round(depth, 1)), "table (%)": str(round(table, 1)),
            }
            st.dataframe(pd.DataFrame(display_feats.items(), columns=["feature", "value"]),
                         use_container_width=True, hide_index=True)

    st.caption("Kalkulator demonstracyjny oparty o model wytrenowany na historycznych danych rynkowych. Nie stanowi wyceny jubilerskiej.")

st.markdown(
    """
    <div class="dv-footer">
        <div class="dv-footer-brand">Analityka cen diamentów</div>
        <div class="dv-footer-meta">
            Źródło danych: prawdziwy, publiczny zbiór diamonds (ggplot2/tidyverse) 
            <a href="https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv" target="_blank">Zobacz źródło danych</a>
            &nbsp;\u2022&nbsp; Streamlit + Plotly
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)