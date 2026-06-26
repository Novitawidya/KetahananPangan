import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SatDat - Analisis Ketahanan Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(160deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); }
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 50%, #7c3aed 100%);
    border-radius: 24px; padding: 2.8rem 2.5rem; margin-bottom: 1.8rem;
    box-shadow: 0 0 60px rgba(124,58,237,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; color: white; }
.hero p  { font-size: 1rem; color: #bfdbfe; margin: .6rem 0 0; }
.card {
    background: rgba(30,27,75,0.7); border-radius: 20px; padding: 1.6rem;
    border: 1px solid rgba(124,58,237,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin-bottom: 1rem;
}
.card h3 { font-weight: 700; margin: 0 0 1rem; font-size: 1.05rem; color: #a5b4fc; }
.metric-box {
    background: linear-gradient(135deg, rgba(30,27,75,0.9), rgba(49,46,129,0.9));
    border-radius: 18px; padding: 1.4rem; text-align: center;
    border: 1px solid rgba(124,58,237,0.3);
}
.metric-box .val { font-size: 2rem; font-weight: 800; color: #a5b4fc; }
.metric-box .lbl { font-size: .72rem; color: #94a3b8; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 1.2px; margin-top: .4rem; }
.aspek-card {
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: .8rem;
    border-left: 5px solid; font-size: .95rem;
}
.aspek-title { font-weight: 800; font-size: 1rem; margin-bottom: .4rem; }
.aspek-val   { font-size: 1.5rem; font-weight: 700; }
.aspek-desc  { font-size: .82rem; color: #cbd5e1; margin-top: .3rem; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
}
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] label { color: #c7d2fe !important; }
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color: #a5b4fc !important; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; font-weight: 700 !important;
    width: 100% !important;
}
p, span, li { color: #e2e8f0 !important; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
.stTabs [aria-selected="true"] { color: #a5b4fc !important; border-bottom-color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    # Cari file Excel dari berbagai kemungkinan lokasi
    search_dirs = [
        os.path.join(BASE, 'data'),   # ./data/
        BASE,                          # root repo
        os.path.join(BASE, '..'),      # parent folder
        '/mount/src',                  # Streamlit Cloud mount root
    ]

    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.xlsx', '.xls')) and not f.startswith('~'):
                    candidates.append(os.path.join(d, f))

    if not candidates:
        # Tampilkan info debug yang membantu
        st.error("❌ File Excel tidak ditemukan!")
        st.markdown("**Debug info:**")
        st.code(f"BASE path: {BASE}\nSearched dirs: {search_dirs}")
        for d in search_dirs:
            if os.path.exists(d):
                try:
                    contents = os.listdir(d)
                    st.code(f"{d}:\n" + "\n".join(contents[:20]))
                except Exception as e:
                    st.code(f"{d}: ERROR - {e}")
        st.markdown("""
**Solusi:** Pastikan file Excel sudah di-upload ke folder `data/` di dalam repository GitHub kamu.

Struktur yang benar:
```
repo/
├── app.py
├── requirements.txt
└── data/
    └── nama_file_kamu.xlsx
```
        """)
        st.stop()

    path = candidates[0]
    try:
        df = pd.read_excel(path, sheet_name='Sheet1')
    except Exception:
        df = pd.read_excel(path, sheet_name=0)
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

FEATURE_COLS = [
    'IKP', 'Produksi Padi', 'produksi jagung', 'PDRB ADHB',
    'harga beras', 'akses sanitasi layak', 'akses air minum layak',
    'Curah hujan (mm/hari)', 'Kecepatan angin pada ketinggian 2 meter (m/s)'
]
FEATURE_LABELS = {
    'IKP': 'Indeks Ketahanan Pangan',
    'Produksi Padi': 'Produksi Padi (X1)',
    'produksi jagung': 'Produksi Jagung (X2)',
    'PDRB ADHB': 'PDRB / Pendapatan Per Kapita (X3)',
    'harga beras': 'Harga Beras (X4)',
    'akses sanitasi layak': 'Akses Sanitasi Layak / X5 (%)',
    'akses air minum layak': 'Akses Air Minum Layak / X6 (%)',
    'Curah hujan (mm/hari)': 'Curah Hujan / X7 (mm/hari)',
    'Kecepatan angin pada ketinggian 2 meter (m/s)': 'Kecepatan Angin / X8 (m/s)',
}

ASPEK = {
    'Ketersediaan': {
        'cols': ['Produksi Padi', 'produksi jagung'],
        'color': '#34d399',
        'bg': 'rgba(52,211,153,0.08)',
        'border': '#34d399',
        'icon': 'X1 & X2',
        'desc': 'Produksi Padi (X1) + Produksi Jagung (X2)',
    },
    'Aksesibilitas': {
        'cols': ['PDRB ADHB', 'harga beras'],
        'color': '#60a5fa',
        'bg': 'rgba(96,165,250,0.08)',
        'border': '#60a5fa',
        'icon': 'X3 & X4',
        'desc': 'PDRB/Pendapatan Per Kapita (X3) + Harga Beras (X4)',
    },
    'Pemanfaatan': {
        'cols': ['akses sanitasi layak', 'akses air minum layak'],
        'color': '#f472b6',
        'bg': 'rgba(244,114,182,0.08)',
        'border': '#f472b6',
        'icon': 'X5 & X6',
        'desc': 'Akses Sanitasi Layak (X5) + Akses Air Minum Layak (X6)',
    },
    'Stabilitas': {
        'cols': ['Curah hujan (mm/hari)', 'Kecepatan angin pada ketinggian 2 meter (m/s)'],
        'color': '#fb923c',
        'bg': 'rgba(251,146,60,0.08)',
        'border': '#fb923c',
        'icon': 'X7 & X8',
        'desc': 'Curah Hujan (X7) + Kecepatan Angin (X8)',
    },
}

PALETTE = ['#60a5fa','#f472b6','#34d399','#fb923c','#a78bfa','#fbbf24','#f87171','#22d3ee']
PLOT_THEME = dict(
    plot_bgcolor='rgba(15,23,42,0.8)', paper_bgcolor='rgba(15,23,42,0)',
    font_color='#e2e8f0', margin=dict(t=40, b=20, l=20, r=20)
)

with st.sidebar:
    st.markdown("## SatDat Dashboard")
    st.markdown("---")
    page = st.radio("Menu", [
        "Beranda", "Eksplorasi Data",
        "Klasterisasi DBSCAN", "Visualisasi t-SNE", "Analisis Gabungan",
        "Karakteristik Klaster"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Parameter DBSCAN")
    eps_val = st.slider("Epsilon (eps)", 0.1, 5.0, 1.5, 0.1)
    min_samples_val = st.slider("Min Samples", 2, 10, 3)
    st.markdown("### Parameter t-SNE")
    perplexity_val = st.slider("Perplexity", 5, 30, 10)
    tsne_iter = st.slider("Max Iterasi", 500, 2000, 1000, 100)
    st.markdown("---")
    st.markdown("**Metode:** t-SNE + DBSCAN")
    st.markdown("**Provinsi:** 38")

X = df[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
labels_db = dbscan.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = labels_db
df['Cluster_Label'] = df['Cluster_DBSCAN'].apply(
    lambda x: 'Noise (-1)' if x == -1 else f'Klaster {x+1}'
)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)
valid_mask = labels_db != -1

if n_clusters >= 2 and sum(valid_mask) > n_clusters:
    sil_score = silhouette_score(X_scaled[valid_mask], labels_db[valid_mask])
    db_score  = davies_bouldin_score(X_scaled[valid_mask], labels_db[valid_mask])
else:
    sil_score = None
    db_score  = None

@st.cache_data
def run_tsne(data_hash, perplexity, n_iter):
    return TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=42, init='pca').fit_transform(X_scaled)

tsne_result = run_tsne(hash(X_scaled.tobytes()), perplexity_val, tsne_iter)
df['TSNE_1'] = tsne_result[:, 0]
df['TSNE_2'] = tsne_result[:, 1]

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)
df['PCA_1'], df['PCA_2'], df['PCA_3'] = pca_result[:,0], pca_result[:,1], pca_result[:,2]

unique_labels = sorted(df['Cluster_DBSCAN'].unique())
color_map = {}
for i, lbl in enumerate(unique_labels):
    key = 'Noise (-1)' if lbl == -1 else f'Klaster {lbl+1}'
    color_map[key] = '#64748b' if lbl == -1 else PALETTE[i % len(PALETTE)]

def level_label(val, col, thresholds=(33, 66)):
    arr = df[col].dropna()
    p33, p66 = np.percentile(arr, thresholds[0]), np.percentile(arr, thresholds[1])
    if val <= p33:   return 'Rendah'
    elif val <= p66: return 'Sedang'
    else:            return 'Tinggi'

# ════ BERANDA ════
if page == "Beranda":
    st.markdown("""<div class="hero">
        <h1>Analisis Ketahanan Pangan Indonesia</h1>
        <p>Klasterisasi 38 Provinsi menggunakan DBSCAN dan Visualisasi t-SNE</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (str(len(df)),"Jumlah Provinsi"),(str(n_clusters),"Klaster Ditemukan"),
        (str(n_noise),"Titik Noise"),(f"{sil_score:.3f}" if sil_score else "N/A","Silhouette Score")
    ]):
        col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card"><h3>Tentang Proyek</h3>', unsafe_allow_html=True)
        st.markdown("""
**Tujuan:** Mengidentifikasi pola kemiripan antar provinsi berdasarkan 4 aspek ketahanan pangan.

**4 Aspek Ketahanan Pangan:**
- **Ketersediaan** (X1, X2) - Produksi Padi & Jagung
- **Aksesibilitas** (X3, X4) - Pendapatan Per Kapita & Harga Beras
- **Pemanfaatan** (X5, X6) - Akses Sanitasi & Air Minum
- **Stabilitas** (X7, X8) - Curah Hujan & Kecepatan Angin

**Metode:** DBSCAN + t-SNE
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        fig = px.scatter(df, x='TSNE_1', y='TSNE_2', color='Cluster_Label',
                         color_discrete_map=color_map, hover_name='PROVINSI',
                         title='Peta t-SNE - Klaster DBSCAN')
        fig.update_traces(marker=dict(size=12, line=dict(width=1.5, color='white')))
        fig.update_layout(**PLOT_THEME, height=380, legend=dict(font_color='#ccc'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card"><h3>Preview Dataset</h3>', unsafe_allow_html=True)
    prev = df[['PROVINSI','IKP','Cluster_Label','akses sanitasi layak','akses air minum layak','harga beras']].copy()
    prev.columns = ['Provinsi','IKP','Klaster','Sanitasi (%)','Air Minum (%)','Harga Beras']
    st.dataframe(prev.style.background_gradient(subset=['IKP'], cmap='Blues'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════ EKSPLORASI DATA ════
elif page == "Eksplorasi Data":
    st.markdown("""<div class="hero">
        <h1>Eksplorasi Data</h1>
        <p>Distribusi dan korelasi variabel ketahanan pangan 38 provinsi</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribusi Variabel", "Heatmap Korelasi", "Ranking Provinsi"])

    with tab1:
        feat_sel = st.selectbox("Pilih Variabel", FEATURE_COLS, format_func=lambda x: FEATURE_LABELS.get(x,x))
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=feat_sel, nbins=15, color_discrete_sequence=['#818cf8'],
                               title=f'Distribusi - {FEATURE_LABELS.get(feat_sel)}')
            fig.update_layout(**PLOT_THEME, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, y=feat_sel, points='all', color_discrete_sequence=['#a78bfa'],
                         hover_name='PROVINSI', title=f'Box Plot - {FEATURE_LABELS.get(feat_sel)}')
            fig.update_layout(**PLOT_THEME, height=350)
            st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(df.sort_values(feat_sel), x='PROVINSI', y=feat_sel,
                     color=feat_sel, color_continuous_scale='Viridis',
                     title=f'{FEATURE_LABELS.get(feat_sel)} per Provinsi')
        fig.update_layout(**PLOT_THEME, height=420, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df[FEATURE_COLS].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title='Heatmap Korelasi Antar Variabel', height=550)
        fig.update_layout(**PLOT_THEME)
        fig.update_xaxes(tickangle=-30, tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i,j]
                if abs(v) >= 0.5:
                    high_corr.append({'Variabel A': FEATURE_LABELS.get(corr.columns[i], corr.columns[i]),
                                      'Variabel B': FEATURE_LABELS.get(corr.columns[j], corr.columns[j]),
                                      'Korelasi': round(v,3), 'Tipe': 'Negatif' if v<0 else 'Positif'})
        if high_corr:
            st.markdown('<div class="card"><h3>Korelasi Kuat (|r| >= 0.5)</h3>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(high_corr).sort_values('Korelasi', key=abs, ascending=False), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        top_n = st.slider("Top N Provinsi", 5, 38, 15)
        sort_by = st.selectbox("Urutkan berdasarkan", FEATURE_COLS, format_func=lambda x: FEATURE_LABELS.get(x,x))
        df_s = df.nlargest(top_n, sort_by)
        fig = go.Figure(go.Bar(x=df_s['PROVINSI'], y=df_s[sort_by],
                               marker=dict(color=df_s[sort_by], colorscale='Plasma'),
                               text=[f"{v:.2f}" for v in df_s[sort_by]],
                               textposition='outside', textfont_color='white'))
        fig.update_layout(title=f'Top {top_n} - {FEATURE_LABELS.get(sort_by)}',
                          xaxis_tickangle=-40, **PLOT_THEME, height=430)
        st.plotly_chart(fig, use_container_width=True)

# ════ KLASTERISASI DBSCAN ════
elif page == "Klasterisasi DBSCAN":
    st.markdown("""<div class="hero">
        <h1>Klasterisasi DBSCAN</h1>
        <p>Density-Based Spatial Clustering - Menemukan kelompok alami tanpa asumsi jumlah klaster</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (str(n_clusters),"Klaster Ditemukan"),(str(n_noise),"Titik Noise"),
        (f"{sil_score:.4f}" if sil_score else "N/A","Silhouette Score"),
        (f"{db_score:.4f}" if db_score else "N/A","Davies-Bouldin"),
    ]):
        col.markdown(f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Visualisasi PCA 2D", "PCA 3D", "Anggota Klaster"])

    with tab1:
        x_ax = st.selectbox("Sumbu X", FEATURE_COLS, index=0, format_func=lambda x: FEATURE_LABELS.get(x,x), key='xax')
        y_ax = st.selectbox("Sumbu Y", FEATURE_COLS, index=5, format_func=lambda x: FEATURE_LABELS.get(x,x), key='yax')
        fig = px.scatter(df, x=x_ax, y=y_ax, color='Cluster_Label', color_discrete_map=color_map,
                         hover_name='PROVINSI',
                         title=f'DBSCAN - {FEATURE_LABELS.get(x_ax)} vs {FEATURE_LABELS.get(y_ax)}')
        fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color='white')))
        fig.update_layout(**PLOT_THEME, height=480, legend=dict(font_color='#ccc'))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x='PCA_1', y='PCA_2', color='Cluster_Label', color_discrete_map=color_map,
                          hover_name='PROVINSI', title='DBSCAN pada Ruang PCA 2D')
        fig2.update_traces(marker=dict(size=14, line=dict(width=1.5, color='white')))
        fig2.update_layout(**PLOT_THEME, height=450,
                           xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                           yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                           legend=dict(font_color='#ccc'))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = px.scatter_3d(df, x='PCA_1', y='PCA_2', z='PCA_3', color='Cluster_Label',
                             color_discrete_map=color_map, hover_name='PROVINSI',
                             hover_data={'IKP':':.2f'}, title='Visualisasi 3D DBSCAN (PCA)', height=600)
        fig3.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        fig3.update_layout(**PLOT_THEME,
                           scene=dict(bgcolor='rgba(15,23,42,0.8)',
                                      xaxis=dict(gridcolor='#334155', color='#94a3b8'),
                                      yaxis=dict(gridcolor='#334155', color='#94a3b8'),
                                      zaxis=dict(gridcolor='#334155', color='#94a3b8')),
                           legend=dict(font_color='#ccc'))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="card"><h3>Explained Variance PCA</h3>', unsafe_allow_html=True)
        fig_p = go.Figure(go.Bar(x=[f'PC{i+1}' for i in range(3)],
                                  y=pca.explained_variance_ratio_*100,
                                  marker_color=['#818cf8','#a78bfa','#c4b5fd'],
                                  text=[f'{v:.1f}%' for v in pca.explained_variance_ratio_*100],
                                  textposition='outside', textfont_color='white'))
        fig_p.update_layout(**PLOT_THEME, height=300, title='Variance per Komponen PCA')
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        for cluster_id in sorted(df['Cluster_DBSCAN'].unique()):
            sub = df[df['Cluster_DBSCAN']==cluster_id]
            label = 'Noise (Outlier)' if cluster_id==-1 else f'Klaster {cluster_id+1}'
            color = '#64748b' if cluster_id==-1 else PALETTE[cluster_id % len(PALETTE)]
            with st.expander(f"{label} -- {len(sub)} Provinsi"):
                show = sub[['PROVINSI','IKP','akses sanitasi layak','akses air minum layak','Produksi Padi','harga beras']].copy()
                show.columns = ['Provinsi','IKP','Sanitasi (%)','Air Minum (%)','Prod. Padi','Harga Beras']
                st.dataframe(show.sort_values('IKP', ascending=False), use_container_width=True)
                if len(sub) > 1:
                    means = sub[FEATURE_COLS].mean()
                    fig_b = go.Figure(go.Bar(x=[FEATURE_LABELS.get(c,c) for c in FEATURE_COLS],
                                             y=means.values, marker_color=color))
                    fig_b.update_layout(**PLOT_THEME, height=280,
                                        title=f'Rata-rata Variabel - {label}', xaxis_tickangle=-30)
                    st.plotly_chart(fig_b, use_container_width=True)

# ════ VISUALISASI t-SNE ════
elif page == "Visualisasi t-SNE":
    st.markdown("""<div class="hero">
        <h1>Visualisasi t-SNE</h1>
        <p>Reduksi dimensi untuk memetakan kemiripan antar provinsi dalam ruang 2D</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["t-SNE 2D", "Warna Variabel", "Penjelasan t-SNE"])

    with tab1:
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.scatter(df, x='TSNE_1', y='TSNE_2', color='Cluster_Label',
                             color_discrete_map=color_map, hover_name='PROVINSI',
                             hover_data={'TSNE_1':False,'TSNE_2':False,'IKP':':.2f'},
                             title=f't-SNE 2D - DBSCAN (perplexity={perplexity_val})', text='PROVINSI')
            fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color='white')),
                              textposition='top center', textfont=dict(size=8, color='white'))
            fig.update_layout(**PLOT_THEME, height=550, legend=dict(font_color='#ccc'))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<div class="card"><h3>Legenda Klaster</h3>', unsafe_allow_html=True)
            for cluster_id in sorted(df['Cluster_DBSCAN'].unique()):
                lbl = 'Noise (-1)' if cluster_id==-1 else f'Klaster {cluster_id+1}'
                clr = color_map.get(lbl, '#888')
                members = df[df['Cluster_DBSCAN']==cluster_id]['PROVINSI'].tolist()
                st.markdown(
                    f'<div style="background:{clr}22;border-left:4px solid {clr};'
                    f'padding:.8rem;border-radius:8px;margin:.5rem 0;">'
                    f'<b style="color:{clr}">{lbl}</b><br>'
                    f'<small style="color:#94a3b8">{len(members)} provinsi</small><br>'
                    f'<small style="color:#cbd5e1">{", ".join(members[:5])}{"..." if len(members)>5 else ""}</small>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if sil_score:
                quality = "Baik" if sil_score>0.5 else ("Sedang" if sil_score>0.25 else "Lemah")
                st.markdown('<div class="card"><h3>Kualitas Klaster</h3>', unsafe_allow_html=True)
                st.markdown(f"**Silhouette:** `{sil_score:.4f}` - {quality}\n\n**Davies-Bouldin:** `{db_score:.4f}`")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        color_by = st.selectbox("Warnai berdasarkan", FEATURE_COLS, format_func=lambda x: FEATURE_LABELS.get(x,x))
        cscale = st.selectbox("Color Scale", ['Plasma','Viridis','Turbo','RdYlGn','Blues'])
        fig = px.scatter(df, x='TSNE_1', y='TSNE_2', color=color_by, color_continuous_scale=cscale,
                         hover_name='PROVINSI', title=f't-SNE - {FEATURE_LABELS.get(color_by)}', text='PROVINSI')
        fig.update_traces(marker=dict(size=16, line=dict(width=1, color='rgba(255,255,255,0.3)')),
                          textposition='top center', textfont=dict(size=8, color='white'))
        fig.update_layout(**PLOT_THEME, height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="card"><h3>Apa itu t-SNE?</h3>', unsafe_allow_html=True)
        st.markdown("""
**t-SNE (t-Distributed Stochastic Neighbor Embedding)** adalah teknik reduksi dimensi non-linear
yang sangat efektif untuk memvisualisasikan data berdimensi tinggi dalam 2D atau 3D.

**Cara Kerja:**
1. Menghitung kemiripan antar titik data di dimensi tinggi
2. Memetakan ke dimensi rendah dengan mempertahankan struktur lokal
3. Mengoptimalkan posisi agar kemiripan di dimensi rendah mirip dimensi tinggi

**Parameter Penting:**
- **Perplexity** – jumlah tetangga efektif yang dipertimbangkan (5–50)
- **Max Iterasi** – lebih banyak iterasi = lebih stabil, tapi lebih lambat

**Interpretasi:**
- Titik berdekatan = profil ketahanan pangan mirip
- Klaster terpisah = kelompok provinsi dengan karakteristik berbeda
- Titik berjauhan = profil berbeda signifikan
- Jarak ANTAR klaster di t-SNE tidak bermakna langsung
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ════ ANALISIS GABUNGAN ════
elif page == "Analisis Gabungan":
    st.markdown("""<div class="hero">
        <h1>Analisis Gabungan t-SNE + DBSCAN</h1>
        <p>Insight mendalam tentang pola ketahanan pangan antar klaster provinsi</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Profil Klaster", "Radar Chart", "Sebaran IKP"])

    with tab1:
        summary = df.groupby('Cluster_Label')[FEATURE_COLS].mean().round(2)
        summary.columns = [FEATURE_LABELS.get(c,c) for c in FEATURE_COLS]
        st.dataframe(summary.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)
        feat_cmp = st.multiselect("Pilih variabel untuk dibandingkan", FEATURE_COLS,
                                  default=['IKP','akses sanitasi layak','akses air minum layak'],
                                  format_func=lambda x: FEATURE_LABELS.get(x,x))
        if feat_cmp:
            means_df = df.groupby('Cluster_Label')[feat_cmp].mean()
            fig = go.Figure()
            for i, cl in enumerate(means_df.index):
                fig.add_trace(go.Bar(name=cl, x=[FEATURE_LABELS.get(f,f) for f in feat_cmp],
                                     y=means_df.loc[cl].values,
                                     marker_color=color_map.get(cl, PALETTE[i%len(PALETTE)])))
            fig.update_layout(barmode='group', title='Perbandingan Variabel Antar Klaster',
                              **PLOT_THEME, height=450, xaxis_tickangle=-20, legend=dict(font_color='#ccc'))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        valid_clusters = [c for c in df['Cluster_Label'].unique() if 'Noise' not in c]
        norm_df = df.copy()
        for col in FEATURE_COLS:
            mn, mx = norm_df[col].min(), norm_df[col].max()
            norm_df[col] = (norm_df[col] - mn) / (mx - mn + 1e-9)
        categories = [FEATURE_LABELS.get(f,f) for f in FEATURE_COLS]
        fig_r = go.Figure()
        for i, cl in enumerate(sorted(valid_clusters)):
            sub = norm_df[norm_df['Cluster_Label']==cl]
            vals = sub[FEATURE_COLS].mean().values.tolist()
            vals += vals[:1]
            clr = color_map.get(cl, PALETTE[i%len(PALETTE)])
            fig_r.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]],
                                             fill='toself', name=cl, line_color=clr))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor='#334155', color='#94a3b8'),
                       angularaxis=dict(gridcolor='#334155', color='#94a3b8'),
                       bgcolor='rgba(15,23,42,0.6)'),
            paper_bgcolor='rgba(15,23,42,0)', font_color='#e2e8f0',
            title='Radar Chart Profil Klaster (Normalisasi 0-1)', height=550, legend=dict(font_color='#ccc'))
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        df_ikp = df.sort_values('IKP')
        fig = go.Figure(go.Bar(x=df_ikp['IKP'], y=df_ikp['PROVINSI'], orientation='h',
                               marker=dict(color=df_ikp['IKP'], colorscale='RdYlGn'),
                               text=[f"{v:.2f}" for v in df_ikp['IKP']],
                               textposition='outside', textfont_color='white'))
        fig.update_layout(**PLOT_THEME, title='Indeks Ketahanan Pangan per Provinsi',
                          height=1000, xaxis_title='IKP Score')
        st.plotly_chart(fig, use_container_width=True)
        ikp_cl = df.groupby('Cluster_Label')['IKP'].agg(['mean','min','max','count'])
        ikp_cl.columns = ['Rata-rata IKP','Min IKP','Max IKP','Jumlah Provinsi']
        st.dataframe(ikp_cl.sort_values('Rata-rata IKP', ascending=False)
                     .style.background_gradient(subset=['Rata-rata IKP'], cmap='Greens'),
                     use_container_width=True)

# ════ KARAKTERISTIK KLASTER (4 ASPEK) ════
elif page == "Karakteristik Klaster":
    st.markdown("""<div class="hero">
        <h1>Karakteristik Klaster</h1>
        <p>Analisis 4 Aspek Ketahanan Pangan: Ketersediaan, Aksesibilitas, Pemanfaatan, Stabilitas</p>
    </div>""", unsafe_allow_html=True)

    all_clusters = sorted(df['Cluster_Label'].unique())
    cluster_sel = st.selectbox("Pilih Klaster untuk Dianalisis", all_clusters)
    sub = df[df['Cluster_Label'] == cluster_sel]
    clr = color_map.get(cluster_sel, '#888')

    st.markdown(f"""
    <div style="background:{clr}18;border:2px solid {clr};border-radius:16px;
                padding:1.2rem 1.6rem;margin-bottom:1.5rem;">
        <b style="color:{clr};font-size:1.2rem">{cluster_sel}</b>
        <span style="color:#94a3b8;margin-left:1rem">{len(sub)} Provinsi</span><br>
        <small style="color:#cbd5e1">{' · '.join(sub['PROVINSI'].tolist())}</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Profil 4 Aspek Ketahanan Pangan")

    for aspek_name, aspek_info in ASPEK.items():
        cols_aspek = aspek_info['cols']
        clr_a = aspek_info['color']
        bg_a  = aspek_info['bg']
        brd_a = aspek_info['border']

        cluster_vals = sub[cols_aspek].mean()
        global_vals  = df[cols_aspek].mean()

        st.markdown(f"""
        <div class="aspek-card" style="background:{bg_a};border-left-color:{brd_a};">
            <div class="aspek-title" style="color:{clr_a}">
                {aspek_name} &nbsp;<small style="font-weight:400;color:#94a3b8">({aspek_info['icon']}) — {aspek_info['desc']}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        metric_cols = st.columns(len(cols_aspek) * 2)
        for idx, col_name in enumerate(cols_aspek):
            c_val = cluster_vals[col_name]
            g_val = global_vals[col_name]
            diff  = ((c_val - g_val) / g_val * 100) if g_val != 0 else 0
            lvl   = level_label(c_val, col_name)
            lvl_color = '#34d399' if lvl=='Tinggi' else ('#fb923c' if lvl=='Sedang' else '#f87171')
            arrow = '▲' if diff > 0 else '▼'
            arrow_color = '#34d399' if diff > 0 else '#f87171'

            with metric_cols[idx*2]:
                st.markdown(f"""
                <div class="metric-box" style="border-color:{clr_a}44;">
                    <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px">
                        {FEATURE_LABELS.get(col_name, col_name)}
                    </div>
                    <div class="val" style="color:{clr_a};margin:.4rem 0">{c_val:,.2f}</div>
                    <div style="font-size:.78rem;color:{lvl_color};font-weight:700">{lvl}</div>
                    <div style="font-size:.75rem;color:{arrow_color};margin-top:.2rem">
                        {arrow} {abs(diff):.1f}% vs rata-rata nasional
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with metric_cols[idx*2 + 1]:
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Bar(
                    x=['Klaster ini', 'Rata-rata Nasional'],
                    y=[c_val, g_val],
                    marker_color=[clr_a, '#475569'],
                    text=[f'{c_val:,.1f}', f'{g_val:,.1f}'],
                    textposition='outside', textfont_color='white', textfont_size=10,
                ))
                fig_mini.update_layout(**PLOT_THEME, height=180, margin=dict(t=10,b=10,l=5,r=5),
                                       showlegend=False, yaxis=dict(visible=False),
                                       xaxis=dict(tickfont_size=9))
                st.plotly_chart(fig_mini, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Radar Chart - Skor 4 Aspek")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    norm_aspek_cluster = {}
    norm_aspek_nasional = {}
    for aspek_name, aspek_info in ASPEK.items():
        cols_a = aspek_info['cols']
        scores = []
        scores_global = []
        for c in cols_a:
            mn, mx = df[c].min(), df[c].max()
            norm_val = (sub[c].mean() - mn) / (mx - mn + 1e-9)
            norm_glob = (df[c].mean() - mn) / (mx - mn + 1e-9)
            scores.append(norm_val)
            scores_global.append(norm_glob)
        norm_aspek_cluster[aspek_name]  = np.mean(scores)
        norm_aspek_nasional[aspek_name] = np.mean(scores_global)

    aspek_names = list(ASPEK.keys())
    vals_cl  = [norm_aspek_cluster[a]  for a in aspek_names] + [norm_aspek_cluster[aspek_names[0]]]
    vals_nas = [norm_aspek_nasional[a] for a in aspek_names] + [norm_aspek_nasional[aspek_names[0]]]
    theta    = aspek_names + [aspek_names[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=vals_cl, theta=theta, fill='toself',
                                        name=cluster_sel, line_color=clr))
    fig_radar.add_trace(go.Scatterpolar(r=vals_nas, theta=theta, fill='toself',
                                        name='Rata-rata Nasional', line_color='#475569',
                                        line_dash='dash'))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor='#334155', color='#94a3b8'),
                   angularaxis=dict(gridcolor='#334155', color='#94a3b8', tickfont_size=13),
                   bgcolor='rgba(15,23,42,0.6)'),
        paper_bgcolor='rgba(15,23,42,0)', font_color='#e2e8f0',
        title=f'Radar 4 Aspek - {cluster_sel} vs Nasional',
        height=480, legend=dict(font_color='#ccc'))
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Ringkasan Semua Klaster per Aspek")
    rows = []
    for cl in all_clusters:
        sub_cl = df[df['Cluster_Label']==cl]
        row = {'Klaster': cl, 'Jumlah Provinsi': len(sub_cl), 'IKP Rata-rata': round(sub_cl['IKP'].mean(),2)}
        for aspek_name, aspek_info in ASPEK.items():
            scores = []
            for c in aspek_info['cols']:
                mn, mx = df[c].min(), df[c].max()
                scores.append((sub_cl[c].mean() - mn) / (mx - mn + 1e-9))
            row[f'Skor {aspek_name}'] = round(np.mean(scores), 3)
        rows.append(row)
    summary_df = pd.DataFrame(rows).set_index('Klaster')
    skor_cols = [f'Skor {a}' for a in ASPEK.keys()]
    st.dataframe(summary_df.style.background_gradient(subset=skor_cols, cmap='RdYlGn', axis=None)
                                  .background_gradient(subset=['IKP Rata-rata'], cmap='Blues'),
                 use_container_width=True)

    st.markdown("### Perbandingan Skor Aspek Antar Klaster")
    fig_asp = go.Figure()
    for i, aspek_name in enumerate(ASPEK.keys()):
        clr_a = list(ASPEK.values())[i]['color']
        fig_asp.add_trace(go.Bar(
            name=aspek_name,
            x=summary_df.index.tolist(),
            y=summary_df[f'Skor {aspek_name}'].tolist(),
            marker_color=clr_a,
        ))
    fig_asp.update_layout(barmode='group', **PLOT_THEME, height=420,
                          title='Skor 4 Aspek per Klaster (Normalisasi 0-1)',
                          legend=dict(font_color='#ccc'), yaxis_title='Skor (0-1)')
    st.plotly_chart(fig_asp, use_container_width=True)
