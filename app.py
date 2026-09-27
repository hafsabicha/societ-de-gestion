# -*- coding: utf-8 -*-
"""
================================================================================
 MENARA PREFA - Application de gestion d'entreprise (Streamlit)
--------------------------------------------------------------------------------
 Société de préfabrication béton (Marrakech).
 Basée sur le schéma relationnel : CLIENT, EMPLOYE, DEPARTEMENT, FOURNISSEUR,
 PRODUIT, COMMANDE, CONTENIR, ACHETER, PROJET, TRAVAILLER.
 Les données sont dénormalisées dans un fichier unique : dataset.csv
================================================================================
"""

import os
import io
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ------------------------------------------------------------------------------
APP_TITLE = "Ménara Préfa — Gestion d'entreprise"
DATA_FILE = os.path.join(os.path.dirname(__file__), "dataset.csv")
MOT_DE_PASSE = "minara hafs"  # mot de passe demandé pour accéder à l'application

st.set_page_config(
    page_title="Ménara Préfa | Gestion",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["figure.facecolor"] = "none"

# ------------------------------------------------------------------------------
# STYLE CSS - INTERFACE MODERNE
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #111827 100%); }
    h1, h2, h3, h4, .stMarkdown, label, p, span { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }
    div[data-testid="stMetric"] {
        background: #111c33;
        border: 1px solid #24334d;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .kmenara-card {
        background: #111c33;
        border: 1px solid #24334d;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .stButton>button {
        background: linear-gradient(90deg,#d97706,#b45309);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; padding: 0.5rem 1.1rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .stTextInput>div>div>input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]>div {
        background-color: #0b1220 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
    hr { border-color: #24334d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 1) ECRAN DE CONNEXION (MOT DE PASSE)
# ==============================================================================
def ecran_connexion():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='kmenara-card' style='text-align:center;'>
                <h1>🏗️ MENARA PREFA</h1>
                <p style='color:#94a3b8;'>Plateforme de gestion d'entreprise — préfabrication béton</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("form_connexion"):
            mdp = st.text_input("Mot de passe", type="password", placeholder="Entrez le mot de passe")
            valider = st.form_submit_button("Se connecter", use_container_width=True)
            if valider:
                if mdp.strip().lower() == MOT_DE_PASSE:
                    st.session_state["connecte"] = True
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect. Veuillez réessayer.")
        st.caption("Accès réservé au personnel autorisé de Ménara Préfa.")


if "connecte" not in st.session_state:
    st.session_state["connecte"] = False

if not st.session_state["connecte"]:
    ecran_connexion()
    st.stop()

# ==============================================================================
# 2) CHARGEMENT + NETTOYAGE DES DONNEES
# ==============================================================================
COLONNES_ATTENDUES = [
    "id_commande", "date_commande", "client", "ville_client", "telephone_client",
    "employe", "departement", "produit", "categorie_produit", "quantite",
    "prix_unitaire", "prix_total", "fournisseur", "ville_fournisseur",
    "statut_commande", "projet", "budget_projet", "heures_travaillees",
]


@st.cache_data(show_spinner=False)
def charger_et_nettoyer(chemin: str, version: int) -> pd.DataFrame:
    """
    Charge le CSV et applique un nettoyage :
    - suppression des doublons
    - normalisation du texte (espaces, casse)
    - conversion des types (dates, nombres)
    - recalcul de prix_total si incohérent
    - suppression des lignes sans identifiant de commande
    'version' force le cache à se rafraîchir après chaque enregistrement.
    """
    if not os.path.exists(chemin):
        return pd.DataFrame(columns=COLONNES_ATTENDUES)

    df = pd.read_csv(chemin, encoding="utf-8-sig")

    # Colonnes manquantes -> on les crée vides pour éviter les erreurs
    for col in COLONNES_ATTENDUES:
        if col not in df.columns:
            df[col] = np.nan

    # Nettoyage des chaînes de caractères
    colonnes_texte = [
        "client", "ville_client", "employe", "departement", "produit",
        "categorie_produit", "fournisseur", "ville_fournisseur",
        "statut_commande", "projet",
    ]
    for col in colonnes_texte:
        df[col] = (
            df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        )
        df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    # Types numériques
    for col in ["quantite", "prix_unitaire", "prix_total", "budget_projet", "heures_travaillees"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    df["date_commande"] = pd.to_datetime(df["date_commande"], errors="coerce")

    # Recalcul du prix total quand possible (cohérence quantite * prix_unitaire)
    mask = df["quantite"].notna() & df["prix_unitaire"].notna()
    df.loc[mask, "prix_total"] = (df.loc[mask, "quantite"] * df.loc[mask, "prix_unitaire"]).round(2)

    # Suppression des doublons stricts et des lignes sans identifiant
    df = df.dropna(subset=["id_commande"])
    df = df.drop_duplicates(subset=["id_commande"], keep="last")

    # Valeurs par défaut pour le statut manquant
    df["statut_commande"] = df["statut_commande"].fillna("En attente")

    df = df.sort_values("date_commande").reset_index(drop=True)
    return df


def enregistrer(df: pd.DataFrame):
    """Ecrit le DataFrame sur disque (dataset.csv) et invalide le cache."""
    df_to_save = df.copy()
    df_to_save["date_commande"] = df_to_save["date_commande"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    st.session_state["version_donnees"] = st.session_state.get("version_donnees", 0) + 1


if "version_donnees" not in st.session_state:
    st.session_state["version_donnees"] = 0

df = charger_et_nettoyer(DATA_FILE, st.session_state["version_donnees"])

# ==============================================================================
# 3) BARRE LATERALE - NAVIGATION + FILTRES
# ==============================================================================
st.sidebar.markdown("## 🏗️ Ménara Préfa")
st.sidebar.caption("Gestion d'entreprise — préfabrication béton")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Tableau de bord", "📋 Données & filtres", "➕ Nouvelle commande", "ℹ️ À propos / Aide"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filtres")

if df.empty:
    st.sidebar.info("Aucune donnée disponible pour le moment.")
    df_filtre = df.copy()
else:
    min_date = df["date_commande"].min()
    max_date = df["date_commande"].max()
    periode = st.sidebar.date_input(
        "Période (date de commande)",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    clients_sel = st.sidebar.multiselect("Client", sorted(df["client"].dropna().unique()))
    produits_sel = st.sidebar.multiselect("Produit", sorted(df["produit"].dropna().unique()))
    fournisseurs_sel = st.sidebar.multiselect("Fournisseur", sorted(df["fournisseur"].dropna().unique()))
    departements_sel = st.sidebar.multiselect("Département", sorted(df["departement"].dropna().unique()))
    statuts_sel = st.sidebar.multiselect("Statut de commande", sorted(df["statut_commande"].dropna().unique()))

    df_filtre = df.copy()
    if isinstance(periode, tuple) and len(periode) == 2:
        d1, d2 = periode
        df_filtre = df_filtre[
            (df_filtre["date_commande"].dt.date >= d1) & (df_filtre["date_commande"].dt.date <= d2)
        ]
    if clients_sel:
        df_filtre = df_filtre[df_filtre["client"].isin(clients_sel)]
    if produits_sel:
        df_filtre = df_filtre[df_filtre["produit"].isin(produits_sel)]
    if fournisseurs_sel:
        df_filtre = df_filtre[df_filtre["fournisseur"].isin(fournisseurs_sel)]
    if departements_sel:
        df_filtre = df_filtre[df_filtre["departement"].isin(departements_sel)]
    if statuts_sel:
        df_filtre = df_filtre[df_filtre["statut_commande"].isin(statuts_sel)]

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Se déconnecter"):
    st.session_state["connecte"] = False
    st.rerun()

# ==============================================================================
# 4) PAGE : TABLEAU DE BORD
# ==============================================================================
if page == "📊 Tableau de bord":
    st.title("📊 Tableau de bord — Ménara Préfa")
    st.caption(f"Dernière mise à jour de l'affichage : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    if df_filtre.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    else:
        # --- KPI ---
        ca_total = df_filtre["prix_total"].sum()
        nb_commandes = df_filtre["id_commande"].nunique()
        panier_moyen = df_filtre["prix_total"].mean()
        heures_total = df_filtre["heures_travaillees"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Chiffre d'affaires", f"{ca_total:,.0f} MAD".replace(",", " "))
        c2.metric("📦 Commandes", f"{nb_commandes}")
        c3.metric("🧾 Panier moyen", f"{panier_moyen:,.0f} MAD".replace(",", " "))
        c4.metric("⏱️ Heures travaillées", f"{heures_total:,.0f} h".replace(",", " "))

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Chiffre d'affaires par produit")
            fig, ax = plt.subplots(figsize=(6, 4))
            data_prod = df_filtre.groupby("produit")["prix_total"].sum().sort_values(ascending=False)
            sns.barplot(x=data_prod.values, y=data_prod.index, ax=ax, color="#d97706")
            ax.set_xlabel("Chiffre d'affaires (MAD)")
            ax.set_ylabel("")
            ax.tick_params(colors="#f1f5f9")
            ax.xaxis.label.set_color("#f1f5f9")
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.subheader("Répartition des statuts de commande")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            statut_counts = df_filtre["statut_commande"].value_counts()
            ax2.pie(
                statut_counts.values,
                labels=statut_counts.index,
                autopct="%1.0f%%",
                colors=sns.color_palette("Set2"),
                textprops={"color": "#0b1220", "fontweight": "bold"},
            )
            ax2.set_ylabel("")
            st.pyplot(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Évolution mensuelle des commandes")
            serie = df_filtre.set_index("date_commande").resample("ME")["prix_total"].sum()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(serie.index, serie.values, marker="o", color="#22c55e")
            ax3.set_ylabel("Chiffre d'affaires (MAD)")
            ax3.tick_params(axis="x", rotation=45, colors="#f1f5f9")
            ax3.tick_params(axis="y", colors="#f1f5f9")
            ax3.yaxis.label.set_color("#f1f5f9")
            st.pyplot(fig3, use_container_width=True)

        with col4:
            st.subheader("Top fournisseurs (volume d'achat)")
            data_fourn = df_filtre.groupby("fournisseur")["prix_total"].sum().sort_values(ascending=False).head(6)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=data_fourn.values, y=data_fourn.index, ax=ax4, color="#3b82f6")
            ax4.set_xlabel("Montant (MAD)")
            ax4.set_ylabel("")
            ax4.tick_params(colors="#f1f5f9")
            ax4.xaxis.label.set_color("#f1f5f9")
            st.pyplot(fig4, use_container_width=True)

        st.markdown("---")
        st.subheader("Charge de travail par département (heures)")
        fig5, ax5 = plt.subplots(figsize=(10, 3.5))
        data_dept = df_filtre.groupby("departement")["heures_travaillees"].sum().sort_values(ascending=False)
        sns.barplot(x=data_dept.index, y=data_dept.values, ax=ax5, color="#a855f7")
        ax5.set_ylabel("Heures")
        ax5.tick_params(colors="#f1f5f9")
        ax5.yaxis.label.set_color("#f1f5f9")
        st.pyplot(fig5, use_container_width=True)

# ==============================================================================
# 5) PAGE : DONNEES & FILTRES
# ==============================================================================
elif page == "📋 Données & filtres":
    st.title("📋 Données détaillées")
    st.caption("Les filtres de la barre latérale s'appliquent au tableau ci-dessous.")

    st.write(f"**{len(df_filtre)}** ligne(s) affichée(s) sur **{len(df)}** au total.")
    st.dataframe(df_filtre, use_container_width=True, height=460)

    buffer = io.StringIO()
    df_filtre.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Télécharger la sélection (CSV)",
        data=buffer.getvalue().encode("utf-8-sig"),
        file_name="menara_prefa_export.csv",
        mime="text/csv",
    )

    st.markdown("---")
    with st.expander("🧹 Détail du nettoyage automatique appliqué"):
        st.markdown(
            """
            À chaque chargement, `dataset.csv` est nettoyé automatiquement :
            - suppression des espaces superflus et normalisation du texte,
            - conversion des colonnes numériques et des dates,
            - recalcul de `prix_total = quantité × prix_unitaire`,
            - suppression des doublons sur `id_commande`,
            - remplacement des statuts manquants par **« En attente »**.
            """
        )

# ==============================================================================
# 6) PAGE : NOUVELLE COMMANDE (SAISIE UTILISATEUR)
# ==============================================================================
elif page == "➕ Nouvelle commande":
    st.title("➕ Enregistrer une nouvelle commande")
    st.caption("Les champs ci-dessous correspondent aux tables CLIENT, PRODUIT, FOURNISSEUR, EMPLOYE, PROJET du modèle de données.")

    with st.form("form_commande", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            client = st.text_input("Client *")
            ville_client = st.text_input("Ville du client")
            telephone_client = st.text_input("Téléphone du client")
            employe = st.text_input("Employé en charge *")
            departement = st.selectbox(
                "Département",
                ["Production", "Commercial", "Logistique", "Finance", "Ressources Humaines", "Qualité"],
            )
            projet = st.text_input("Projet / Chantier")
            budget_projet = st.number_input("Budget du projet (MAD)", min_value=0.0, step=1000.0)

        with col2:
            produit = st.text_input("Produit *")
            categorie_produit = st.selectbox(
                "Catégorie produit",
                ["Structure", "Façade", "Maçonnerie", "Fondation", "Aménagement", "Autre"],
            )
            quantite = st.number_input("Quantité *", min_value=1, step=1)
            prix_unitaire = st.number_input("Prix unitaire (MAD) *", min_value=0.0, step=10.0)
            fournisseur = st.text_input("Fournisseur")
            ville_fournisseur = st.text_input("Ville du fournisseur")
            statut_commande = st.selectbox("Statut de la commande", ["En attente", "En cours", "Livrée", "Annulée"])
            heures_travaillees = st.number_input("Heures travaillées estimées", min_value=0, step=1)
            date_commande = st.date_input("Date de la commande", value=date.today())

        envoyer = st.form_submit_button("💾 Enregistrer la commande", use_container_width=True)

        if envoyer:
            champs_obligatoires = {"Client": client, "Employé": employe, "Produit": produit}
            manquants = [nom for nom, val in champs_obligatoires.items() if not val.strip()]
            if manquants:
                st.error(f"Champs obligatoires manquants : {', '.join(manquants)}")
            else:
                nouvel_id = f"CMD-{(len(df) + 1):04d}"
                # on s'assure de l'unicité de l'identifiant même après suppressions
                ids_existants = set(df["id_commande"]) if not df.empty else set()
                compteur = len(df) + 1
                while nouvel_id in ids_existants:
                    compteur += 1
                    nouvel_id = f"CMD-{compteur:04d}"

                nouvelle_ligne = pd.DataFrame([{
                    "id_commande": nouvel_id,
                    "date_commande": pd.to_datetime(date_commande),
                    "client": client.strip(),
                    "ville_client": ville_client.strip(),
                    "telephone_client": telephone_client.strip(),
                    "employe": employe.strip(),
                    "departement": departement,
                    "produit": produit.strip(),
                    "categorie_produit": categorie_produit,
                    "quantite": quantite,
                    "prix_unitaire": prix_unitaire,
                    "prix_total": round(quantite * prix_unitaire, 2),
                    "fournisseur": fournisseur.strip(),
                    "ville_fournisseur": ville_fournisseur.strip(),
                    "statut_commande": statut_commande,
                    "projet": projet.strip(),
                    "budget_projet": budget_projet,
                    "heures_travaillees": heures_travaillees,
                }])

                df_maj = pd.concat([df, nouvelle_ligne], ignore_index=True)
                enregistrer(df_maj)
                st.success(f"✅ Commande **{nouvel_id}** enregistrée avec succès dans dataset.csv !")
                st.balloons()
                st.rerun()

# ==============================================================================
# 7) PAGE : A PROPOS / AIDE (EXPLICATION LIAISON DONNEES + GITHUB)
# ==============================================================================
else:
    st.title("ℹ️ À propos de l'application")

    st.markdown(
        """
        <div class='kmenara-card'>
        <h3>🏗️ Ménara Préfa — Application de gestion</h3>
        <p>Cette application couvre les processus de l'entreprise décrits dans le modèle de données :
        clients, employés, départements, fournisseurs, produits, commandes, achats et projets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🔗 Comment fonctionne l'enregistrement des données ?")
    st.markdown(
        """
        1. **Lecture** : au démarrage, l'application lit le fichier `dataset.csv` avec `pandas.read_csv`,
           puis le nettoie (types, doublons, textes) via la fonction `charger_et_nettoyer()`.
        2. **Saisie** : la page *Nouvelle commande* affiche un formulaire Streamlit (`st.form`). Les valeurs
           saisies sont validées puis ajoutées au DataFrame en mémoire avec `pandas.concat`.
        3. **Écriture** : la fonction `enregistrer()` réécrit **tout le fichier `dataset.csv`** sur disque avec
           `DataFrame.to_csv(...)`. Le cache Streamlit (`st.cache_data`) est invalidé (via un compteur de version)
           pour que la prochaine lecture recharge bien les données à jour.
        4. **Rafraîchissement** : `st.rerun()` relance l'application pour que les dashboards, filtres et tableaux
           reflètent immédiatement la nouvelle commande.
        """
    )

    st.subheader("🐙 Quel est le rôle de GitHub dans ce système ?")
    st.markdown(
        """
        Le fichier `dataset.csv` vit **localement** sur le serveur qui exécute Streamlit. Deux cas de figure :

        - **En local (votre PC)** : les écritures sur `dataset.csv` sont permanentes tant que vous gardez le
          dossier du projet. GitHub sert alors d'**historique et de sauvegarde** : après chaque session de travail,
          vous pouvez faire `git add dataset.csv`, `git commit -m "maj données"`, puis `git push` pour garder
          une trace de chaque version des données (comme un système de versions/rollback).

        - **Sur Streamlit Community Cloud** (hébergement gratuit à partir d'un dépôt GitHub) : le système de
          fichiers est **éphémère** — à chaque redéploiement ou redémarrage du conteneur, les fichiers non commités
          sur GitHub sont perdus. Dans ce cas, GitHub n'est pas seulement une sauvegarde, il devient la
          **source de vérité** : Streamlit Cloud redéploie automatiquement l'app à chaque `git push` sur la branche
          connectée, et le `dataset.csv` du dépôt sert de point de départ à chaque redémarrage.

        **Recommandations pour une vraie continuité en production :**
        - committer régulièrement `dataset.csv` sur GitHub (manuellement, ou via une action planifiée) ;
        - ou, pour un usage intensif, migrer vers une base de données persistante (SQLite, PostgreSQL, Supabase...)
          reliée à l'app par une variable d'environnement / un secret Streamlit, ce qui évite toute perte de données
          liée au caractère éphémère du stockage cloud.
        """
    )

    st.subheader("🧰 Technologies utilisées")
    st.markdown(
        """
        - **Streamlit** — interface web interactive
        - **Pandas** — lecture, nettoyage et manipulation des données
        - **Matplotlib / Seaborn** — visualisations des dashboards
        - **CSV (`dataset.csv`)** — stockage simple, portable et lisible par tous les outils (Excel, Git, etc.)
        """
    )

    st.caption("Ménara Préfa © 2026 — Application développée avec Streamlit.")
