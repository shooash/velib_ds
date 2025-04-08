import streamlit as st

def show():

    st.markdown("""
        # 🚲 Projet Vélib’ – Amélioration du service via la data science

        Bienvenue dans la présentation ce projet visant à **améliorer le service Vélib’** opéré par **Smovengo**, grâce à l’analyse des données.

        ---

        ## 🔍 Constat

        Après analyse des retours d’utilisateurs réguliers et occasionnels, plusieurs **problématiques majeures** ont été identifiées :

        - ❌ **Maintenance insuffisante des vélos** :  
        De nombreux vélos sont défectueux (freins, chaînes, pédales...), et restent immobilisés plusieurs jours en station avant réparation.

        - 🚫 **Manque de disponibilité** :  
        Il est fréquent de ne **pas trouver de vélo disponible**, notamment aux heures de pointe ou dans les zones très fréquentées.  
        Certaines stations sont **vides**, d'autres **saturées**, empêchant même de restituer un vélo.

        ---

        ## 🎯 Objectif du projet

        Nous avons choisi de **nous concentrer sur la disponibilité des vélos**.

        Notre but :  
        ➡️ **Prédire le flux de vélos dans chaque station** pour comprendre les dynamiques d’arrivée et de départ.

        Cela permettra de :
        - **Anticiper la demande** et les déséquilibres entre les stations.
        - **Améliorer la répartition de la flotte** à travers la ville.

        ---

        ## 📊 Ce que nous proposons

        - Entraînement de modèles prédictifs.
        - Mise en place d’un **dashboard interactif** :
            - Visualisation en temps réel des stations.
            - Alertes pour repérer les stations en **sous-effectif** ou **surcharge**.
            - Aide à la **régulation de la flotte** pour garantir un meilleur service.

        ---

        Nous espérons qu'à travers cette approche, **le service Vélib’ pourra devenir plus fiable et plus fluide pour tous les utilisateurs**.
        """, unsafe_allow_html=True)
    st.write("Ce projet est réalisé par Louise Poirey et Andre ")