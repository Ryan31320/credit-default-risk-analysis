# Méthodologie et limites

## Préparation

Le fichier brut comporte 30 000 lignes. Le notebook retire 35 lignes identiques hors identifiant,
puis crée un split stratifié reproductible : 23 972 observations pour l’entraînement et
5 993 pour le test. Cette déduplication est une convention ; elle ne démontre pas que les
lignes concernent les mêmes personnes. L’EDA porte sur le train.

Le notebook étudie les codes de catégories, les distributions et les valeurs atypiques.
Cinq variables dérivées résument les comportements de paiement. Les transformations
et les modèles nécessitant une standardisation sont décrits dans l’étude.

## Sélection et décision

Après la comparaison initiale, dix familles sont optimisées avec Optuna : 5 plis externes,
5 plis internes et 50 essais par recherche. Une variante de soft voting est également comparée.
Le SVM est étudié initialement mais exclu du tuning imbriqué en raison de son coût.
La ROC-AUC moyenne externe guide la sélection de la famille.

Une nouvelle recherche est menée sur le train. Les prédictions hors pli du train permettent
de choisir le seuil maximisant le F1 (environ 0,54), avant l’évaluation finale sur le test.
Les tableaux initiaux constituent des diagnostics ; le résultat final à citer est celui
présenté dans le README et en section 3.5 du notebook.

## Interprétation

L’étude comprend les courbes ROC et precision-recall, la matrice de confusion,
les courbes d’apprentissage, la calibration, les importances par permutation,
les dépendances partielles et les contributions SHAP de CatBoost.

## Limites

- Données historiques de Taïwan en 2005 : la généralisation à une autre période ou banque reste à établir.
- CatBoost utilise une pondération des classes ; ses scores doivent être recalibrés avant d’être interprétés comme probabilités de défaut.
- Les petites différences entre modèles ne démontrent pas une supériorité générale.
- La sélection parmi plusieurs familles et l’optimisation du seuil rendent les scores OOF finaux optimistes ; ils ne sont pas une estimation indépendante.
- Le seuil F1 ne reflète pas un coût économique bancaire explicite.
- Les variables corrélées compliquent l’interprétation ; les explications ne prouvent pas une causalité.
- Les diagnostics par groupes, la validation temporelle et le suivi de dérive restent des pistes de travail.

Il s’agit d’une étude de machine learning sur données publiques, sans déploiement bancaire.
