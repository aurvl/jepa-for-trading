# Architecture du JEPA Trading Bot

Ce document explique l'architecture de `jepa-for-trading`, le rôle de chaque
bloc, et le processus complet depuis les données jusqu'au backtest final.

## 1. Idée générale

Le projet construit un agent de trading multi-assets basé sur deux idées :

1. Apprendre une représentation latente du marché avec un modèle JEPA.
2. Utiliser cette représentation dans un agent RL direct qui gère un
   portefeuille réel avec cash, frais, poids, turnover et drawdown.

Le point important est que l'action de l'agent ne modifie pas le marché. Dans
ce cadre daily retail/research, l'agent n'a pas de market impact. Le modèle
prédit donc l'évolution latente du marché indépendamment de l'action, puis
l'action détermine uniquement l'exposition du portefeuille et donc le PnL,
les coûts, le risque et le drawdown.

## 2. Pipeline de données

Les prix viennent de `yfinance`. La macro vient de `macro_data.parquet` ou du
CSV `estimated_volatility_with_macro.csv` fourni sur Kaggle. Si le CSV est
utilisé, les colonnes `price` et `sigma` sont retirées du bloc macro, car la
volatilité doit être recalculée actif par actif.

Pour chaque actif, le pipeline construit :

- `log_return`;
- `overnight_gap`;
- `intraday_range`;
- features de volume;
- RSI normalisé;
- MACD;
- volatilité réalisée rolling;
- `sigma` par actif, calculée depuis son propre historique.

Les actifs peuvent commencer à des dates différentes. Le pipeline ne backfill
pas avant l'existence réelle d'un actif. Chaque date conserve un
`tradable_mask`, utilisé ensuite par le modèle et par l'agent pour empêcher
une allocation vers un actif non tradable.

Les splits sont temporels :

- train : apprentissage JEPA et heads;
- validation : checkpointing et early selection;
- test : backtest final jamais vu.

Les scalers sont fit uniquement sur train pour éviter le leakage.

## 3. Dataset multi-assets

Le dataset transforme le marché en fenêtres daily de 70 jours. Chaque sample
contient :

- `context` : fenêtre passée observée par l'encoder online;
- `target` : fenêtre future réelle observée par l'encoder EMA;
- `context_mask` et `target_mask` : disponibilité historique des actifs;
- `tradable_mask` : actifs réellement tradables à la date de décision;
- `horizon` : horizon de prédiction parmi `5, 15, 20, 45, 60`;
- targets auxiliaires : return futur, sigma future, drawdown futur.

La convention tensor principale est :

```text
batch x assets x time x features
```

## 4. Modèle JEPA de marché

Le JEPA est composé de trois blocs :

```text
online_encoder(context) -> z_context
predictor(z_context, horizon) -> z_future_hat
target_encoder_EMA(target) -> z_future
```

L'encoder online apprend normalement par gradient. L'encoder target est une
copie EMA de l'encoder online, mise à jour progressivement. Le predictor est
conditionné par l'horizon, ce qui permet au même modèle de prédire des
représentations futures à 5, 15, 20, 45 ou 60 jours.

La loss JEPA compare les latents normalisés :

```text
loss = distance(z_future_hat, stop_gradient(z_future))
```

Elle est masquée par `tradable_mask`, pour ne pas apprendre sur des actifs
non disponibles.

Le JEPA ne reconstruit pas les prix. Il apprend une représentation latente du
futur marché, plus proche de l'esprit I-JEPA/V-JEPA que d'un autoencoder
classique.

## 5. Market heads

Les heads transforment les latents JEPA en signaux financiers exploitables
par l'agent :

- quantiles de return : p10, p25, p50, p75, p90;
- sigma future par actif;
- drawdown futur;
- probabilité de return positif.

Ces heads ne remplacent pas le JEPA. Ils servent d'interface entre le monde
latent appris et la décision de portefeuille.

## 6. Environnement de trading

L'environnement simule une comptabilité de portefeuille daily :

- equity;
- cash;
- poids courants;
- turnover;
- coûts de transaction;
- drawdown courant;
- masque des actifs tradables.

A chaque step :

1. l'agent observe l'état de marché et son portefeuille;
2. il propose des poids cibles;
3. l'environnement applique les contraintes;
4. les frais sont retirés;
5. les returns du jour suivant modifient l'equity;
6. la reward est calculée.

Les contraintes V1 sont :

- long-only;
- cash autorisé;
- pas de leverage;
- poids maximum par actif;
- turnover maximum optionnel;
- impossible d'acheter un actif non tradable.

## 7. Agent PPO direct

L'agent est une policy PPO qui produit directement des target weights. Il ne
choisit pas un simple `buy/sell/hold` discret, car cela scale mal avec un
portefeuille de dizaines d'actifs.

Observation de l'agent :

- latents JEPA;
- outputs des market heads;
- validité des actifs;
- état du portefeuille : equity, cash, poids actuels, turnover, drawdown.

Action de l'agent :

```text
target_weights = [w_asset_1, ..., w_asset_n, w_cash]
```

La policy utilise une sortie softmax, puis l'environnement applique les
contraintes financières. La reward est risk-adjusted :

```text
reward =
    log_return_portfolio
    - transaction_costs
    - drawdown_penalty
    - turnover_penalty
    - concentration_penalty
```

Cette reward force l'agent à apprendre une gestion de portefeuille, pas
seulement une maximisation brute du PnL.

## 8. Evaluation

Le backtest final est lancé sur la période test uniquement. L'agent est
comparé à plusieurs baselines :

- Buy & Hold;
- equal weight;
- momentum simple;
- volatility targeting;
- 100 stratégies random long-only.

Les métriques calculées incluent :

- total return;
- CAGR;
- volatilité annualisée;
- Sharpe;
- Sortino;
- max drawdown;
- hit rate;
- turnover moyen;
- coûts totaux;
- poids cash moyen.

Les plots principaux sont :

- courbes d'equity;
- random strategies en gris;
- Buy & Hold en blanc;
- agent JEPA-PPO en vert fluo;
- drawdown;
- turnover.

Le test statistique principal est une p-value empirique :

```text
p_value = proportion des stratégies random qui battent l'agent
```

Une p-value faible indique que la performance de l'agent est difficile à
expliquer par un comportement random sous les mêmes contraintes.

## 9. Workflow Kaggle

Le notebook `notebooks/kaggle_run_jepa_trading.ipynb` orchestre tout :

1. clone la branche `version1`;
2. installe le package avec le Python du kernel;
3. ajoute `src/` au `sys.path` pour éviter les problèmes Kaggle;
4. détecte le fichier macro dans `/kaggle/input`;
5. prépare les données multi-assets;
6. train le JEPA en `max_steps`;
7. train les market heads;
8. train PPO;
9. lance le backtest;
10. affiche métriques, plots et tests statistiques.

Le notebook contient aussi une cellule optionnelle pour pousser les artefacts
vers GitHub après entraînement. Elle utilise :

- `GITHUB_TOKEN` depuis Kaggle Secrets;
- `git lfs` si disponible;
- `git add .`;
- `git add -f` pour les dossiers d'artefacts ignorés par défaut;
- commit et push vers `version1`.

## 10. Limites de la V1

Cette V1 est une architecture complète mais reste une base de recherche :

- les données `yfinance` ne sont pas survivorship-bias-free;
- les coûts sont simplifiés;
- il n'y a pas encore de borrow cost ni de short;
- pas de market impact;
- PPO peut surfit si la validation temporelle est faible;
- le nombre d'actifs et les hyperparamètres doivent être testés sérieusement.

La direction naturelle pour V2 est :

- univers plus large et plus propre;
- meilleure modélisation cross-assets;
- covariance/risk model explicite;
- walk-forward validation;
- policy distillation depuis un planner JEPA;
- contraintes de portefeuille plus institutionnelles.

