# CulturalUnlearn

Post-training unlearning framework for reducing Western socio-cultural bias in Arabic language models.

## About

Arabic LLMs systematically prefer Western entities over culturally appropriate Arab ones (e.g., completing "after Maghrib prayer, drink..." with *wine* instead of *coffee*). CulturalUnlearn is a post-training, parameter-modifying framework that targets this bias without retraining the model from scratch, using four objectives (GABG, CCUT, PARD, TISS) that each target a specific linguistic mechanism behind the bias.

## Structure

```text
Assignment/ screening assignment
data/         Datasets (forget set, retain set, probe sets)
src/          Framework implementation
docs/         Additional documentation
```