# Progetto: World Model Visivi per Manipolazione Robotica

**Studenti**: 2 | **Durata**: 8 settimane | **Compute**: cluster A100/H100 (SLURM)

## Obiettivo

Capire se la scelta del backbone visivo pre-addestrato cambia in modo significativo le performance di un world model per manipolazione robotica.

In pratica: si parte da DINO-WM (Zhou et al., 2024), si sostituisce DINOv2 con altri encoder (CLIP, SigLIP, MAE), si addestra il modello sugli stessi task, si confrontano i risultati.

## Cos'è un world model (in breve)

Un modello che predice come evolve la scena dopo una sequenza di azioni. Si addestra su un dataset offline di traiettorie del tipo $(o_1, a_1, o_2, a_2, \ldots)$, dove $o_t$ è un frame RGB e $a_t$ è l'azione applicata. A test time si usa per fare planning: si simulano "in immaginazione" diverse sequenze di azioni, si sceglie quella che porta più vicino al goal, si esegue.

DINO-WM lavora nello spazio delle feature DINOv2 invece che nello spazio dei pixel. L'encoder è "congelato"; si addestra solo un Transformer che predice token visivi futuri dati quelli passati e l'azione.

## Setup

**Codebase**: [`gaoyuezhou/dino_wm`](https://github.com/gaoyuezhou/dino_wm) — include i task 2D del paper con dataset di traiettorie già pronti da scaricare.

**Simulatore per estensione 3D**: [ManiSkill 3](https://github.com/haosulab/ManiSkill) — fornisce task di manipolazione con braccio Franka e scripted solutions per generare automaticamente le traiettorie di training.

**Stack**: Python, PyTorch, HuggingFace.

## Task

1. **PushT** (2D, dal paper DINO-WM) — pushing planare di una T verso una posa target. Baseline da riprodurre.
2. **Wall** (2D, dal paper DINO-WM) — navigazione con ostacoli. Per il test di generalizzazione.
3. **PushCube** (3D, ManiSkill 3) — pushing 3D con braccio Franka. Estensione del lavoro originale.

## Backbone da confrontare

- **DINOv2** (self-distillation) — baseline del paper
- **CLIP** (contrastive image-text)
- **SigLIP** (variante più recente di CLIP)
- **MAE** (masked autoencoder)

## Metriche

**Primaria**: *success rate* del planning, cioè quante volte si raggiunge il goal usando il world model.

**Di supporto**: errore di predizione dei token futuri, qualità visiva dei rollout decodificati, generalizzazione a oggetti/configurazioni non viste in training.

## Reading list essenziale

In ordine di lettura:

1. **Ha & Schmidhuber (2018)** — *World Models*. arXiv:1803.10122. Lettura veloce per il framing concettuale.
2. **Oquab et al. (2023)** — *DINOv2*. arXiv:2304.07193. Capire cosa producono le patch features (token visivi).
3. **Zhou et al. (2024)** — *DINO-WM*. arXiv:2411.04983. Paper centrale, da studiare a fondo insieme al codice.
4. **Tao et al. (2024)** — *ManiSkill 3*. arXiv:2410.00425. Documentazione del simulatore (lettura selettiva, solo per il task PushCube).

## Cronoprogramma

**Settimane 1–2 — Riproduzione baseline**
Setup ambiente e codebase DINO-WM. Run di training su PushT con DINOv2. Verifica che le metriche del paper siano riproducibili (entro ±15%).

**Settimane 3–4 — Confronto backbone sui task 2D**
Implementazione dell'astrazione "backbone interchangeable". Training dei 4 backbone su PushT e Wall. Prima tabella di confronto.

**Settimana 5 — Estensione a ManiSkill 3 (eventuale)**
Generazione delle traiettorie di training su PushCube tramite gli scripted solver di ManiSkill 3. Training dei backbone sul nuovo task.

**Settimana 6 — Esperimenti completi e analisi**
Re-run con 3 seed per significatività. Visualizzazione qualitativa dei rollout (success vs failure). Analisi dei failure mode per ciascun backbone.

**Settimana 7 — Stesura report**
Bozza completa.

**Settimana 8 — Finale**
Revisione, presentazione, repo pulita.

## Deliverable

1. Report scritto (8-12 pagine, template NeurIPS o simile).
2. Repository Git con codice riproducibile e README.
3. Presentazione finale (15-20 min).

## Riferimenti

- Ha, D. & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
- He, K. et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. arXiv:2111.06377.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.
- Radford, A. et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision* (CLIP). arXiv:2103.00020.
- Tao, S. et al. (2024). *ManiSkill 3*. arXiv:2410.00425.
- Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training* (SigLIP). arXiv:2303.15343.
- Zhou, G., Pan, H., LeCun, Y. & Pinto, L. (2024). *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*. arXiv:2411.04983.
