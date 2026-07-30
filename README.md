# Advanced Crystallographic Simulator (IITI SOC 2026)

An integrated computational software platform that combines first-principles crystallographic calculations with modern deep learning techniques to perform both **Forward XRD Simulation** and **AI-Based Backward Material Identification** within a unified PySide6 desktop dashboard.

Developed by **Jainesh Roy (Roll No: 250005019)** and team members from the Department of Metallurgical Engineering and Materials Science, Indian Institute of Technology Indore.

---

## 🚀 Key Features

* **Forward Simulation Engine:** Generates realistic X-ray Diffraction (XRD) patterns directly from basic crystal configurations (SC, BCC, FCC), applying selection rules, structure factors, intensity corrections, and real-world peak broadening (Gaussian/Lorentzian).
* **AI Backward Identification Engine:** Replaces traditional database peak-matching with a data-driven 1D Convolutional Neural Network (CNN) built via PyTorch. It identifies materials from raw experimental data, yielding ranked predictions and confidence scores.
* **Synthetic Data Pipeline:** Generates thousands of diverse, noisy training patterns with randomized crystallite sizes, instrument shifts, and background baselines to create robust ML datasets.
* **Interactive UI Dashboard:** A modern desktop application built using PySide6 featuring embedded interactive Matplotlib subplots for seamless zooming, panning, and peak analysis.

---

## 📊 System Processing Pipeline

Below is the computational data flow mapping the complete engine lifecycle:

```text
       ┌───────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
START ─►│User Input ├─────►│ Generate Miller Indices ├─────►│Apply Reflection Selection├─────►│ Compute Diffraction Peaks│
       │• Material │      │  (h, k, l) Reflections  │      │  Rules (SC / BCC / FCC)  │      │• d-spacing               │
       │• Crystal  │      └─────────────────────────┘      └──────────────────────────┘      │• Bragg's Law             │
       │• Lattice  │                                                                         │• Structure Factor        │
       │• X-ray    │                                                                         │• Relative Intensity      │
       │• Sim Pars │                                                                         └──────────────┬───────────┘
       └───────────┘                                                                                        │
                                                                                                            │
 ┌─────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────▼───┐
 │   END   │◄─────┤Display & Export Results │◄─────┤Generate Continuous Profile│◄─────┤Apply Physical Corrections│
 └─────────┘      │• XRD Plot               │      │• Peak Broadening         │      │• Multiplicity            │
                  │• CSV File               │      │• Noise                   │      │• Lorentz-Polarization    │
                  │• Plot Image             │      │• Background              │      │• Debye-Waller Factor     │
                  └─────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
