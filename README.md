This repository contains the simulation code and analysis pipeline for the manuscript:
**Inhibitory Synaptic Timescales Induce State Transitions Between Distinct Oscillatory Regimes in Recurrent Cortical Networks**


### Overview

Cortical networks exhibit oscillatory activity across multiple frequency bands, including gamma (30–80 Hz) and high-frequency oscillations (>100 Hz).
In this work, we investigate how inhibitory synaptic decay time constants influence the emergence and stability of oscillatory regimes in recurrent excitatory–inhibitory (E–I) networks.

Using conductance-based leaky integrate-and-fire neurons implemented in NEST, we systematically vary:
  - Inhibitory → Inhibitory synaptic decay time
  - Inhibitory → Excitatory synaptic decay time
while keeping excitatory synaptic dynamics fixed.

We show that inhibitory synaptic timescale acts as a control parameter inducing sharp transitions between:
  - Coherent high-frequency oscillatory (HFO) regime
  - Coherent gamma-band regime
  - Intermediate low-synchrony transition regime

rather than producing a continuous modulation of oscillation frequency.
