# 🚗 ParkFormer: A Transformer-Based Parking Policy with Goal Embedding and Pedestrian-Aware Control

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**ParkFormer** is an end-to-end Transformer-based framework designed for **autonomous parking in dynamic environments**. By integrating goal point attention, pedestrian trajectory prediction, and BEV-based perception, ParkFormer enables safe and accurate parking in both vertical and parallel layouts.

🔗 **Project Page**: [https://github.com/little-snail-f/ParkFormer](https://github.com/little-snail-f/ParkFormer)  
📄 **Paper**: Available upon request  
📦 **Dataset & Code**: Coming soon!

---

## 🚀 Key Features

- ✅ End-to-end parking control using **Transformer architecture**
- ✅ Inputs: multi-view RGB, pedestrian trajectories, ego-motion, and goal embedding
- ✅ Outputs: discrete control tokens (steering, acceleration, braking, gear)
- ✅ Dynamic environment handling with pedestrian-aware decision making
- ✅ Cross-attention fusion of BEV features and goal slot information
- ✅ Extensive evaluation in **CARLA 0.9.14** under both perpendicular and parallel scenarios

---

## 🧠 Method Overview

ParkFormer consists of the following modules:

1. **Image & Goal Encoder**  
   - Converts multi-view camera inputs to BEV features
   - Embeds the target parking slot with a goal-aware attention mechanism

2. **Ego-Pedestrian Encoder**  
   - Uses GRU to predict pedestrian trajectories
   - Fuses ego-motion and pedestrian dynamics via cross-attention

3. **Feature Fusion**  
   - Merges spatial and dynamic features using Transformer encoders

4. **Control Decoder**  
   - Autoregressively generates discrete control tokens for parking maneuvers

<p align="center">
  <img src="figures/system_architecture.png" alt="System Architecture" width="600"/>
</p>

---

## 🧪 Experiments

We evaluate ParkFormer on the CARLA 0.9.14 simulator with two layouts:

- 🅿️ Town04-Opt: Vertical parking  
- 🅿️ Town10HD-Opt: Parallel parking  

**Metrics:**  
- **Success Rate**: 96.57%  
- **Positional Error**: 0.21 m  
- **Orientation Error**: 0.41°  
- **Collision Rate**: 1.16%  

<table>
  <thead>
    <tr><th>Method</th><th>SR (%)</th><th>PE (m)</th><th>OE (°)</th><th>CR (%)</th></tr>
  </thead>
  <tbody>
    <tr><td>ParkFormer</td><td>96.57</td><td>0.21</td><td>0.41</td><td>1.16</td></tr>
    <tr><td>E2E Parking [25]</td><td>91.41</td><td>0.30</td><td>0.87</td><td>2.08</td></tr>
  </tbody>
</table>

---

## 📦 Dataset

We provide a multimodal dataset for end-to-end parking, containing:

- ~46,400 frames over 272 structured parking episodes  
- Synchronized RGB images, depth maps, BEV maps  
- Pedestrian motion data and expert control signals  
- Scenes with dynamic pedestrians in diverse layouts

---

## 📂 Directory Structure (Coming Soon)

