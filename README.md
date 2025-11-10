# Software

# 🪐 Image Enhancement for Mars Rover Imagery



> Deep learning–based **image enhancement and deblurring** for Mars rover images using an **Attention U-Net** architecture.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌍 Overview

This project enhances blurry or degraded Mars surface images captured by rover missions.  
It uses an **Attention U-Net** to restore sharpness, improve detail, and preserve the natural Martian colors.

---

## ⚙️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- Pillow, NumPy, tqdm, matplotlib  
- CUDA GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt


# Hardware and Electronics Design

This directory contains the hardware design files, schematics, and architectural diagrams for our Autonomous Health Monitoring System.

Our design philosophy is based on adapting the high-performance JPL FPGA Payload for a new purpose. We treat this powerful, 10-watt FPGA payload as the "high-risk subsystem" that our autonomous AI system must monitor and protect.

Our solution is a "watchdog" system that runs on a separate, low-power microcontroller, ensuring it can function even if the main FPGA payload fails.

---

## 1. System Architecture: Modified JPL Payload

This block diagram illustrates our core system architecture.

The base architecture consists of:
* *FPGA (Virtex-5):* The high-performance, high-risk processing unit.
* *Microcontroller (8051-series or similar):* The low-power, "always-on" control processor.
* *SRAM & PROM:* Memory units for the FPGA and Microcontroller.
* *Oscillators:* System clocks for the main components.

Our design introduces a dedicated AI-driven health monitoring loop (shown in red):
* *Health Sensors (Temp, Current):* We have added a sensor block to draw real-time thermal and power telemetry directly from the high-power FPGA.
* *AI Model Host:* This sensor data is fed to the Microcontroller, not the FPGA. The Microcontroller, which is continuously powered, runs our lightweight AI anomaly detection model.
* *Autonomous Correction:* The AI model analyzes the sensor data for pre-failure signatures (e.g., thermal runaway, abnormal current draw). Because the Microcontroller already has power control over the other components, it can execute an autonomous hardware correction—such as a forced power-cycle or an emergency shutdown of the FPGA—to prevent permanent failure.

## 2. FPGA Internal Science Data Flow

This diagram shows the internal data flow for the MSPI image processing algorithm. This is the primary science task of the FPGA payload that our system is monitoring.

It is crucial to understand that this data path is separate from our AI health model.
* *Science Path:* Camera → FPGA Buffers → FPGA Matrix Multiplier → Data Recorder.
* *Our Health Path:* FPGA Health Sensors → Microcontroller (AI Model) → FPGA Power Control.

The "Software/Model" block is part of the image processing algorithm (it's used to recalculate matrices) and is not our separate AI health monitoring model. This separation ensures our health system is non-intrusive.

## 3. High-Risk Component: Virtex-5 Processor

We include this diagram to justify our choice of subsystem. This diagram details the embedded PowerPC 440 processor within the Virtex-5 FPGA.

This powerful, complex processor, with its *32KB caches*, *hard 5X2 crossbar switch*, and significant *10-watt power consumption*, is precisely what makes the FPGA payload both highly capable and a mission-critical failure risk. Its thermal and power characteristics are the primary targets for our autonomous monitoring system.

## 4. System Integration & Physical Layout

This Computer-Aided Drawing (CAD) model illustrates the physical integration of all subsystems. It serves as our verification for Space Environment Readiness, demonstrating how the components fit within the standard CubeSat volume constraints.

This 1U CubeSat structure clearly shows the board stack and component layout:
* *JPL FPGA:* The high-risk payload our system is designed to monitor.
* *C&DH BOARD (Command & Data Handling):* The board that hosts the low-power microcontroller running our AI "watchdog" model.
* *EPS BOARD (Electrical Power System):* The power system that our AI helps stabilize by managing the FPGA's health and power draw.
* *PAYLOAD & COMM BOARD:* The other primary mission subsystems.

This model validates our design's feasibility, proving that our monitoring system (the sensors and AI logic on the C&DH board) is integrated with the payload (JPL FPGA) in a compact, standard 1U form factor that easily fits within the 3U challenge requirement.
