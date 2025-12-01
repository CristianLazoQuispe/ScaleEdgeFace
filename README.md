# ⚡ ScaleEdgeFace: Real-Time Face Recognition Framework

**Cristian Lazo-Quispe**, Ricardo R. Rodríguez-Bustinza, Renato Castro-Cruz  
*National University of Engineering · Pontifical Catholic University of Peru*  
📄 *Published at [ISCMI 2025 – IEEE International Conference on Soft Computing & Machine Intelligence](https://www.iscmi.us/)*  

---

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github&style=flat-square)](https://github.com/CristianLazoQuispe/ScaleEdgeFace)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&style=flat-square)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

</div>

---

## 🚀 Overview

**ScaleEdgeFace** is a **modular, parallel, and hardware-agnostic framework** for real-time face recognition.  
It unifies **FaceBoxes**, **MediaPipe**, **YOLOv8**, and **Norfair tracking** under a lock-free multithreaded design, integrating **local NumPy** and **Pinecone vector databases** for scalable retrieval.

> 🧩 Achieves **412 FPS on RTX 2070 Super** and **90 FPS on Jetson Nano**  
> 🎯 Accuracy: **94% on LFW**, **92% on VGGFace2**

---

## ⚙️ Quick Setup

### 🧪 Using Virtual Environment

```bash
source env_3.10_com/bin/activate
python analisis_recognition.py -c configs/webcam.conf
sudo python3 analisis_recognition.py -c configs/recognition_jetson_b01.conf
````

### 🐳 Using Docker

```bash
docker compose up --build
```

---

## 🧠 Jetson Optimization

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Fan Control**

```bash
# ON
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
# OFF
sudo sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'
```

---

## 📊 Performance Summary

| Config              | Device      |   FPS (mean/max)  |  Accuracy |
| :------------------ | :---------- | :---------------: | :-------: |
| MediaPipe + Norfair | RTX 2070    | **350.7 / 412.4** | 94% / 92% |
| MediaPipe + Norfair | Jetson Nano |  **57.7 / 90.2**  | 94% / 92% |

> Vector retrieval via Pinecone adds 15–30 ms latency, hidden by asynchronous queues.

---

## 🧩 Citation

```bibtex
@inproceedings{LazoQuispe2025ScaleEdgeFace,
  title={Real-Time Face Tracking and Vector Databases for Scalable Face Recognition},
  author={Lazo-Quispe, Cristian and Rodríguez-Bustinza, Ricardo R. and Castro-Cruz, Renato},
  booktitle={International Conference on Soft Computing & Machine Intelligence (ISCMI)},
  year={2025},
  address={Rio de Janeiro, Brazil}
}
```

---

## 🪶 License

Released under the **MIT License**. See [LICENSE](./LICENSE) for details.

