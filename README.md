# ⚽ Deep Learning & AI in Sport: From Tracking Data to Playing Styles  

**Authors:** Pegah Rahimian, David Sumpter  

---

## 📘 Course Overview  

This course introduces participants to modern **football analytics** workflows using **event + tracking data**.  
We focus on **practical data science applications** in sport, bridging the gap between machine learning techniques and football-specific insights.  

By the end of the course, participants will:  
- Understand how to **synchronize tracking and event data**.  
- Detect and quantify **defensive structures** (lines, compactness, depth).  
- Extract **playing styles** from possession sequences.  
- Build **role representations** of players for scouting and recruitment.  
- Use **dimensionality reduction and clustering** to uncover tactical patterns.  

The course combines **lectures, coding tutorials, and hands-on exercises** with **Premier League** and **Real Madrid** datasets.  

---

## 📊 Data Used  

We use **multi-modal football data** provided in three formats per match:  
- `dynamic/` → Event-level data (Parquet)  
- `tracking/` → Player and ball tracking (JSON)  
- `meta/` → Metadata including players, roles, and teams (JSON)  

Example data directories:  

RealMadrid_data/
│
├── dynamic/ # Event-level parquet files
├── tracking/ # Tracking JSON files
└── meta/ # Metadata JSON files

PremierLeague_data/2024/
│
├── dynamic/ # Event-level parquet files
├── tracking/ # Tracking JSON files
└── meta/ # Metadata JSON files

yaml
Copy code

> ⚠️ Data is anonymized and adapted for educational use only.  

---

## 🏟️ Course Modules  

### **Session 1. Defensive Structures & Lines**
- Synchronization of event & tracking data  
- Coordinate transformations & attack direction inference  
- Detecting defensive lines (k-means clustering of defenders)  
- Compactness & line depth analysis  
- Season-wide defensive shape trends  
- **Exercise Sheet:** Analyze Real Madrid’s defensive structures  

---

### **Session 2. Playing Style & Role Representations**
- Constructing **possession sequences** from event data  
- Feature engineering (progress, duration, pass counts)  
- Playing style embeddings (direct play, tiki-taka, high press)  
- Team style fingerprints (possession distributions)  
- Player-level embeddings and similarity search  
- Role archetype clustering (e.g., destroyers, playmakers, strikers)  
- **Exercise Sheet:** Playing style clustering + role similarity (Premier League)  

---

### **Session 3. Advanced Tactical Analysis (Optional)**
- Pressing intensity and defensive transition analysis  
- Ball progression networks and flow motifs  
- Expected Threat (xT) in possession and defensive phases  
- Neural embeddings for tactical actions  
- **Capstone Exercise:** Build a scouting dashboard combining style, role, and defensive metrics  

---

## 📂 Repository Structure  

notebooks/
│ ├── session1_defensive_structures.ipynb
│ ├── session2_playing_styles_roles.ipynb
│ └── session3_advanced_tactics.ipynb
data/
│ ├── RealMadrid_data/
│ └── PremierLeague_data/2024/
README.md

yaml
Copy code

---

## 🧑‍🏫 Target Audience  

- **Football analysts** looking to adopt machine learning workflows  
- **Data scientists** interested in applying AI to sport  
- **Students & researchers** in sports analytics and applied statistics  
- **Coaches & scouts** seeking data-driven tactical insights  

---

## 📚 Prerequisites  

- Strong knowledge of **Python** and **data science libraries** (`pandas`, `numpy`, `scikit-learn`)  
- Basic familiarity with **football tactics** (formations, roles, phases of play)  
- Some exposure to **machine learning concepts** (clustering, PCA, embeddings)  

---

## ✨ Learning Outcomes  

After completing the course, participants will be able to:  
1. Process and synchronize event + tracking data.  
2. Detect and quantify defensive line structures.  
3. Cluster possessions into tactical playing styles.  
4. Represent players via embeddings and compare roles.  
5. Apply AI/ML methods to derive actionable football insights.  

---

## 👥 Authors  

- **Pegah Rahimian** — Postdoctoral Researcher in Football Analytics, Uppsala University  
- **David Sumpter** — Professor of Applied Mathematics, Uppsala University; Author of *Soccermatics*  

---

## 📜 License  

This repository is for **educational and research purposes only**.  
Commercial use of the data or materials is not permitted.  

---

⚽ *“Data is only useful when it helps us understand football.”*  