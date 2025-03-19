# 🎵 Hit or Flop

**HitOrFlop** is a machine learning project that predicts whether a song will be a *hit* or *flop* based on its musical attributes. The model is trained using **decades of Spotify song data** and leverages **Random Forest Classification** for predictions.  

## Features  
- 📊 **Trains on Historical Song Data** (60s to 2010s)  
- 🎶 **Extracts Features** like danceability, loudness, valence, etc.  
- 🌲 **Random Forest Classifier** for robust predictions  
- 🔄 **Automated Model Training & Saving** with joblib  
- ⚡ **Scalable Preprocessing** using `StandardScaler`  


## 📂 How to Run  
1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/HitOrFlop.git
   cd HitOrFlop
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Place dataset CSV files in the project folder.  
4. Run the script:  
   ```bash
   python main.py
   ```  
5. Trained models and scalers will be saved as `.pkl` files.  

---

Let me know if you'd like any changes! 🚀
