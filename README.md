# 🐝 Pollinator Computer Vision Toolkit  

*A repository for building and validating CV models to quantify and classify pollinators/floral visitors.*  

---

## 📌 Overview  
This repository hosts **computer vision models** trained to detect and classify pollinators of Colombia using verified imagery from [iNaturalist](https://www.inaturalist.org/). It includes:  
- **Data pipelines** for image extraction/processing.  
- **Model experiments** (YOLO, Faster R-CNN).  
- **A Dash app** for bounding-box validation.  

---

## 🗂 Repository Structure  

### 📂 `clean_data/` *(Experimental)*  
- **Purpose**: Data loading, model prototyping, and fine-tuning.  
- **Format**: Jupyter Notebooks (Google Drive-linked).  
- **Content**:  
  - 🧪 Test scripts for YOLO/Faster R-CNN.  
  - 🔄 Exploratory code (no rigid structure).  

### 📂 `app_dash/` *(In Development)*  
- **Purpose**: Dash-based UI for validating auto-generated bounding boxes.  
- **Goal**: Accelerate image curation for model improvement.  

### 🔒 Hidden Directories  
- `database/`: MySQL integration for image metadata.  
- `keys/`: Secure credentials (Google Drive API, etc.).  

---

## 🛠 Key Features  
✅ **iNaturalist-Powered Dataset**  
✅ **Flexible Model Testing** (YOLO & Faster R-CNN)  
✅ **Validation Tool** (Dash app for QA workflows)  

---

## 🚧 Future Work  
- [ ] Deploy Dash app for team collaboration.  
- [ ] Standardize training pipelines.  
- [ ] Expand MySQL scalability.  
