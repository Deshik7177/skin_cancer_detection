# skin_cancer_detection
A simple skin cancer detection app using a trained model and FastAPI backend.

---

## ▶️ Run locally

### Run on Windows
#Git clone
```bash
git clone https://github.com/Deshik7177/skin_cancer_detection.git
```

1. **Create virtual environment**

```powershell
python -m venv venv
venv\Scripts\activate
```

2. **Upgrade pip**

```bash
pip install --upgrade pip
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Start the server**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

5. **Open in browser**

http://localhost:8000

---

### Run on Ubuntu / Linux
1. **Install Python (if needed)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv -y
```

2. **Create virtual environment**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. **Upgrade pip**

```bash
pip install --upgrade pip
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Start the server**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

6. **Open in browser**

http://localhost:8000

---

**Notes**

- If `backend/best.pt` or other large artifacts were previously committed, they have been removed from tracking and added to `.gitignore`.
- For development, use `uvicorn --reload` to enable auto-reload when editing the code.
