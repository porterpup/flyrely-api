# FlyRely - Quick Start Guide

Everything is ready! Here's what you need to do:

---

## Step 1: Get Your Weather API Key (2 minutes)

The Tomorrow.io signup page is open in your browser.

1. **Sign up** using Google, GitHub, or email
2. After logging in, go to: **https://app.tomorrow.io/development/keys**
3. **Copy your API key** (it looks like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`)

---

## Step 2: Deploy the API (5 minutes)

### Option A: Railway (Easiest)

1. Go to **https://railway.app** and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Create a new repo and upload the `flyrely-api` folder, OR:
   - Extract `flyrely-api-deploy.tar.gz` locally
   - Push to a new GitHub repo
4. Railway will auto-detect and deploy
5. Add environment variable: `WEATHER_API_KEY=your_key_here`
6. Get your URL (e.g., `flyrely-api-production.up.railway.app`)

### Option B: Run Locally

```bash
# Extract the archive
tar -xzf flyrely-api-deploy.tar.gz
cd flyrely-api

# Install dependencies
pip install -r requirements.txt

# Set your API key
export WEATHER_API_KEY=your_key_here

# Run the server
uvicorn main:app --reload
```

Server runs at: **http://localhost:8000**

---

## Step 3: Test with the UI (1 minute)

1. Open **`test-page.html`** in your browser
2. Enter your API URL (localhost:8000 or your Railway URL)
3. Select origin/destination airports
4. Click **"Check Flight Risk"**

---

## What's Included

| File | Description |
|------|-------------|
| `main.py` | FastAPI prediction API |
| `test-page.html` | Beautiful test UI |
| `models/` | Trained ML model (145MB) |
| `DEPLOY.md` | Detailed deployment guide |
| `WEATHER_API_SETUP.md` | Weather API setup guide |
| `Dockerfile` | Docker deployment |
| `railway.json` | Railway config |

---

## API Usage

### Predict Endpoint

```bash
curl -X POST https://your-api.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "JFK",
    "destination": "LAX",
    "departure_time": "2025-03-15T14:30:00",
    "airline": "AA"
  }'
```

### Response

```json
{
  "risk_level": "medium",
  "delay_probability": 0.27,
  "origin_weather": {
    "temperature_f": 45,
    "wind_speed_mph": 15,
    "visibility_miles": 8
  },
  "risk_factors": [
    "Evening departure (peak congestion time)"
  ],
  "recommendations": [
    "Monitor flight status closer to departure"
  ]
}
```

---

## Need Help?

- **Deployment issues**: See `DEPLOY.md`
- **Weather API**: See `WEATHER_API_SETUP.md`
- **API reference**: Visit `http://your-api/docs` (auto-generated Swagger UI)

🚀 **You're ready to predict flight delays!**
