# FlyRely API Deployment Guide

Quick deployment guides for popular platforms. Choose one:

---

## Option 1: Railway (Recommended - Easiest)

**Time: ~5 minutes | Free tier: $5/month credit**

### Steps:

1. **Create a GitHub repo** and push the `flyrely-api` folder:
   ```bash
   cd flyrely-api
   git init
   git add .
   git commit -m "Initial commit"
   gh repo create flyrely-api --public --push
   ```

2. **Go to [Railway.app](https://railway.app)** and sign in with GitHub

3. **Click "New Project" → "Deploy from GitHub repo"**

4. **Select your `flyrely-api` repo**

5. **Add environment variables** (Settings → Variables):
   ```
   WEATHER_API_KEY=your_tomorrow_io_key
   WEATHER_API_PROVIDER=tomorrow
   ```

6. **Railway auto-detects Python and deploys!**

7. **Get your URL** from the deployment (e.g., `flyrely-api-production.up.railway.app`)

---

## Option 2: Render

**Time: ~5 minutes | Free tier available**

### Steps:

1. **Push to GitHub** (same as above)

2. **Go to [Render.com](https://render.com)** and sign in

3. **New → Web Service → Connect your repo**

4. **Configure:**
   - Name: `flyrely-api`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Add environment variables:**
   ```
   WEATHER_API_KEY=your_key
   WEATHER_API_PROVIDER=tomorrow
   ```

6. **Deploy!**

---

## Option 3: Fly.io

**Time: ~10 minutes | Free tier: 3 shared VMs**

### Steps:

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and launch:**
   ```bash
   cd flyrely-api
   fly auth login
   fly launch
   ```

3. **Set secrets:**
   ```bash
   fly secrets set WEATHER_API_KEY=your_key
   fly secrets set WEATHER_API_PROVIDER=tomorrow
   ```

4. **Deploy:**
   ```bash
   fly deploy
   ```

---

## Option 4: Docker (Any Platform)

Works on AWS ECS, Google Cloud Run, Azure, DigitalOcean, etc.

### Build and run:

```bash
cd flyrely-api
docker build -t flyrely-api .
docker run -p 8000:8000 \
  -e WEATHER_API_KEY=your_key \
  -e WEATHER_API_PROVIDER=tomorrow \
  flyrely-api
```

### Push to registry:

```bash
docker tag flyrely-api your-registry/flyrely-api:latest
docker push your-registry/flyrely-api:latest
```

---

## After Deployment

### Test your API:

```bash
# Replace with your actual URL
API_URL="https://your-app.railway.app"

# Health check
curl $API_URL/health

# Test prediction
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"origin": "JFK", "destination": "LAX", "departure_time": "2025-03-15T14:30:00"}'
```

### Update the test page:

Edit `test-page.html` and replace `API_URL` with your deployed URL.

---

## Troubleshooting

### "Model not found" error
The model file (`flight_delay_model.joblib`) is 145MB. Some platforms have file size limits.

**Solutions:**
- Railway/Render: Should work fine (no strict limits)
- Fly.io: Increase VM memory if needed
- Alternative: Host model on S3/GCS and load at startup

### "Weather API not working"
1. Check your API key is set correctly
2. Verify the provider name (`tomorrow` or `openweathermap`)
3. Check API quota hasn't been exceeded

### Slow cold starts
The model takes ~2-3 seconds to load. First request after idle may be slow.

**Solutions:**
- Use a paid tier with always-on instances
- Add a health check ping to keep warm
