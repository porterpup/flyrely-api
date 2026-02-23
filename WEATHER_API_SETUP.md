# Weather API Setup Guide

FlyRely uses real-time weather data to improve delay predictions. Choose one provider:

---

## Option 1: Tomorrow.io (Recommended)

**Best for:** Aviation-specific weather data, detailed conditions
**Free tier:** 500 API calls/day (plenty for testing)

### Setup Steps:

1. **Go to:** https://www.tomorrow.io/weather-api/

2. **Click "Get Started Free"**

3. **Create account** (email or Google/GitHub)

4. **After signup, go to:** https://app.tomorrow.io/development/keys

5. **Copy your API key** (looks like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`)

6. **Add to your `.env` file:**
   ```
   WEATHER_API_PROVIDER=tomorrow
   WEATHER_API_KEY=aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
   ```

### Tomorrow.io Features:
- Temperature, wind, visibility
- Weather codes (fog, rain, snow, thunderstorm)
- Forecast up to 14 days
- Aviation-specific metrics (ceiling, dew point)

---

## Option 2: OpenWeatherMap

**Best for:** Simple weather data, higher free quota
**Free tier:** 1,000 API calls/day

### Setup Steps:

1. **Go to:** https://openweathermap.org/api

2. **Click "Sign Up"** (top right)

3. **Create account and verify email**

4. **Go to:** https://home.openweathermap.org/api_keys

5. **Copy your API key** (looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`)

6. **Add to your `.env` file:**
   ```
   WEATHER_API_PROVIDER=openweathermap
   WEATHER_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
   ```

### OpenWeatherMap Features:
- Temperature, wind, visibility
- Weather descriptions
- Forecast up to 5 days (free tier)

---

## Testing Your API Key

### Quick test with curl:

**Tomorrow.io:**
```bash
curl "https://api.tomorrow.io/v4/weather/realtime?location=40.7128,-74.0060&apikey=YOUR_KEY&units=imperial"
```

**OpenWeatherMap:**
```bash
curl "https://api.openweathermap.org/data/2.5/weather?lat=40.7128&lon=-74.0060&appid=YOUR_KEY&units=imperial"
```

If you get a JSON response with weather data, it's working!

---

## Running Without Weather API

The FlyRely API works fine without a weather API key:
- Predictions will use default weather values
- Accuracy will be slightly lower (~5% less accurate)
- All other features work normally

This is fine for development and testing.

---

## API Key Security

**DO NOT:**
- Commit your API key to GitHub
- Share your key publicly
- Put the key in frontend JavaScript

**DO:**
- Use environment variables
- Add `.env` to `.gitignore`
- Use platform secrets (Railway, Render, etc.)

---

## Rate Limits

| Provider | Free Tier | Rate Limit |
|----------|-----------|------------|
| Tomorrow.io | 500/day | 3/second |
| OpenWeatherMap | 1,000/day | 60/minute |

FlyRely caches weather data for 30 minutes, so you won't hit limits with normal usage.
