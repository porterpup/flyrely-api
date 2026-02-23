# FlyRely API

Flight delay prediction API with real-time weather integration.

## Features

- **ML-Powered Predictions**: Trained on 2M+ real BTS flight records (2024-2025)
- **Real-Time Weather**: Integrates with Tomorrow.io or OpenWeatherMap
- **Risk Classification**: Low/Medium/High delay risk levels
- **Risk Factors**: Explains why a flight might be delayed
- **Recommendations**: Actionable advice based on conditions

## Quick Start

### 1. Install Dependencies

```bash
cd flyrely-api
pip install -r requirements.txt
```

### 2. Configure Weather API (Optional but Recommended)

Get a free API key from:
- **Tomorrow.io**: https://www.tomorrow.io/weather-api/ (500 free calls/day)
- **OpenWeatherMap**: https://openweathermap.org/api (1,000 free calls/day)

```bash
cp .env.example .env
# Edit .env with your API key
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `GET /`
API information and available endpoints.

### `GET /health`
Health check - confirms model is loaded and weather API is configured.

### `GET /airports`
List all supported airports with their historical delay rates.

### `POST /predict`
Predict delay risk for a flight.

**Request Body:**
```json
{
  "origin": "JFK",
  "destination": "LAX",
  "departure_time": "2025-03-15T14:30:00",
  "airline": "AA"
}
```

**Response:**
```json
{
  "risk_level": "medium",
  "delay_probability": 0.32,
  "confidence": 0.68,
  "origin": "JFK",
  "destination": "LAX",
  "departure_time": "2025-03-15T14:30:00",
  "airline": "AA",
  "origin_weather": {
    "temperature_f": 45.2,
    "wind_speed_mph": 18.5,
    "visibility_miles": 8.0,
    "conditions": "partly cloudy"
  },
  "destination_weather": {
    "temperature_f": 72.1,
    "wind_speed_mph": 8.2,
    "visibility_miles": 10.0,
    "conditions": "clear"
  },
  "risk_factors": [
    "Origin airport has high historical delay rate (26%)",
    "Evening departure (peak congestion time)"
  ],
  "recommendations": [
    "Monitor flight status closer to departure",
    "Have a backup plan for tight connections"
  ]
}
```

## Risk Levels

| Level | Delay Probability | What It Means |
|-------|-------------------|---------------|
| **Low** | < 25% | Flight conditions are favorable |
| **Medium** | 25-50% | Some risk factors present |
| **High** | > 50% | Significant delay risk |

## Supported Airports

The API supports 30 major US airports:
ATL, ORD, DFW, DEN, LAX, JFK, SFO, SEA, MIA, PHX, LAS, MCO, CLT, EWR, BOS, MSP, DTW, PHL, LGA, DCA, IAH, SLC, SAN, TPA, PDX, BWI, FLL, MDW, BNA, AUS

## Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: 2M flights from BTS (2024-2025)
- **Features**: 25 features including time, airport, airline, and weather
- **AUC-ROC**: 0.70

## Example Usage

### Python
```python
import httpx

response = httpx.post("http://localhost:8000/predict", json={
    "origin": "JFK",
    "destination": "LAX",
    "departure_time": "2025-03-15T14:30:00",
    "airline": "AA"
})

result = response.json()
print(f"Risk: {result['risk_level']} ({result['delay_probability']*100:.0f}%)")
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"origin": "JFK", "destination": "LAX", "departure_time": "2025-03-15T14:30:00"}'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    origin: 'JFK',
    destination: 'LAX',
    departure_time: '2025-03-15T14:30:00',
    airline: 'AA'
  })
});

const result = await response.json();
console.log(`Risk: ${result.risk_level}`);
```

## Production Deployment

### Using Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WEATHER_API_KEY` | API key for weather service | (none) |
| `WEATHER_API_PROVIDER` | `tomorrow` or `openweathermap` | `tomorrow` |

## License

MIT
