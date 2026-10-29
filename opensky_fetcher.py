#!/usr/bin/env python3
"""
OpenSky Flight Data Fetcher
Fetches real flight data for UA, AA, DL domestic flights
"""

import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

OPENSKY_USERNAME = os.getenv('OPENSKY_USERNAME')
OPENSKY_PASSWORD = os.getenv('OPENSKY_PASSWORD')

# OpenSky API endpoints
OPENSKY_API_URL = "https://opensky-network.org/api"

# Airlines of interest
AIRLINES = {
    'UAL': 'United',      # UA
    'AAL': 'American',    # AA
    'DAL': 'Delta'        # DL
}

# Domestic US airport pairs (sample major routes)
DOMESTIC_ROUTES = [
    ('KJFK', 'KLAX'), ('KJFK', 'KORD'), ('KJFK', 'KDFW'),
    ('KLAX', 'KJFK'), ('KLAX', 'KORD'), ('KLAX', 'KDFW'),
    ('KORD', 'KJFK'), ('KORD', 'KLAX'), ('KORD', 'KDFW'),
    ('KDFW', 'KJFK'), ('KDFW', 'KLAX'), ('KDFW', 'KORD'),
    ('KATL', 'KJFK'), ('KATL', 'KLAX'), ('KATL', 'KORD'),
    ('KDEN', 'KJFK'), ('KDEN', 'KLAX'), ('KDEN', 'KORD'),
]

def fetch_flight_data(days_back=28):
    """
    Fetch flight data from OpenSky for past N days
    
    Args:
        days_back: Number of days of history to fetch (default 28 = 4 weeks)
    
    Returns:
        List of flight records with delays
    """
    
    flights = []
    auth = (OPENSKY_USERNAME, OPENSKY_PASSWORD)
    
    print(f"Fetching flight data from OpenSky (past {days_back} days)...")
    print(f"Airlines: {', '.join(AIRLINES.values())}")
    print(f"Routes: {len(DOMESTIC_ROUTES)} major domestic routes")
    print()
    
    # Fetch data for each day
    for day_offset in range(days_back):
        date = datetime.utcnow() - timedelta(days=day_offset)
        start_time = int(date.replace(hour=0, minute=0, second=0).timestamp())
        end_time = int(date.replace(hour=23, minute=59, second=59).timestamp())
        
        print(f"Fetching {date.date()}...", end=" ")
        
        try:
            # Query arrivals for each route
            for origin, dest in DOMESTIC_ROUTES:
                try:
                    response = requests.get(
                        f"{OPENSKY_API_URL}/flights/arrival",
                        params={
                            'airport': dest,
                            'begin': start_time,
                            'end': end_time
                        },
                        auth=auth,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        arrivals = response.json()
                        
                        for flight in arrivals:
                            # Filter for our airlines
                            callsign = flight.get('callsign', '').strip()
                            airline_code = callsign[:3] if callsign else None
                            
                            if airline_code in AIRLINES:
                                # Calculate delay (actual - scheduled)
                                scheduled = flight.get('firstSeen')
                                actual = flight.get('lastSeen')
                                
                                if scheduled and actual:
                                    delay_minutes = (actual - scheduled) / 60
                                    
                                    flights.append({
                                        'date': date.date().isoformat(),
                                        'flight_id': callsign,
                                        'airline': AIRLINES[airline_code],
                                        'origin': flight.get('estDepartureAirport'),
                                        'destination': flight.get('estArrivalAirport'),
                                        'scheduled_time': scheduled,
                                        'actual_time': actual,
                                        'delay_minutes': delay_minutes,
                                        'is_delayed': 1 if delay_minutes > 15 else 0
                                    })
                
                except Exception as e:
                    pass  # Skip failed route queries
            
            print("✓")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    print()
    print(f"Total flights fetched: {len(flights)}")
    return flights


def save_flights_to_csv(flights, filename='opensky_flights.csv'):
    """Save flight data to CSV for training"""
    if not flights:
        print("No flights to save.")
        return False
    
    df = pd.DataFrame(flights)
    df.to_csv(filename, index=False)
    print(f"Saved {len(flights)} flights to {filename}")
    return True


def main():
    if not OPENSKY_USERNAME or not OPENSKY_PASSWORD:
        print("ERROR: OpenSky credentials not found in .env file")
        print("Set OPENSKY_USERNAME and OPENSKY_PASSWORD in .env")
        return False
    
    # Fetch 4 weeks of data
    flights = fetch_flight_data(days_back=28)
    
    if flights:
        save_flights_to_csv(flights, 'data/opensky_flights_recent.csv')
        
        # Print summary stats
        df = pd.DataFrame(flights)
        delayed = (df['is_delayed'] == 1).sum()
        on_time = (df['is_delayed'] == 0).sum()
        
        print("\n=== Data Summary ===")
        print(f"On-time flights: {on_time} ({on_time/len(df)*100:.1f}%)")
        print(f"Delayed flights (>15min): {delayed} ({delayed/len(df)*100:.1f}%)")
        print(f"Average delay: {df['delay_minutes'].mean():.1f} minutes")
        
        return True
    else:
        print("ERROR: No flights fetched. Check credentials and API status.")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
