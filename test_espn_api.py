#!/usr/bin/env python3
"""Test ESPN API endpoints to understand response structure."""
import requests
import json
from datetime import datetime

def test_scoreboard(sport, league):
    """Test scoreboard endpoint."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    print(f"\n{'='*80}")
    print(f"Testing: {url}")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Print top-level keys
            print(f"\nTop-level keys: {list(data.keys())}")
            
            # Check for events
            events = data.get("events", [])
            print(f"\nNumber of events: {len(events)}")
            
            if events:
                # Print first event structure
                print(f"\n{'='*80}")
                print("First Event Structure:")
                print(f"{'='*80}")
                event = events[0]
                print(json.dumps(event, indent=2)[:4000])
                
                # Extract key info
                print(f"\n{'='*80}")
                print("Key Event Fields:")
                print(f"{'='*80}")
                print(f"ID: {event.get('id')}")
                print(f"Name: {event.get('name')}")
                print(f"Date: {event.get('date')}")
                
                # Check competitions
                competitions = event.get("competitions", [])
                print(f"\nCompetitions: {len(competitions)}")
                
                if competitions:
                    comp = competitions[0]
                    print(f"\nCompetition keys: {list(comp.keys())}")
                    
                    # Check competitors
                    competitors = comp.get("competitors", [])
                    print(f"\nCompetitors: {len(competitors)}")
                    
                    for i, competitor in enumerate(competitors):
                        print(f"\n  Competitor {i+1}:")
                        team = competitor.get("team", {})
                        print(f"    Team Name: {team.get('displayName')}")
                        print(f"    Team Abbrev: {team.get('abbreviation')}")
                        print(f"    Score: {competitor.get('score')}")
                        print(f"    Home/Away: {competitor.get('homeAway')}")
                    
                    # Check status
                    status = comp.get("status", {})
                    print(f"\n  Status:")
                    print(f"    Type: {status.get('type', {}).get('description')}")
                    print(f"    Period: {status.get('period')}")
                    print(f"    Clock: {status.get('displayClock')}")
            else:
                print("\nNo events found in response")
                print(f"\nFull response structure:")
                print(json.dumps(data, indent=2)[:2000])
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test NBA
    test_scoreboard("basketball", "nba")
    
    # Test NFL
    test_scoreboard("football", "nfl")
    
    # Test NHL
    test_scoreboard("hockey", "nhl")
    
    # Test with date filter
    today = datetime.now().strftime("%Y%m%d")
    print(f"\n\n{'='*80}")
    print(f"Testing with date filter: {today}")
    print(f"{'='*80}")
    test_scoreboard("basketball", "nba")

