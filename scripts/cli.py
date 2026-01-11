#!/usr/bin/env python3
"""
Music Agent Pro - Command Line Interface
=========================================

Interactive terminal interface for the Music Agent.

Usage:
    python scripts/cli.py
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import Agent
from agent_tools.api_clients import get_llm


def print_header():
    """Print CLI header."""
    print()
    print("  Music Agent Pro - CLI")
    print()


def print_help():
    """Print help message."""
    print("""
Commands:
  help          Show this help
  quit/exit     Exit the CLI
  
Analysis:
  analyze       Library statistics
  track <id>    Track information
  report        Generate library report
  
Discovery:
  similar <id>  Find similar tracks
  compare <a> <b>  Compare two tracks
  mood <mood>   Find by mood (happy, sad, calm, energetic)
  tempo <min>-<max>  Find by BPM range
  
Playlists:
  playlist <theme>  Generate playlist (workout, relax, study, party)
  
Online:
  trends        Trending music
  youtube <url> Download from YouTube
  
Examples:
  > analyze
  > similar 42
  > compare 10 20
  > playlist workout
  > mood happy
  > tempo 120-140
""")


def main():
    """Main CLI loop."""
    print_header()
    
    # Check database
    db_path = ROOT / "data" / "processed" / "music_library.db"
    if not db_path.exists():
        print("Database not found!")
        print("   Run: python scripts/setup_database.py")
        print()
        return
    
    # Initialize agent
    print("Initializing...")
    llm = get_llm()
    print(f"LLM: {llm.name if llm else 'Fallback mode'}")
    
    agent = Agent(llm)
    print("Ready!\n")
    print_help()
    
    # Main loop
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            if cmd in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if cmd in ["help", "?"]:
                print_help()
                continue
            
            # Process with agent
            response = agent.chat(user_input)
            print()
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
