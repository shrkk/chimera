import argparse
import csv
import io
import json
import logging
import sys
import urllib.request
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

def download_and_process_puzzles(args):
    """
    Downloads Lichess puzzle CSV (zstandard-compressed), streams it, 
    and stratifies puzzles across Elo bands.
    """
    total_count = args.count
    target_per_band = total_count // 3
    
    counts = {
        "800_1100": 0,
        "1100_1400": 0,
        "1400_1700": 0
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Streaming puzzles from {LICHESS_PUZZLE_URL}...")
    
    req = urllib.request.Request(LICHESS_PUZZLE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            csv_reader = csv.reader(text_stream)
            
            with open(output_path, 'w', encoding='utf-8') as outfile:
                pbar = tqdm(total=total_count, desc="Extracting Puzzles")
                
                for row in csv_reader:
                    # Expected columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
                    if not row or len(row) < 8:
                        continue
                        
                    puzzle_id = row[0]
                    if puzzle_id == "PuzzleId":
                        continue
                        
                    fen = row[1]
                    moves = row[2].split()
                    
                    try:
                        rating = int(row[3])
                    except ValueError:
                        continue
                    
                    themes = row[7].split()
                    
                    # Stratification check
                    band = None
                    if 800 <= rating < 1100:
                        band = "800_1100"
                    elif 1100 <= rating < 1400:
                        band = "1100_1400"
                    elif 1400 <= rating <= 1700:
                        band = "1400_1700"
                        
                    if band and counts[band] < target_per_band:
                        counts[band] += 1
                        
                        record = {
                            "puzzle_id": puzzle_id,
                            "fen": fen,
                            "moves": moves,
                            "rating": rating,
                            "themes": themes
                        }
                        outfile.write(json.dumps(record) + "\n")
                        pbar.update(1)
                        
                        if sum(counts.values()) >= total_count:
                            break
                            
                pbar.close()
                
    print(f"Finished extracting {sum(counts.values())} puzzles to {output_path}")
    print(f"Distribution: {counts}")

def main():
    parser = argparse.ArgumentParser(description="Download and stratify Lichess puzzles.")
    parser.add_argument("--count", type=int, default=1200, help="Total number of puzzles to extract")
    parser.add_argument("--output", type=str, default="datasets/puzzles.jsonl", help="Output JSONL file path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    download_and_process_puzzles(args)

if __name__ == "__main__":
    main()
