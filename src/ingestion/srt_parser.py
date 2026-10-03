import os
import re
from typing import List, Dict

def parse_srt(file_path: str, movie_title: str) -> List[Dict]:
    """
    Parses an .srt file in the most robust and simplest way.
    
    Args:
        file_path: Path to the .srt file.
        movie_title: The name of the movie for metadata tagging.
        
    Returns:
        A list of dictionaries with 'text', 'start_time', 'end_time', and 'movie_title'.
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read().replace("\r\n", "\n")

    # Split the file by blank lines, which separate subtitle blocks
    blocks = content.strip().split("\n\n")
    
    subtitles = []
    
    for block in blocks:
        lines = block.split("\n")
        
        # A valid SRT block usually has at least 3 lines: Index, Timecode, Text
        if len(lines) >= 3:
            # Find the timecode line which contains '-->'
            timecode_line = next((line for line in lines if '-->' in line), None)
            
            if not timecode_line:
                continue
                
            time_parts = timecode_line.split(" --> ")
            if len(time_parts) == 2:
                start_time = time_parts[0].strip()
                end_time = time_parts[1].strip()
                
                # The text is everything after the timecode
                text_index = lines.index(timecode_line) + 1
                dialogue = " ".join(lines[text_index:])
                
                # Remove any HTML styling like <i> or <font>
                clean_dialogue = re.sub(r'<[^>]+>', '', dialogue).strip()
                
                if clean_dialogue:
                    subtitles.append({
                        "text": clean_dialogue,
                        "metadata": {
                            "movie_title": movie_title,
                            "start_time": start_time,
                            "end_time": end_time
                        }
                    })
                    
    return subtitles

if __name__ == "__main__":
    # Test script for learning purposes
    print("Testing parser...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_file = os.path.join(base_dir, "data", "subtitles", "The_Matrix_1999.srt")
    try:
        results = parse_srt(sample_file, "The Matrix")
        for r in results:
            print(r)
        print(f"Parsed {len(results)} subtitle entries successfully.")
    except Exception as e:
        print(f"Error testing parser: {e}")
