#!/usr/bin/env python3
"""
Process Room Arranger Data
Reads paste-room-here.txt and updates room_data.csv and room_data.json
"""

import csv
import json
import os
from datetime import datetime

INPUT_FILE = 'paste-room-here.txt'
CSV_FILE = 'room_data.csv'
JSON_FILE = 'room_data.json'

def parse_object_line(line):
    """Parse a single object line from Room Arranger format"""
    if not line.strip():
        return None
    
    try:
        parts = line.split(';')
        if len(parts) != 3:
            return None
        
        object_name = parts[0].strip()
        dimensions = parts[1].strip()
        position = parts[2].strip()
        
        # Parse position (x, y, z)
        pos_parts = position.split(',')
        x = float(pos_parts[0].strip())
        y = float(pos_parts[1].strip()) 
        z = float(pos_parts[2].strip()) if len(pos_parts) > 2 else 0
        
        return {
            'object': object_name,
            'dimensions': dimensions,
            'x': x,
            'y': y,
            'z': z
        }
    except:
        return None

def process_room_data():
    """Process room data from paste-room-here.txt"""
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found!")
        print("Please create the file and paste your Room Arranger data into it.")
        return False
    
    # Read input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        print(f"[ERROR] {INPUT_FILE} is empty!")
        return False
    
    # Parse objects
    lines = content.split('\n')
    objects = []
    for line in lines:
        obj = parse_object_line(line)
        if obj:
            objects.append(obj)
    
    if not objects:
        print("[ERROR] No valid objects found in the file!")
        return False
    
    # Generate room ID and timestamp
    timestamp = datetime.now()
    room_id = timestamp.strftime("room_%Y%m%d_%H%M%S")
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add room info to each object
    for obj in objects:
        obj['room_id'] = room_id
        obj['timestamp'] = timestamp_str
    
    # Update CSV file
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['room_id', 'timestamp', 'object', 'dimensions', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for obj in objects:
            writer.writerow(obj)
    
    # Update JSON file
    json_data = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except:
            json_data = []
    
    # Create room entry
    room_entry = {
        'room_id': room_id,
        'timestamp': timestamp_str,
        'object_count': len(objects),
        'objects': [
            {
                'object': obj['object'],
                'dimensions': obj['dimensions'],
                'x': obj['x'],
                'y': obj['y'],
                'z': obj['z']
            }
            for obj in objects
        ]
    }
    json_data.append(room_entry)
    
    # Save updated JSON
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    # Clear the input file for next use
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('')
    
    print(f"[SUCCESS] Processed {len(objects)} objects")
    print(f"  Room ID: {room_id}")
    print(f"  CSV: {CSV_FILE}")
    print(f"  JSON: {JSON_FILE}")
    print(f"\n[INFO] {INPUT_FILE} has been cleared for next use")
    
    # Show summary
    print("\nObjects processed:")
    for obj in objects:
        print(f"  - {obj['object']}: {obj['dimensions']} at ({obj['x']}, {obj['y']}, {obj['z']})")
    
    return True

def main():
    print("Room Arranger Data Processor")
    print("="*30)
    print(f"Reading from: {INPUT_FILE}")
    print()
    
    if process_room_data():
        print("\n[TIP] Paste new room data into paste-room-here.txt and run again!")
    
if __name__ == "__main__":
    main()