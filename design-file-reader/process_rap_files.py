#!/usr/bin/env python3
"""
Room Arranger RAP File Processor
Automates extraction of object listings from RAP files using Room Arranger
Uses: Project -> Export -> Copy Object Listing to Clipboard
"""

import os
import time
import subprocess
import pyperclip
import pyautogui
from datetime import datetime
import json
import csv
from pathlib import Path
import gc  # For garbage collection to prevent memory issues

# Configure pyautogui
pyautogui.PAUSE = 0.8  # Increased pause between actions to prevent crashes
pyautogui.FAILSAFE = False  # Disable fail-safe temporarily

# Configuration
ROOM_ARRANGER_PATH = r"C:\Program Files\Room Arranger\rooarr.exe"
RAP_FOLDER = "room-plans"
CSV_FILE = "room_data.csv"
JSON_FILE = "room_data.json"
PROCESSED_LOG = "processed_raps.log"

# Timing delays (increased to prevent crashes)
DELAYS = {
    'app_start': 10,       # Wait for Room Arranger to start (increased)
    'dialog_dismiss': 3,   # Wait after dismissing registration dialog (increased)
    'menu_open': 2,        # Wait for menu to open (increased)
    'after_copy': 4,       # Wait after copying to clipboard (increased)
    'before_close': 2,     # Wait before closing (increased)
    'between_actions': 1,  # General delay between actions
}

def check_dependencies():
    """Check if required libraries are installed"""
    required = []
    
    try:
        import pyautogui
    except ImportError:
        required.append('pyautogui')
    
    try:
        import pyperclip
    except ImportError:
        required.append('pyperclip')
    
    if required:
        print("[INFO] Installing required dependencies...")
        import subprocess
        for package in required:
            subprocess.check_call(['pip', 'install', package, '--quiet'])
        print("[OK] Dependencies installed. Please run the script again.")
        return False
    
    if not os.path.exists(ROOM_ARRANGER_PATH):
        print(f"[ERROR] Room Arranger not found at: {ROOM_ARRANGER_PATH}")
        return False
    
    return True

def get_processed_files():
    """Get list of already processed RAP files"""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_as_processed(filename):
    """Mark a RAP file as processed"""
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{filename}\n")

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

def process_clipboard_data(room_id):
    """Process object data from clipboard"""
    try:
        content = pyperclip.paste()
        
        # Debug: Show what's in clipboard
        if content:
            print(f"    Debug - Clipboard length: {len(content)} chars")
            print(f"    Debug - First 100 chars: {content[:100]}")
        
        # Check if clipboard contains registration dialog text
        if "Please register" in content or "shareware" in content:
            print(f"    Debug - Registration dialog text detected")
            return None
        
        if not content or len(content) < 10:
            print(f"    Debug - Clipboard empty or too short")
            return None
        
        # Parse objects
        lines = content.split('\n')
        objects = []
        for line in lines:
            obj = parse_object_line(line)
            if obj:
                objects.append(obj)
        
        if not objects:
            return None
        
        # Add room info
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        for obj in objects:
            obj['room_id'] = room_id
            obj['timestamp'] = timestamp_str
        
        return {
            'room_id': room_id,
            'timestamp': timestamp_str,
            'object_count': len(objects),
            'objects': objects
        }
    except Exception as e:
        print(f"  [ERROR] Failed to process clipboard: {e}")
        return None

def save_to_csv(room_data):
    """Save room data to CSV"""
    file_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['room_id', 'timestamp', 'object', 'dimensions', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for obj in room_data['objects']:
            writer.writerow(obj)

def save_to_json(room_data):
    """Save room data to JSON"""
    json_data = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except:
            json_data = []
    
    # Create room entry
    room_entry = {
        'room_id': room_data['room_id'],
        'timestamp': room_data['timestamp'],
        'object_count': room_data['object_count'],
        'source_file': f"{room_data['room_id']}.rap",
        'objects': [
            {
                'object': obj['object'],
                'dimensions': obj['dimensions'],
                'x': obj['x'],
                'y': obj['y'],
                'z': obj['z']
            }
            for obj in room_data['objects']
        ]
    }
    json_data.append(room_entry)
    
    # Save updated JSON
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

def process_rap_file(rap_file_path, room_id):
    """Open RAP file and extract object listing via Project -> Export -> Copy Object Listing"""
    print(f"\n[{room_id}]")
    
    try:
        # Clear clipboard first
        pyperclip.copy("")
        print(f"    Debug - Clipboard cleared")
        
        # Open RAP file with Room Arranger
        print(f"  Opening {os.path.basename(rap_file_path)}...")
        process = subprocess.Popen([ROOM_ARRANGER_PATH, rap_file_path])
        time.sleep(DELAYS['app_start'])
        
        # Dismiss registration dialog if present - click OK button
        print(f"  Dismissing any dialogs...")
        pyautogui.press('enter')  # Press Enter to click OK on registration dialog
        time.sleep(DELAYS['dialog_dismiss'])
        # Sometimes needs a second Enter if there are multiple dialogs
        pyautogui.press('enter')
        time.sleep(0.5)
        
        # Navigate to Project -> Export -> Copy Object Listing
        print(f"  Navigating to Copy Object Listing...")
        
        # Press Alt to enter nav bar
        pyautogui.press('alt')
        time.sleep(0.8)
        
        # Press Enter to go into first item (Project)
        pyautogui.press('enter')
        time.sleep(0.8)
        
        # Navigate down 15 times to Export
        print(f"    Navigating to Export...")
        pyautogui.press('down', presses=15)
        time.sleep(0.5)
        
        # Press right to open Export submenu
        pyautogui.press('right')
        time.sleep(0.5)
        
        # Press down 2 times to reach Copy Object List to Clipboard
        print(f"    Selecting Copy Object List to Clipboard...")
        pyautogui.press('down', presses=2)
        time.sleep(0.5)
        
        # Press Enter to select it
        pyautogui.press('enter')
        time.sleep(0.5)
        
        # Press Enter again in export options to copy to clipboard
        print(f"    Confirming copy to clipboard...")
        pyautogui.press('enter')
        time.sleep(DELAYS['after_copy'])
        
        # Debug: Check clipboard immediately
        test_clip = pyperclip.paste()
        if test_clip:
            print(f"    Debug - After menu: Clipboard has {len(test_clip)} chars")
        else:
            print(f"    Debug - After menu: Clipboard still empty")
        
        # Process clipboard data
        room_data = process_clipboard_data(room_id)
        
        if room_data:
            # Save to CSV and JSON
            save_to_csv(room_data)
            save_to_json(room_data)
            
            print(f"  [OK] Processed {room_data['object_count']} objects")
            
            # Show first few objects
            for obj in room_data['objects'][:3]:
                print(f"    - {obj['object']}: {obj['dimensions']}")
            if len(room_data['objects']) > 3:
                print(f"    ... and {len(room_data['objects']) - 3} more")
            
            # Mark as processed
            mark_as_processed(os.path.basename(rap_file_path))
            
            success = True
        else:
            print(f"  [ERROR] No valid object data found in clipboard")
            success = False
        
        # Close Room Arranger
        print(f"  Closing Room Arranger...")
        time.sleep(DELAYS['before_close'])
        pyautogui.hotkey('alt', 'f4')
        time.sleep(2)  # Increased wait for clean close
        
        # Force terminate if still running (safer approach)
        try:
            process.poll()  # Check if process is still running
            if process.returncode is None:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:  # Still running?
                    process.kill()  # Force kill as last resort
        except:
            pass
        
        return success
        
    except Exception as e:
        print(f"  [ERROR] Automation failed: {e}")
        # Try to close Room Arranger
        try:
            pyautogui.hotkey('alt', 'f4')
        except:
            pass
        return False

def main():
    print("Room Arranger RAP File Processor")
    print("="*40)
    print(f"Room Arranger: {ROOM_ARRANGER_PATH}")
    print(f"RAP folder: {RAP_FOLDER}/")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Get list of RAP files
    if not os.path.exists(RAP_FOLDER):
        os.makedirs(RAP_FOLDER)
        print(f"[INFO] Created {RAP_FOLDER}/ folder")
        print(f"[INFO] Place your RAP files in this folder and run again")
        return
    
    rap_files = [f for f in os.listdir(RAP_FOLDER) if f.endswith('.rap')]
    
    if not rap_files:
        print(f"[INFO] No RAP files found in {RAP_FOLDER}/")
        return
    
    # Get already processed files
    processed = get_processed_files()
    
    # Filter out already processed files
    to_process = [f for f in rap_files if f not in processed]
    
    if not to_process:
        print("[INFO] All RAP files have already been processed")
        print(f"[TIP] Delete {PROCESSED_LOG} to reprocess all files")
        return
    
    print(f"Found {len(to_process)} RAP file(s) to process:")
    for f in to_process:
        print(f"  - {f}")
    
    print("\n[WARNING] This will automate your mouse and keyboard!")
    print("DO NOT move the mouse or type during processing")
    print("\nProcess will use: Project -> Export -> Copy Object Listing")
    print("\nStarting in 5 seconds... (Press Ctrl+C to cancel)")
    
    try:
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[CANCELLED]")
        return
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for rap_file in to_process:
        rap_path = os.path.join(RAP_FOLDER, rap_file)
        room_id = os.path.splitext(rap_file)[0]
        
        if process_rap_file(rap_path, room_id):
            success_count += 1
        else:
            failed_files.append(rap_file)
        
        # Wait between files and clear memory
        if to_process.index(rap_file) < len(to_process) - 1:
            print("\n  Waiting before next file...")
            gc.collect()  # Force garbage collection to free memory
            time.sleep(3)  # Increased wait time between files
    
    # Summary
    print("\n" + "="*40)
    print(f"[COMPLETE] Successfully processed {success_count}/{len(to_process)} files")
    
    if success_count > 0:
        print(f"\nOutput files:")
        print(f"  CSV: {CSV_FILE}")
        print(f"  JSON: {JSON_FILE}")
    
    if failed_files:
        print(f"\nFailed to process:")
        for f in failed_files:
            print(f"  - {f}")
        print("\n[TIP] For failed files, try:")
        print("  1. Open Room Arranger manually")
        print("  2. Register the software or dismiss dialog")
        print("  3. Test: Project -> Export -> Copy Object Listing")
        print("  4. Use manual export with process_rooms_batch.py")

if __name__ == "__main__":
    main()