import os
import pandas as pd

print("--- DEBUG STARTING ---")

# 1. Check where Python thinks it is running
current_dir = os.getcwd()
print(f"1. Current Working Directory: {current_dir}")

# 2. Check if the data file exists
file_name = 'SMSSpamCollection'
file_path = os.path.join(current_dir, file_name)

if os.path.exists(file_path):
    print(f"2. SUCCESS: '{file_name}' found!")
    
    # 3. Try to read the first 5 lines
    try:
        print("3. Attempting to read file...")
        df = pd.read_csv(file_name, sep='\t', names=['label', 'message'])
        print(f"4. SUCCESS: Read {len(df)} rows.")
        print("   First 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"4. ERROR reading file: {e}")

else:
    print(f"2. FAILURE: '{file_name}' NOT found.")
    print("   Looked in:", file_path)
    print("   Please verify the file name has no hidden extension (like .txt).")

print("--- DEBUG FINISHED ---")
input("Press Enter to exit...")