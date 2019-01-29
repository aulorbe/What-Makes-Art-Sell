# Looping through .txt invoices (the ones converted from pdf) & only grabbing them if they are 1) greater than 2 bytes 2) have the phrase 'Artist:' in them and/or '1stDibs Ref:'

import os
import shutil

folder = '/Users/flatironschool/Final-Project/Invoices_Txt'

files_with_artists = [] #412 unique files (figure out why you're writing duplicates of each file in the script below later)

bad_files = [] #33 files -- deal with these later

for file in os.listdir("/Users/flatironschool/Final-Project/Invoices_PDFsToText"):
    if os.path.getsize(f'/Users/flatironschool/Final-Project/Invoices_PDFsToText/{file}') > 2:
        with open(f'/Users/flatironschool/Final-Project/Invoices_PDFsToText/{file}', 'rb') as f:
            for line in f:
                if 'Artist:' or '1stDibs Ref 'in line:
                    shutil.copy(f'/Users/flatironschool/Final-Project/Invoices_PDFsToText/{file}', folder)
    else:
        bad_files.append(file)
