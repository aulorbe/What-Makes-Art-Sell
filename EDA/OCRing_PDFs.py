# Looping through folder of PDF invoices and turning each one into a .txt file

import os

file_names = []

for file in os.listdir('/Users/flatironschool/Final-Project/Invoices_PDFs'):
    file_names.append(file)

for file in file_names:
    os.system(f'pdf2txt.py -o {file}.txt /Users/flatironschool/Final-Project/Invoices_PDFs/{file}')
