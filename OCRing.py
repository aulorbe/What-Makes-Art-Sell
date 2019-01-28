import os

file_names = []

for file in os.listdir('/Users/flatironschool/Final-Project/Invoices_PDFs'):
    file_names.append(file)


for file in file_names:
    os.system(f'pdf2txt.py -o {file}.txt /Users/flatironschool/Final-Project/Invoices_PDFs/{file}')

# os.system('pdf2txt.py -o output_test.txt /Users/flatironschool/Desktop/INV19_0229.pdf')
