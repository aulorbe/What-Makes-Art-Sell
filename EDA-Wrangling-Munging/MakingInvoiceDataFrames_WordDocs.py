# Making a list of dictionaries, where each dictionary is an invoice with all the data I need

import re
import os

all_files = [] # to pass to function

for file in os.listdir("/Users/flatironschool/Final-Project/Invoices_WordDocs/relevant-invoices"):
    all_files.append(f'/Users/flatironschool/Final-Project/Invoices_WordDocs/relevant-invoices/{file}')

def make_doc_dicts(files):

    list_of_dicts = []

    for file in files:

        file_dict = {}

        with open(file, 'rb') as f:

            titles = []
            artists = []
            media = []
            dimensions = []
            dates = []
            date_sold = []

            for line in f:

                match_inv = re.search(b'Invoice\s+\d(.+)', line, re.IGNORECASE)
                match_artist = re.search(b'^(Artist:)\s+(.+)', line, re.IGNORECASE)
                match_title = re.search(b'^(Title:)\s+(.+)', line, re.IGNORECASE)
                match_med = re.search(b'^(Medium:)\s+(.+)', line, re.IGNORECASE)
                match_dims = re.search(b'^(Dimensions:)\s+(.+)', line, re.IGNORECASE)
                match_datecreated = re.search(b'^(Date:\s+(.+))', line, re.IGNORECASE)
                match_datesold = re.search(b'(\s?January |\s?February |\s?March |\s?April |\s?May |\s?June |\s?July |\s?August |\s?September |\s?October |\s?November |\s? December )(\s?\d{1,2}(\D? )([0-9]{4}))', line, re.IGNORECASE)

                if match_inv:
                    file_dict['INVOICE ID'] = match_inv.group(0)

                if match_artist:
                    artists.append(match_artist.group(0)[8:])

                if match_title:
                    titles.append(match_title.group(0)[7:])

                if match_med:
                    media.append(match_med.group(0)[8:])

                if match_dims:
                    dimensions.append(match_dims.group(0)[12:])

                if match_datecreated:
                     dates.append(match_datecreated.group(0)[6:])

                if match_datesold:
                    file_dict['Date Sold'] = match_datesold.group(0)

            file_dict['Titles'] = titles
            file_dict['Artist'] = artists
            file_dict['Medium'] = media
            file_dict['Dimensions'] = dimensions
            file_dict['Date Created by Artist'] = dates

            list_of_dicts.append(file_dict)

    return list_of_dicts
