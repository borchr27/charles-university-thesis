import traceback
import numpy as np
import thesis_utils as tu
import re
import time
import logging
import signal

##########################################################################################
# This is the code that I used to translate the text from the website data in the site_data
# db into english then add it to the translated_data table in the db. I used the azure

TIME = 3  # set the timeout duration in seconds
def handle_timeout(signum, frame):
    """A function to handle the timeout for the translator"""
    raise TimeoutError("Timed out")
signal.signal(signal.SIGALRM, handle_timeout)  # set the signal handler for SIGALRM (alarm signal)

def processing(data: tu.Dataset):
    """The goal of this file is to take scraped data and translate it into english then put it into a second table"""
    site_data = data.site_data
    db = tu.PostgresDatabase()  # connect to the database
    db.connect()
    # site_data = site_data._class
    # for i in range(len(site_data)):
    i = 0 
    for row in site_data.itertuples():
        # if site is already in db or text is empty then continue
        text = tu.clean_text_data([row.text]) # check to make sure there is some text to translate
        is_translated, id = db.is_translated(row.id)
        text_language = tu.get_language(text)
        site = tu.Website(id=row.id, name=row.name, i_url=row.initial_url, v_url=row.visited_url, text=row.text, language=None, tags=row.tags, category=row.category, origin=row.origin)
        if is_translated or text[0] == "":
            print(f"Already translated: {row.name}")
            continue
    
        try:
            signal.alarm(TIME) # set the alarm to expire after the given amount of time
            if text_language[0] != "en":
                site.text = tu.azure_translate(text[0])
            else:
                site.text = text[0]
            site.language = text_language[0]
            signal.alarm(0) # cancel the alarm
            time.sleep(1)
        except Exception as e:
            logging.info(f"Error: (data_processing.py) {e}")
        
        # Add the translated text to the database
        if site.language != None:
            print(f"Translated: {site.language}")
            db.add_translation(site)

        print(f"name: {row.name} text: {site.text[:50]}")
        print(f"{i+1} / {len(site_data)}")


if __name__ == "__main__":
    data = tu.Dataset()
    processing(data)
