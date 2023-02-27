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

def processing():
    """The goal of this file is to take scraped data and translate it into english then put it into a second table"""
    data = tu.Dataset("site_data") # load the data
    db = tu.PostgresDatabase()  # connect to the database
    db.connect()
    site_data = data._class
    for i in range(len(site_data)):
        # if site is already in db or text is empty then continue
        text = tu.clean_text_data([site_data[i].text]) # check to make sure there is some text to translate
        is_translated, id = db.is_translated(site_data[i].id)
        text_language = tu.get_language(text)
        if is_translated or text[0] == "": #or text_language == "en":
            print(f"Already translated: {site_data[i].name}")
            continue
        
        # remove special characters from the tags, distinguish 'original' and 'additional' data that I collected
        tag = re.sub('[^A-Za-z0-9 ]+', '', site_data[i].tags) 
        if tag != "Additional Data":
            tag = "Original Data"
        site_data[i].tags = tag
    
        try:
            signal.alarm(TIME) # set the alarm to expire after the given amount of time
            if text_language[0] != "en":
                site_data[i].text = tu.azure_translate(text[0])
            else:
                site_data[i].text = text[0]
            site_data[i].language = text_language[0]
            signal.alarm(0) # cancel the alarm
            time.sleep(1)
        except Exception as e:
            logging.info(f"Error: (data_processing.py) {e}")
        
        # Add the translated text to the database
        if site_data[i].language != None:
            print(f"Translated: {site_data[i].language}")
            db.add_translation(site_data[i])

        print(f"name: {site_data[i].name} text: {site_data[i].text[:50]}")
        print(f"{i+1} / {len(site_data)}")


if __name__ == "__main__":
    processing()
