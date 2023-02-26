import traceback
import translators as ts
import translators.server as tss
import numpy as np
import thesis_utils as tu
import re
import time
import logging
import signal


TIME = 3  # set the timeout duration in seconds
def handle_timeout(signum, frame):
    """A function to handle the timeout for the translator"""
    raise TimeoutError("Timed out")
signal.signal(signal.SIGALRM, handle_timeout)  # set the signal handler for SIGALRM (alarm signal)



def get_available_translators(arr: list):
    """Use as a generator to get a list of available translators"""
    i = 0
    while True:
        yield arr[i]
        i = (i + 1) % len(arr)

def processing():
    """The goal of this file is to take scraped data and translate it into english then put it into a second table"""
    translator_list = ts.translators_pool # load translator options
    translator = get_available_translators(translator_list)
    data = tu.Dataset("site_data") # load the data
    db = tu.PostgresDatabase()  # connect to the database
    db.connect()
    site_data = data._class
    for i in range(len(site_data)):
        # if site is already in db or text is empty then continue
        text = tu.clean_text_data([site_data[i].text]) # check to make sure there is some text to translate
        is_translated, id = db.is_translated(site_data[i].id)
        is_english = tu.get_language(text)
        if is_translated or text[0] == "" or is_english == "en":
            print(f"Already translated: {site_data[i].name}")
            continue
        
        # tag processing remove special characters from the tags, helps distinguish between 'original' given and 'additional' data that I collected
        tag = re.sub('[^A-Za-z0-9 ]+', '', site_data[i].tags) 
        if tag != "Additional Data":
            tag = "Original Data"
        site_data[i].tags = tag
        
        breaker = 0
        while breaker < len(translator_list):            
            t = next(translator)
            try:
                signal.alarm(TIME) # set the alarm to expire after the given amount of time
                is_translated = ts.translate_text(query_text=text[0], translator=t, is_detail_result=True)
                site_data[i].language = is_translated["detectedLanguage"]["language"]
                site_data[i].text = is_translated["translations"][0]["text"]
                signal.alarm(0) # cancel the alarm
                time.sleep(10)
                break
            except Exception as e:
                print(i, t)
                site_data[i].text = f"Error: count={breaker}"
                breaker += 1
                if breaker == 20: return
                logging.info(f"Error: (data_processing.py) {e}")
                continue
        
        # Add the translated text to the database
        if site_data[i].language != None:
            print(f"Translated: {site_data[i].language}")
            db.add_translation(site_data[i])

        print(f"name: {site_data[i].name} text: {site_data[i].text}")
        print(f"{i+1} / {len(site_data)}")

if __name__ == "__main__":
    processing()
