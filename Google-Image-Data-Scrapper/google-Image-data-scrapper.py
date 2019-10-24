from google_images_download import google_images_download   # importing the library
from googletrans import Translator
translator = Translator()


keyword = 'i am super rich '
# Hebrow ,Russian ,Arabic ,French , English , Spanish ,Italian	
languesges = ['iw','ru','ar','ja','fr','it','es'] 

def get_image_keyword_by_lang( lang = 'en' ):
    try:
        translations = translator.translate(keyword, dest=lang)
        response = google_images_download.googleimagesdownload()
        arguments = {"keywords": translations.text ,"limit":1,"print_urls":True}
        paths = response.download(arguments)
    except:
        print("An exception occurred : in " + lang) 


for lang in languesges:
    get_image_keyword_by_lang(lang)
    
get_image_keyword_by_lang()
