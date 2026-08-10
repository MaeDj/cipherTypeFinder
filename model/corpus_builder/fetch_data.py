import json
import mimetypes
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = 'https://de-crypt.org/decrypt-web'

ALLOWED_PLAINTEXT_LANGS = {'french', 'portuguese', 'german', 'spanish', 'latin', 'italian', 'english'}

#None plaintext lang is allowed to train the model on absence of data also
def is_allowed_plaintext_lang(plaintext_lang):
    # plaintext_lang can list several languages (e.g. "French? Portuguese?"),
    # possibly separated by commas/spaces and marked uncertain with "?"
    langs = re.findall(r'[A-Za-z]+', plaintext_lang)
    if not langs:
        return False
    return all(lang.lower() in ALLOWED_PLAINTEXT_LANGS for lang in langs)

def login (base):
    login_resp = requests.post(f'{base}/api/login', data={
        'username': os.getenv('DECODE_RECORD_LOGIN'),
        'password': os.getenv('DECODE_RECORD_PASSWORD')
    })
    #A failed login (bad creds, downtime, rate limit) must not silently turn into a garbage
    #bearer token - that only surfaces later as confusing KeyErrors deep in the callers below.
    login_resp.raise_for_status()
    login=login_resp.text
    if not login.startswith('{"JWT":"') or not login.endswith('"}'):
        raise RuntimeError(f"Unexpected login response, not a JWT payload: {login[:200]!r}")
    login_clear = login.replace('{"JWT":"', '') #Text processing to only keep the core code
    login_clear = login_clear.replace('"}', '')

    # Step 2: use the token
    headers = {'X-Authorization': f'Bearer {login_clear}'}
    return(headers)

def get_cipher(base):
    id_rec=[]
    img=[]
    img_doc=[]

     #adjust thhe wanted number of pages
    for index in range(1,200):
        #A single bad page (transient network error, malformed response) must not discard every
        #id_rec already collected from the pages before it - log and move on to the next page
        #instead of letting the whole fetch crash here.
        try:
            #re-login for each page since the token only lasts 10 min
            headers=login(base)
            record=requests.get(f'{base}/api/list/records?page={index}&record_type=1', headers=headers)
            record.raise_for_status()
            record_doc=record.json()
            for rec in record_doc['records']:
                if rec['id'] not in id_rec:
                    id_rec.append(rec['id'])
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"Warning: could not fetch page {index}, skipping ({e})")
            continue
    return(id_rec)
#print(id_rec)


def get_image_metadata(base,id_rec):
    view_rec=[]
    for id in id_rec:
        headers=login(base)
        view=requests.get(f'{base}/api/view/records/{id}', headers=headers)
        try:
            view.raise_for_status()
            view_doc=view.json()
            if (view_doc['records']['origin_region'] is not None or view_doc['records']['origin_city'] is not None) and view_doc['records']['start_year'] is not None and (view_doc['records']['cipher_types'] is not None and view_doc['records']['cipher_types'] != '') and '6' not in [c.strip() for c in view_doc['records']['cipher_types'].split(',')] and is_allowed_plaintext_lang(view_doc['records'].get('plaintext_lang')):
                image = requests.get(f'{base}/api/view/images/{id}', headers=headers)
                image.raise_for_status()
                images_doc = image.json()
                view_rec.append({id:{'origin_region':view_doc['records']['origin_region'], 'origin_city':view_doc['records']['origin_city'], 'start_year':view_doc['records']['start_year'], 'cipher_types':view_doc['records']['cipher_types'],'plaintext_lang':view_doc['records']['plaintext_lang'],'symbol_sets':view_doc['records']['symbol_sets'],'url':images_doc['images']['path']['url'], 'image_type':images_doc['images']['path']['type']}})
        #Narrowed from a bare `except:` - that swallowed KeyboardInterrupt/SystemExit along with the
        #actual expected failure modes (missing keys, bad JSON, network errors), and gave no clue
        #which of those it was.
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"Unusable image in records {id} ({e})")

    print(view_rec)
    print(len(view_rec))
    return(view_rec)

#WARNING, There is in total 200 pages
#WARNING: Authorization stands for 10 min

def image_extension(image_type):
    # image_type may be a MIME type (e.g. "image/png") or a plain extension (e.g. "png")
    if not image_type:
        return '.jpg'
    if '/' in image_type:
        return mimetypes.guess_extension(image_type) or '.jpg'
    return image_type if image_type.startswith('.') else f'.{image_type}'

# Step 3: build a corpus folder with the images and metadata from view_rec
def corpus_build(base,view_rec):
    CORPUS_DIR = Path('/media/mae/PHILIPS UFD/corpus_cipherTypeFinder_Caramba')
    #binarize.py reads images from {DATASET_PATH}/original_corpus/img
    IMG_DIR = CORPUS_DIR / 'original_corpus' / 'img'
    METADATA_DIR = CORPUS_DIR / 'original_corpus' / 'metadata'
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for entry in view_rec:
        for id, rec_meta in entry.items():
            # re-login for each download since the token only lasts 10 min
            headers=login(base)

            try:
                img_resp = requests.get(rec_meta['url'], headers=headers)
                img_resp.raise_for_status()
                ext = image_extension(rec_meta.get('image_type'))
                with open(IMG_DIR / f'{id}{ext}', 'wb') as f:
                    f.write(img_resp.content)
            except Exception:
                print(f"Could not download image for record {id}")
                continue

            metadata = {
                'origin_region': rec_meta['origin_region'],
                'origin_city': rec_meta['origin_city'],
                'start_year': rec_meta['start_year'],
                'cipher_types': rec_meta['cipher_types'],
                'plaintext_lang': rec_meta['plaintext_lang'],
                'symbol_sets': rec_meta['symbol_sets'],
            }
            with open(METADATA_DIR / f'{id}.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            saved += 1

            print(f"Images {id} successfully downloaded")

    print(f"Corpus saved to {CORPUS_DIR} ({saved} records)")



def main ():
    print(f'Fetching cipher\'s id\'s...')
    id_rec=get_cipher(BASE) #get all available cipher id
    print(f'Cipher IDs successfully fetched: {id_rec}')
    print(f'Fetching available cipher\'s information...')
    view_rec=get_image_metadata(BASE,id_rec) #get all associated records that correspond to restrictions, their metadata and url
    print(f'Cipher information successfully fetched: {len(view_rec)} cipher processed')
    print('Building corpus...')
    corpus_build(BASE,view_rec) #create locally a folder with all the data organized




main()
