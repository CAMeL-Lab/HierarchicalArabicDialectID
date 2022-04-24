import os
import re

# get ids that are in training
alghad = []
alriyadh = []
youm7 = []

folder = './AOC-dialectal-annotations/'
files = [i for i in os.listdir(folder) if 'ids' in i]
for file in files:
    with open(folder+file, 'r') as f:
        lines = [i[i.find('/')+1:-1] for i in f.readlines()]
    if 'alghad' in file:
        alghad.extend(lines)
    elif 'alriyadh' in file:
        alriyadh.extend(lines)
    else:
        youm7.extend(lines)

folder_unl = './AOC/AOC_v1.1/'
files = [i for i in os.listdir(folder_unl) if 'xml' in i and (
    'sample' in i) == False and ('processed' in i) == False]

alghad_docs = dict()  # key doc_id, val = list of segments
alriyadh_docs = dict()
youm7_docs = dict()

for file in files:
    print(f'starting {file}')
    with open(folder_unl+file, 'r') as f:
        doc_id = -1
        for i in f.readlines():
            if 'docid' in i:
                doc_id = i[i.find('docid="') +
                           7: i[i.find('docid="')+7:].find('"')+12]
                doc_id = re.sub('[^0-9\_]', '', doc_id)[1:]
                if doc_id[-1] == '_':
                    doc_id = doc_id[:-1]
                # print(doc_id)
            if 'seg' in i:
                seg_id = i[i.find(
                    'seg id="')+8: i[i.find('seg id="')+8:].find('"')+9].zfill(3)
                if re.search('<seg id=".*">(.*)</seg>', i):
                    sentence = re.search(
                        '<seg id=".*">(.*)</seg>', i).group(1).strip()
                    if 'alghad' in file:
                        if (f'{doc_id}_{seg_id}' in alghad) == False:
                            if doc_id in alghad_docs:
                                alghad_docs[doc_id].append(sentence)
                            else:
                                alghad_docs[doc_id] = [sentence]
                    elif 'alriyadh' in file:
                        if (f'{doc_id}_{seg_id}' in alriyadh) == False:
                            if doc_id in alriyadh_docs:
                                alriyadh_docs[doc_id].append(sentence)
                            else:
                                alriyadh_docs[doc_id] = [sentence]
                    elif 'youm7' in file:
                        if (f'{doc_id}_{seg_id}' in youm7) == False:
                            if doc_id in youm7_docs:
                                youm7_docs[doc_id].append(sentence)
                            else:
                                youm7_docs[doc_id] = [sentence]

folder = './data_unlabelled/'
os.mkdir(folder)


def put_into_files(folder, docs):
    os.mkdir(folder)
    for doc in docs.keys():
        if len(docs[doc]) > 0:
            with open(f'{folder}{doc}.doc', 'w') as f:
                for line in docs[doc]:
                    f.write(f'{line}\n')


put_into_files(f'{folder}alghad/', alghad_docs)
put_into_files(f'{folder}alriyadh/', alriyadh_docs)
put_into_files(f'{folder}youm7/', youm7_docs)
