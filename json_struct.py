import json
import os

json_path = '/home1/qiuliwang/Data/Glioma/20220502_labeled/B202005122-2.json'

def getStruct(json_path):
    load_dict = json.load(open(json_path, encoding = 
    'utf8'))

    print(load_dict.keys())

    for one_key in load_dict.keys():
        print('First Level Key:', one_key)
        one_content = load_dict[one_key]
        for one_inner_key in one_content.keys():
            print('\tSecond Level Key:', one_inner_key)
            two_content = one_content[one_inner_key]
            for two_inner_key in two_content.keys():
                print('\t\tThird Level Key:', two_inner_key)
                if two_inner_key == 'regions':
                    three_content = two_content[two_inner_key]
                    print('\t\t\tNumber of Annotations:', len(three_content))
                    content0 = three_content[0]
                    for three_inner_key in content0.keys():
                        print('\t\t\tForth Level Key:', three_inner_key)
                        four_content = content0[three_inner_key]
                        for four_inner_ley in four_content.keys():
                            print('\t\t\t\tFifth Level Key:', four_inner_ley)

files = os.listdir('./')
json_files = []
for one_file in files:
    if '.json' in one_file:
        json_files.append(one_file)

print(json_files)

for one_json in json_files:
    getStruct(one_json)