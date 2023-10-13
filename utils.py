import re

def preprocess(dataset, source, taret):
    
    data_list = []
    for data in dataset:
        temp_dict = {}

        item = data["translation"]

        source_text = re.sub(r'\s+', ' ', item[source])
        target_text = re.sub(r'\s+', ' ', item[target])

        temp_dict[source] = source_text.lower() 
        temp_dict[target] = target_text.lower()

        data_list.append(temp_dict)

    return data_list