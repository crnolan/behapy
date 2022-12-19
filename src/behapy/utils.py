import json


def get_parameters(fnl):
    f = open(fnl)
    data = json.load(f)
    f.close()
    return data
