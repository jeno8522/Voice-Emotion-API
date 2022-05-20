# -*- coding:utf-8 -*-
import urllib3
import json
from collections import OrderedDict

openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU_spoken"

accessKey = "e9db9b68-f41e-4b6a-8456-582d9fffc181"
analysisCode = "ner"
text = ""


def analyzeEntities(t):
    text = t
    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL, headers={"Content-Type": "application/json; charset=UTF-8"}, body=json.dumps(requestJson)
    )

    # print("[responseCode] " + str(response.status))
    # print("[responBody]")
    #print(str(response.data, "utf-8"))
    data = json.loads(str(response.data, "utf-8"))

    res = OrderedDict()
    for sentence in data['return_object']['sentence']:

        for ne in sentence['NE']:
            sub_json = OrderedDict()
            sub_json['text'] = ne['text']
            sub_json['type'] = ne['type']
            sub_json['weight'] = ne['weight']
            # print(ne)

            res[ne['text']] = sub_json
    return res
# print(data['return_object']['sentence'])
