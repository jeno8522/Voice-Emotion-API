# Imports the Google Cloud client library
import os
from google.cloud import language_v1
import json
from collections import OrderedDict

credential_path = "meecord-223cc-9946ac5d75e5.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
# Instantiates a client


def analyzeEntities(text_content):
    """
    Analyzing Entities in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'California is a state.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "ko"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entities(
        request={'document': document, 'encoding_type': encoding_type})

    res = OrderedDict()

    # Loop through entitites returned from the API
    for entity in response.entities:
        sub_json = OrderedDict()

        print(u"Representative name for the entity: {}".format(entity.name))
        sub_json["entity_name"] = entity.name

        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        print(u"Entity type: {}".format(
            language_v1.Entity.Type(entity.type_).name))
        sub_json["entity_type"] = entity.type_
        
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(entity.salience))
        sub_json["entity_salience"] = entity.salience

        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        meta_json = OrderedDict()
        for metadata_name, metadata_value in entity.metadata.items():
            print(u"{}: {}".format(metadata_name, metadata_value))
            meta_json[metadata_name] = metadata_value

        sub_json["metadata"] = meta_json
        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        
        
        mention_json = OrderedDict()
        for mention in entity.mentions:
            print(u"Mention text: {}".format(mention.text.content))
            mention_json["mention_text"] = mention.text.content
            # Get the mention type, e.g. PROPER for proper noun
            print(
                u"Mention type: {}".format(
                    language_v1.EntityMention.Type(mention.type_).name)
            )
            mention_json["mention type"] = language_v1.EntityMention.Type(mention.type_).name
        sub_json["mention"] = mention_json
        
        
        res[entity.name] = sub_json


    return res
    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    # print(u"Language of the text: {}".format(response.language))


if __name__ == '__main__':
    analyzeEntities("")
