import argparse
import json

MAPPING = {'-LSB-': '[', '-RSB-': ']', '-LRB-': '(', '-RRB-': ')'}


def replace_brackets(token):
    if token in MAPPING:
        return MAPPING[token]

    return token


def convert_doc(orig_doc):
    total_sentences = orig_doc['sentences']
    total_entities = orig_doc['ner']
    total_relations = orig_doc['relations']
    orig_id = orig_doc['doc_key']

    converted_docs = []
    offset = 0

    assert len(total_sentences) == len(total_entities) == len(total_relations)
    for s_idx, (tokens, entities, relations) in enumerate(zip(total_sentences, total_entities, total_relations)):
        converted_entities = []
        converted_relations = []

        entity_mapping = dict()

        for idx, e in enumerate(entities):
            start = e[0] - offset
            end = e[1] - offset
            prep_entity = dict(type=e[2], start=start, end=end+1)
            converted_entities.append(prep_entity)

            entity_mapping[(start, end)] = idx

        for rel in relations:
            head_start = rel[0] - offset
            head_end = rel[1] - offset
            head_idx = entity_mapping[(head_start, head_end)]

            tail_start = rel[2] - offset
            tail_end = rel[3] - offset
            tail_idx = entity_mapping[(tail_start, tail_end)]

            rel_type = rel[4].capitalize()
            converted_relation = dict(type=rel_type, head=head_idx, tail=tail_idx)
            converted_relations.append(converted_relation)

        offset += len(tokens)

        doc = dict(tokens=[replace_brackets(t) for t in tokens], entities=converted_entities,
                   relations=converted_relations, orig_id=orig_id + '_' + str(s_idx))
        converted_docs.append(doc)

    return converted_docs


def convert(source_path, dest_path):
    data = open(source_path).read()
    docs = data.split('\n')
    converted_docs = []

    for doc in docs:
        if doc.strip():
            doc_dict = json.loads(doc)
            converted_doc = convert_doc(doc_dict)
            converted_docs.extend(converted_doc)

    json.dump(converted_docs, open(dest_path, 'w'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--dest_path', type=str, help="Destination file path (JSON format)")

    args = arg_parser.parse_args()
    convert(args.source_path, args.dest_path)
