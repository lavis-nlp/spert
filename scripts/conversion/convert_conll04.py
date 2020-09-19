import argparse
import csv
import json

MAPPING = {'COMMA': ',', '-LRB-': '(', '-RRB-': ')'}


def replace_token(token):
    if token in MAPPING:
        return MAPPING[token]

    return token


def convert(source_path, indices_path, dest_path):
    indices = set([int(s.split(':')[0]) for s in open(indices_path)])

    lines = []
    with open(source_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)

    converted_documents = []

    entities = dict()
    document = dict(tokens=[], entities=[], relations=[])

    new_doc = 0
    entity_id = 0
    doc_index = -1

    for line in lines:
        if not line:
            new_doc += 1

            if new_doc == 2:
                if doc_index in indices:
                    converted_documents.append(document)

                new_doc = 0
                entity_id = 0
                document = dict(tokens=[], entities=[], relations=[])
                entities = dict()

        # entities
        if len(line) == 9:
            doc_index = int(line[0])
            entity_type = line[1]
            token_idx = int(line[2])
            token = line[5]

            token = replace_token(token)
            document['orig_id'] = doc_index

            if entity_type == 'O':
                document['tokens'].append(token)
            else:
                start = len(document['tokens'])
                tokens = [replace_token(t) for t in token.split('/')]
                end = len(tokens) + start

                document['tokens'].extend(tokens)
                document['entities'].append(dict(type=entity_type, start=start, end=end))

                entities[token_idx] = entity_id
                entity_id += 1

        # relation
        if len(line) == 3:
            head_token_idx = int(line[0])
            tail_token_idx = int(line[1])
            rel_type = line[2]

            head = entities[head_token_idx]
            tail = entities[tail_token_idx]

            relation = dict(type=rel_type, head=head, tail=tail)
            document['relations'].append(relation)

    json.dump(converted_documents, open(dest_path, 'w'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--indices_path', type=str, help="Path to split indices file")
    arg_parser.add_argument('--dest_path', type=str, help="Destination file path (JSON format)")

    args = arg_parser.parse_args()
    convert(args.source_path, args.indices_path, args.dest_path)
