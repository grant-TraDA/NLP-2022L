import json

from tool.gender_checker import get_personal_titles


def has_intersection(ent_1, ent_2):
    if ent_1[1] <= ent_2[0] or ent_2[1] <= ent_1[0]:
        return False
    return True


def fix_personal_titles(annotations):
    personal_titles = tuple(get_personal_titles())
    for anno in annotations:
        for ent in anno['entities']:
            text = anno['content'][ent[0]:ent[1]]
            if text.startswith(personal_titles):
                ent[0] += (1 + len(text.split(' ')[0]))
    return annotations


def personal_titles_stats(annotations):
    personal_titles = tuple(get_personal_titles())
    personal_title_annotated = {}
    personal_title_not_annotated = {}

    for title in personal_titles:
        personal_title_annotated[title] = 0
        personal_title_not_annotated[title] = 0

    for anno in annotations:
        for ent in anno['entities']:
            text = anno['content'][ent[0]:ent[1]]
            if text.startswith(personal_titles):
                title = text.split(' ')[0]
                personal_title_annotated[title] += 1
            elif any(ext in anno['content'][(ent[0] - 8):ent[0]] for ext in personal_titles):
                title = anno['content'][(
                    ent[0] - 8):ent[0]].split(' ')[-2].strip('"').strip("'")
                try:
                    personal_title_not_annotated[title] += 1
                except Error:
                    print(title)

    return personal_title_annotated, personal_title_not_annotated


def read_annotations(annotation_path):
    with open(annotation_path, encoding='utf-8') as f:
        annotations = f.read()
    annotations = json.loads(annotations.encode('utf-8').decode('utf-8'))

    return annotations
