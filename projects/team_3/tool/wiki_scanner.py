import requests
from bs4 import BeautifulSoup, Tag
from tool.file_and_directory_management import open_path


def scanner(title, type, dir, filename):
    url = 'https://en.wikipedia.org/wiki/' + title
    resp = requests.get(url)
    list = []

    if resp.status_code == 200:
        path = dir + "\\" + filename
        file = open_path(path, "a")

        soup = BeautifulSoup(resp.text, 'html.parser')
        characters_h2 = soup.find(id="Characters")
        if characters_h2 is None:
            characters_h2 = soup.find(id="Major_characters")
        if characters_h2 is None:
            return "Characters section not found"

        nextNode = characters_h2.next_element
        while True:
            if isinstance(nextNode, Tag):
                if nextNode.name == "ul":
                    for a in nextNode.findAll(type):
                        text = a.text
                        if type == "b":
                            list_of_texts = standarize_name_of_character(text)
                            for text in list_of_texts:
                                print(text)
                                file.write(text + "\n")
                                list.append(text)
                        else:
                            print(text)
                            file.write(text + "\n")
                            list.append(text)
            if isinstance(nextNode, Tag):
                if nextNode.name == "h2":
                    break

            nextNode = nextNode.next_element

        file.close()

    else:
        print("Error")

    return list


def get_list_of_characters(list_dir_path, title):
    return scanner(title, "b", list_dir_path, title)


def get_descriptions_of_characters(descriptions_dir_path, title):
    return scanner(title, "li", descriptions_dir_path, title)


def standarize_name_of_character(name):
    names = []
    new_name_1 = ""
    new_name_2 = ""

    new_char = name.replace(":", "").replace(".", "").replace("\"", "")
    if new_char.find("&") > -1 or new_char.find("and") > -1:
        elements = new_char.split()
        if len(elements) == 3:
            new_name_1 = elements[0]
            new_name_2 = elements[2]
        else:
            separator_index = elements.index(
                "&" if new_char.find("&") > -1 else "and")
            name1 = elements[0:separator_index]
            name1.append(elements[-1])
            name2 = elements[separator_index + 1:]
            for n in name1:
                new_name_1 += " " + n

            for n in name2:
                new_name_2 += " " + n

        names.append(new_name_1)
        names.append(new_name_2)
    else:
        names.append(new_char)

    return names
