###
#Writter/Reader for user input
#Get user's document and information about it
#
###
import os
import json

CHARACTER_TYPES = ["sign", "numerical", "alchemy"]
INFO_FILENAME = "document_info.json"


#Ask the user for the folder path containing his document, until a valid existing directory is given
def prompt_folder_path():
    folder_path = ""
    while not os.path.isdir(folder_path):
        folder_path = input("Enter the path to the folder containing your document: \n").strip()
        if not os.path.isdir(folder_path):
            print(f"'{folder_path}' is not a valid folder, please try again.")
    return folder_path


#Load previously registered information for this folder, if any, otherwise return empty info
def read_existing_info(folder_path):
    info_path = os.path.join(folder_path, INFO_FILENAME)
    if os.path.isfile(info_path):
        with open(info_path, 'r', encoding='utf-8') as info_file:
            return json.load(info_file)
    return {"origin": None, "date": None, "plain_text_language": None, "character_type": None}


#Ask the user for a free text field, keeping the current value if the user presses Enter and clearing it if he types "none"
def prompt_free_field(field_label, current_value):
    prompt_text = f"{field_label} (current: {current_value}) - press Enter to keep, type 'none' to clear, or enter a new value: \n"
    answer = input(prompt_text).strip()
    if answer == "":
        return current_value
    if answer.lower() == "none":
        return None
    return answer


#Ask the user for the character type(s) among CHARACTER_TYPES, accepting a comma separated list, keeping or clearing like other fields
def prompt_character_type(current_value):
    choices_text = ", ".join(CHARACTER_TYPES)
    prompt_text = f"Character type [{choices_text}] (current: {current_value}) - press Enter to keep, type 'none' to clear, or enter a comma separated selection: \n"
    while True:
        answer = input(prompt_text).strip()
        if answer == "":
            return current_value
        if answer.lower() == "none":
            return None
        selection = [token.strip().lower() for token in answer.split(",") if token.strip() != ""]
        if all(token in CHARACTER_TYPES for token in selection) and selection:
            return selection
        print(f"Invalid selection, please only use values among: {choices_text}")


#Gather all 4 information from the user, each of them can stay None if the user has no information about it
def collect_document_info(folder_path):
    existing_info = read_existing_info(folder_path)

    origin = prompt_free_field("Origin", existing_info.get("origin"))
    date = prompt_free_field("Date", existing_info.get("date"))
    plain_text_language = prompt_free_field("Plain text language", existing_info.get("plain_text_language"))
    character_type = prompt_character_type(existing_info.get("character_type"))

    return {
        "origin": origin,
        "date": date,
        "plain_text_language": plain_text_language,
        "character_type": character_type,
    }


#Register the collected information into a JSON file located in the user's document folder
def write_info(folder_path, info):
    info_path = os.path.join(folder_path, INFO_FILENAME)
    with open(info_path, 'w', encoding='utf-8') as info_file:
        json.dump(info, info_file, ensure_ascii=False, indent=2)
    return info_path


def main():
    folder_path = prompt_folder_path()
    info = collect_document_info(folder_path)
    info_path = write_info(folder_path, info)
    print(f"Document information successfully registered into '{info_path}'")


main()
