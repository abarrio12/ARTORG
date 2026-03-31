import json

with open("ABA_annotation_last.json", "r") as f:
    data = json.load(f)

def find_region_by_id_anywhere(obj, target_id):
    if isinstance(obj, dict):
        if str(obj.get("id")) == str(target_id):
            return obj
        for key, value in obj.items():
            result = find_region_by_id_anywhere(value, target_id)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_region_by_id_anywhere(item, target_id)
            if result is not None:
                return result

    return None

def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.get("children", []))

target_id = 500
region = find_region_by_id_anywhere(data, target_id)

if region is None:
    print(f"Region with ID {target_id} not found in JSON")
else:
    print("Found region:", region.get("name"))
    print("Acronym:", region.get("acronym"))
    print("ID:", region.get("id"))
    print("Direct children:", len(region.get("children", [])))
    print("Total nodes:", count_nodes(region))

    print("\nDirect children:")
    for child in region.get("children", []):
        print(child.get("name"), "|", child.get("acronym"), "| ID:", child.get("id"))