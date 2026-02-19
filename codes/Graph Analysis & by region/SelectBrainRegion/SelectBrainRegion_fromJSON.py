"""
Code for visualizing annotated regions in Paraview (JSON version)
Updated to use ABA_annotation_last.json instead of deprecated CSV
Ana Barrio 2025
"""

'''
2 options:
- A: flatten the JSON into a CSV-like structure (id, name, parent_id) and reuse previous code
- B: adapt the previous code to work directly with the JSON structure

>>>>  This is option B <<<<

'''

'''
Difference with previous CSV-based code:
- We now search the JSON tree for the region by name --> JSON is not a table (no columns, rows) --> it's a nested structure, each region has already its children inside it. 
No need for parent_id, the node of the tree = given_name is already the parent. 
- We get descendants directly from the JSON structure  --> runs only through the given node and its children recursively
- We use the "id" field from JSON for annotation matching
'''


globals().clear()

# ========================== IMPORTS =============================
from paraview.simple import *
from vtkmodules.vtkCommonDataModel import vtkSelectionNode, vtkSelection
from collections import Counter
import json


# ========================== LOAD JSON =============================
#json_path = r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\whole brain\Extract Area Annotated from graph Brain\ABA_annotation_last.json"
json_path = "/home/admin/Ana/MicroBrain/whole brain/Select Region/ABA_annotation_last.json"
# open in read mode
with open(json_path, "r") as f:
    atlas = json.load(f) # all content of the JSON file as Python objects (dictionaries, lists, int, float, etc.)

root = atlas["msg"][0]   # root of full ABA hierarchy

# ========================== TREE UTILITIES =============================
def find_node_by_name(node, name):
    """Recursively search region (node) by name in the JSON tree"""
    
    # 1. Check if current node matches the name
    
    if node["name"] == name: # = df.loc[df['name'] == name]
        return node
    
    # 2. If not, search in its children
    for child in node.get("children", []): # = df[df['parent_id'] == node['id']]
        n = find_node_by_name(child, name) # child node recursion
        if n is not None:
            return n
    # 3. If not found (not in node or descendent), return None
    return None


def get_descendants(node, degree=0, results=None):
    """Collect node and all its descendants with their degree of separation"""
    if results is None:
        results = []
    results.append((degree, node["id"], node["name"])) 
    for child in node.get("children", []):
        # Recursively find the descendants of this child (increment degree)
        get_descendants(child, degree + 1, results) 
    return results


# ========================== SELECTION LOGIC =============================
def selecting_points_from_json(descendants, given_name, annotation_array, degree_level=2):
    """Returns point indices from Paraview annotation array that belong to the given 
    region and its descendants up to degree_limit
    """
    if given_name == None:
        return []
    else:
        print(f"\nFor the given name: '{given_name}', we have the following children with their degrees and names:\n")

    # Convert annotation array to Python list
    
    # Extract all annotation values from the array
    annotation_values = [annotation_array.GetValue(i) for i in range(annotation_array.GetNumberOfTuples())]
    
    # Count occurrences of each annotation
    annotation_counts = Counter(annotation_values)
    
    # If we have descendants, process them
    if descendants:
        annotations_to_keep = set()  # Store relevant annotations
        print("\nChild Regions to Keep (up to degree level", degree_level, "):\n")
        for degree, child_id, child_name in descendants:
            if degree <= degree_level:
                count = annotation_counts.get(child_id, 0) # Get count of points for this child region
                print(f"Child region with degree {degree} | {child_name} | ID={child_id} | Count={count}")

                annotations_to_keep.add(child_id) # Store valid child region annotations
                
        print("\nAnnotations to keep", annotations_to_keep)
        selected_points = [i for i, annotation in enumerate(annotation_values)
            if annotation in annotations_to_keep
        ]

        print("\nTotal points to keep:", len(selected_points))
        return selected_points
    else:
        print(f"No descendants found for given name '{given_name}'.")
        return []  # Return empty list if no relevant points



# ========================== RUN FOR ANY REGION =============================
#given_name = "Hypothalamus"   # change this if needed
#given_name = "Somatomotor areas"   
given_name = "Hippocampal region"

region_node = find_node_by_name(root, given_name)
if region_node is None:
    raise ValueError(f"Region '{given_name}' not found in JSON atlas.")

descendants = get_descendants(region_node)


# ========================== PARAVIEW SELECTION =============================
node = find_node_by_name(root, given_name)
if node is None:
    raise ValueError(f"Region '{given_name}' not found in JSON.")
data = GetActiveSource()
data_np = servermanager.Fetch(data)

annotation_array = data_np.GetPointData().GetArray("annotation")

points_to_keep = selecting_points_from_json(
    descendants, given_name, annotation_array, degree_level=2
)

# Convert to ParaView ID selection format
flat_selection_list = []
for pid in points_to_keep:
    flat_selection_list.append(0)   # Process ID
    flat_selection_list.append(pid) # Actual point ID

SelectIDs(IDs=flat_selection_list, FieldType='POINT', Source=data)

extract_selection = ExtractSelection()
Show(extract_selection)
Render()
ClearSelection()