"""
Code for visualizng in Paraview the annotated Region from Renier Dataset
The code is based on ARA2_annotatuon_info.csv
Sofia 04/04/2025
"""
globals().clear()

# Import necessary modules
from paraview.simple import *
from vtkmodules.vtkCommonDataModel import vtkSelectionNode, vtkSelection
import vtk  # Import VTK for array conversion
import pandas as pd


# Read the CSV file containing the labels annotation and Trees
# Here we have the id of the region, name of the region, acronym, parent_id, parent_acronym & red, green, blue color and structure_order. 
df = pd.read_csv(r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\whole brain\Extract Area Annotated from graph Brain\ARA2_annotation_info.csv")


# ======================= FIND DESCENDANTS WITH ID AND DEGREE =======================

# This function finds all descendants of a given brain region name along with their degree and id. 
# Function: takes a region name as input, searches for the id in the table, goes through every son and returns a list of tuples containing (degree, id, name) for each descendant.
# [(degree, id, name), ...] --> degree = depht in the tree (0 = root, 1 = children, 2 = grandchildren, etc)

def find_descendants_with_id_and_degree(df, given_name):
    '''
    Function to collect descendants of the Region of the Brain we are interested in.
    '''

    # Check if the given name exists in the dataframe
    if given_name not in df['name'].values and given_name != None:
        raise ValueError(f"!!! The area '{given_name}' does not exist. Check the spelling.")
    elif given_name == None:
        return []
    else:
        # Define parent ID
        parent_id = df.loc[df['name'] == given_name, 'id'].values[0]  # Assuming unique names

        # Initialize a list to store the descendants, their degree, and id
        descendants = []

        # Function to find children and recursively find descendants
        def get_children(parent_id, degree):
            children = df[df['parent_id'] == parent_id]  # Get children of the current parent

            for _, child in children.iterrows():
                child_name = child['name']
                child_id = child['id']
                descendants.append((degree, child_id, child_name))  # Store child info

                # Recursively find the descendants of this child (increment degree)
                get_children(child_id, degree + 1)

        # Start the recursive process for the given name
        descendants.append((0, parent_id, given_name))  # Add the root with degree 0
        get_children(parent_id, 1)  # Start the recursion from the root (degree 1 for its children)

        return descendants  # Return all descendants (not just leaves)

from collections import Counter

# ======================= SELECTING POINTS TO KEEP =======================
# This function selects the point IDs to keep based on the descendants found. Gets a desired region, annotations, its descendants & limit up to a certain degree level, 
# and returns the point IDs corresponding to those regions.  Creates a list of point IDs to keep for ParaView visualization (all points from region A + children < degree limit)
# Filtering of points from a region. 

def selecting_points2keep(descendants_with_id_and_degree, given_name, annotation_array, degree_level=10):
    '''
    Function to extract the list of the given name of the region and all the children up to the selected degree level.
    This is the input for ParaView.
    It also prints names and children to verify them.
    Returns the list of point IDs to keep.
    '''

    if given_name == None:
        return []
    else:
        print(f"\nFor given name '{given_name}', we have the following children with their degrees and names:")

        # Extract all annotation values from the array
        annotation_values = [annotation_array.GetValue(i) for i in range(annotation_array.GetNumberOfTuples())]

        # Count occurrences of each annotation
        annotation_counts = Counter(annotation_values)

        # If we have descendants, process them
        if descendants_with_id_and_degree:
            
            annotations_to_keep = set()  # Store relevant annotations
            print("\nChild Regions to Keep (up to degree level", degree_level, "):\n")
            for degree, child_id, child_name in descendants_with_id_and_degree:
                if degree <= degree_level and child_name not in ['Supplemental somatosensory area']:
                    count = annotation_counts.get(child_id, 0)  # Get count of points for this child region

                    print(f"Child region with degree {degree}: {child_name} | Annotation count: {count} points.")

                    annotations_to_keep.add(child_id)  # Store valid child region annotations

            # Get point indices where annotation is in annotations_to_keep
            print("\nAnnotations to keep", annotations_to_keep)
            point_ids_to_keep = [i for i, annotation in enumerate(annotation_values) if
                                 annotation in annotations_to_keep]

            print("\nTotal Points to Keep:", len(point_ids_to_keep))
            return point_ids_to_keep  # Return filtered point indices

        else:
            print(f"No descendants found for given name '{given_name}'.")
            return []  # Return empty list if no relevant points


## THIS CODE NEED TO CHANGE... BECAUSE U NEED TO READ JSON "graph_order” not ids in csv..
#given_name = "Hippocampal region"
#given_name = "Somatomotor areas"
#given_name = None
given_name = "Hypothalamus"

#Find the descendant from the root
descendants_with_id_and_degree = find_descendants_with_id_and_degree(df, given_name)

# ======================= PARAVIEW SELECTION AND EXTRACTION =======================

# Read the paraview data NEED TO START THE CODE ON THAT
data = GetActiveSource()
data_np = servermanager.Fetch(data)

# Get the annotation array from PointData
annotation_array = data_np.GetPointData().GetArray("annotation")
point_ids_to_keep = selecting_points2keep(descendants_with_id_and_degree, given_name, annotation_array, degree_level=2)

# **Fix: Include Process ID = 0 for each selected point**
flat_selection_list = []
for point_id in point_ids_to_keep:
    flat_selection_list.append(0)  # Process ID --> no biological meaning --> 0 indicates to Paraview that all points belong to the same process
    flat_selection_list.append(point_id)  # Actual point ID

SelectIDs(IDs=flat_selection_list, FieldType='POINT',Source=data)


# Now extract and show the selected points into another dataset
extract_selection =ExtractSelection()
Show(extract_selection)
Render()

# Clear the selection
ClearSelection()