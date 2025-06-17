import os
import csv
from collections import deque
import pandas as pd



def get_last_processed_patch_name(path: str) -> str:
    """
    Returns the last processed patch name if a respective csv file exist,
    else returns patch_-1_-1_-1

    Parameters:
      path        : path to the respective csv file 

    Returns:
        The last processed patch name
    """
    try:
        default_patch_name = 'patch_-1_-1_-1'

        if not os.path.exists(path):
            print(f'~ No previous processed csv file exist for this progression. Setting the last processed patch name to {default_patch_name}')
            return default_patch_name

        with open(file=path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            last_row = deque(reader, maxlen=1)
            if not last_row:
                print(f'~~ No previous processed patch found for this progression. Setting the last processed patch name to {default_patch_name}')
                return default_patch_name
            
            
            last_processed_patch_name = last_row[0]['patch_name']
            print(f"~~~ Found {last_processed_patch_name} as the last processed patch name for the progression.")
            return last_processed_patch_name
    except Exception as exc:
        raise Exception(
            f"~ Error occured when getting the last processed patch name.\n{exc}"
        )
    


def append_neurons_coords_to_csv(neurons_coords: pd.DataFrame, path: str):
    """
    This function is responsible for saving the detected neurons to the given path
    csv file.

    Parameters:
        neurons_coords : A list containing all the neurons coords detected by the DoG model
        path           : The path of the csv file that does/will contain the detected neurons coords
    """
    try:
        # if such a file exist
        exist = os.path.isfile(path)

        # save the neurons coords to the given file, with
        # header based on if a file exist
        neurons_coords.to_csv(
            path,
            mode='a',
            header=not exist,
            index=False
        )
    except Exception as exc:
        raise Exception(
            f"~~~~~~~~~~~~~~~~~ Error occured when appending detected neuron coords to the csv file.\n{exc}"
        )


def get_all_progressions_saved(outdir: str) -> dict:
    """
    This function is getting the progressions dict which contains the start percetage as the key and 
    a tuple of till percentage and filename as the value of all the progressions saved at the outdir path.

    Parameters:
        outdir : The path where all the detected neurons coords progression csv files are stored

    Returns:
        progression : A dict containing info regarding all the saved progressions.
    """
    try:
        # contain all the progression where key is the start pct, and
        # value is a tuple (till pct, file name)
        progression = {}
        # for every file in outdir folder
        for prog in os.listdir(outdir):
            # skip if it is not the right csv file
            if not prog.endswith("_neurons_coords.csv"):
                continue

            print(f"~~~~~~~~~~~~~~~~~~~ Found progression file: {prog}")
            # get the start and till pct value
            prog_values = [int(prog_value) for prog_value in prog.split("_") if prog_value.isdigit()]

            # add as a {start pct : (till pct, filename)} item
            progression[prog_values[0]] = (prog_values[1], prog)

        # return the sorted progression based on keys
        return dict(sorted(progression.items()))
    except Exception as exc:
        raise Exception(
            f"~~~~~~~~~~~~~~~~~~~ Error when getting all the progression values of the saved csv file.\n{exc}"
        )



def check_all_progressions_csv_files_exist(outdir: str):
    """
    This function is responsible checking if all the progression exist that is from 0 to 100
    and it also returns the progressions dict which contains the start percetage as the key and 
    a tuple of till percentage and filename as the value of all the progressions.

    Parameters:
        outdir : The path where all the detected neurons coords progression csv files are stored

    Returns:
        check_all_progressions_flag, progression : A boolean flag, A dict containing all the info regarding the progressions
    """
    try:
        # contain all the progression where key is the start pct, and
        # value is a tuple (till pct, file name)
        progression = get_all_progressions_saved(outdir=outdir)

        if len(progression) == 0:
            return False, None
        
        # quick check if all the progression exist 
        next_start = 0
        for start, (till, _) in progression.items():
            if next_start != start:
                return False, None
            next_start = till
        if next_start != 100:
            return False, None
        
        return True, progression
    except Exception as exc:
        raise Exception(
            f"~~~~~~~~~~~~~~~~~~~ Error when checking all the progression csv files exist.\n{exc}"
        )
        

def clean_merge_neurons_coords_csv_files(outdir: str, merged_filename: str):
    """
    This function is responsible for dropping all the null rows and merging all the detected neurons coords progression csv files
    into one csv file.

    Parameters:
        outdir          : The path where all the detected neurons coords progression csv files are stored
        merged_filename : The filename to save the clean and mergerd neurons coords.
    """
    try:
        # perform check if all the right progressions saved exist and
        # also get the progressions dict where key is the start pct and value is till pct
        check_all_progressions_flag, progression = check_all_progressions_csv_files_exist(outdir=outdir)

        if not check_all_progressions_flag:
            raise Exception(
                f"~~~~~~~~~~~~~~~~~~ Error: Not all 0 to 100 progression saved csv file found."
            )
        
        # to determine if to add header to the final merge file
        header = True
        # for every progression
        for _, progression_csv_filename in progression.values():
            print(f"~~~~~~~~~~~~~~~~~~~~ Appending neurons coords found in the {progression_csv_filename} to 'neurons_coords.csv'")
            # read the progression csv file
            path_to_progression_csv = os.path.join(outdir, progression_csv_filename)
            detected_neurons_coords = pd.read_csv(path_to_progression_csv)
            # drop the null rows
            detected_neurons_coords = detected_neurons_coords.dropna()

            # append the data to the final merge csv file
            detected_neurons_coords.to_csv(
                os.path.join(outdir, merged_filename),
                mode='a',
                header=header,
                index=False
            )

            # remove the progression csv file
            os.remove(path_to_progression_csv)
            # set the header to false if true
            if header:
                header = False
    except Exception as exc:
        raise Exception(
            f"~~~~~~~~~~~~~~~~~~ Error when clearning and merging all the detected neurons progressions csv files.\n{exc}"
        )