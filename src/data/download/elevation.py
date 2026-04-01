import requests
from dotenv import load_dotenv
import os
import time
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv() 
api_key = os.getenv("ELEVATION") 


def get_elevation(lat, lng):
    """
    Get elevation of given coordinates. 

    Args:
        lat (float|str): lattitude of given location
        lng (float|str): longitude of given location

    Raises:
        ValueError: If status != 200

    Returns:
        float: unit = meter 
    """
    url = f"https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lng}",
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    time.sleep(0.1)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "OK" and "results" in data:
            return data["results"][0]["elevation"]
        else:
            raise ValueError(f"API Error: {data.get('status')}")
    else:
        response.raise_for_status()
        

with open(PROJECT_ROOT / "data" / "meta" / "minmax_height_in_meter.json", "r") as f:
    elevation_info = json.load(f)

with open(PROJECT_ROOT / "data" / "meta" / "turbine_coordinate_information.json", "r") as f:
    coordinate_info = json.load(f)

TURBINE_SPEC_PATH = PROJECT_ROOT / "data" / "meta" / "turbine_spec.xlsx"
excel_file = pd.ExcelFile(TURBINE_SPEC_PATH) 
sheet_names = excel_file.sheet_names 

TURBINE_SPEC = {}
for sheet_name in sheet_names: 
    df = excel_file.parse(sheet_name).T
    df.columns = df.iloc[0] 
    df = df[1:]
    TURBINE_SPEC[sheet_name] = df

    
if __name__ == "__main__": 
    changed = False
    for farm_nm in coordinate_info.keys(): 
        if farm_nm in list(elevation_info.keys()): 
            continue 
        else: 
            try: 
                elevation_info[farm_nm] = {}
                new_farm_elevation = {} 
                for turbine_no, coordinate in coordinate_info[farm_nm].items(): 
                    new_farm_elevation[turbine_no] = {
                        get_elevation(coordinate[0], coordinate[1])
                    }
                
                farm_spec = TURBINE_SPEC[farm_nm] 
                for turbine_i in new_farm_elevation.keys(): 
                    turbine_height = farm_spec.loc[int(turbine_i)]["height"] 
                    rotor_diameter = farm_spec.loc[int(turbine_i)]["rotor_diameter"] 
                    elevation = new_farm_elevation[turbine_i] 
                    
                    maximum_height = elevation + turbine_height + (rotor_diameter/2) 
                    minimum_height = elevation + turbine_height - (rotor_diameter/2) 
                    
                    elevation_info[farm_nm].update({
                        turbine_i: {
                            "max": maximum_height,
                            "min": minimum_height, 
                            "hub": (maximum_height + minimum_height) * 0.5 
                        }
                    })
                changed = True 
            except KeyError as e: 
                print(f"There is no turbine spec information: {e}")
    
    if changed: 
        with open(PROJECT_ROOT / "data" / "meta" / "minmax_height_in_meter.json", "w") as f:
            json.dump(elevation_info, f) 
    

                