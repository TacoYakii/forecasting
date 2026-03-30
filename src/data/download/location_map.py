import folium 
from pathlib import Path 
from typing import List, Dict, Optional


def get_area(coordinate_list:Dict[str, List[float]]) -> List[float]: 
    lats = [c[0] for c in coordinate_list.values()] 
    lons = [c[1] for c in coordinate_list.values()] 
    
    return [max(lats), min(lons), min(lats), max(lons)] 

def get_request_area(coordinate_list:Dict[str, List[float]], add_value:float=0.2) -> List[float]: 
    max_lat, min_lon, min_lat, max_lon = get_area(coordinate_list) 
    request_max_lat = max_lat + add_value 
    request_min_lon = min_lon - add_value 
    request_min_lat = min_lat - add_value 
    request_max_lon = max_lon + add_value 
    
    return [request_max_lat, request_min_lon, request_min_lat, request_max_lon]

    
class GetLocationMap: 
    """
    A class to generate and visualize a map based on a list of geographical coordinates.
    Attributes:
        coordinate_list (Dict[str, List[float]]): A dictionary where keys are point IDs and values are 
            lists containing latitude and longitude of the points.
        area_range (List[float]): A list containing the geographical bounds of the coordinates in the 
            format [max_latitude, min_longitude, min_latitude, max_longitude].
    Methods:
        _get_area() -> List[float]:
            Computes the geographical bounds of the coordinates.
        _add_coordinate_points(map_obj: folium.Map) -> None:
            Adds circle markers for each coordinate point to the given folium map object.
        draw(sv_dir: str) -> None:
            Generates a folium map with a rectangle representing the area range and markers for each 
            coordinate point, then saves the map as an HTML file in the specified directory.
    """
    
    def __init__(self, coordinate_list:Dict[str, List[float]], request_area_add_value:Optional[float]=None): 
        self.coordinate_list = coordinate_list
        if request_area_add_value: 
            self.area_range = get_request_area(coordinate_list, request_area_add_value)
        else: 
            self.area_range = get_area(self.coordinate_list)

    def _add_coordinate_points(self, map_obj: folium.Map) -> None:
        for point_id, (lat, lon) in self.coordinate_list.items():
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=f"ID: {point_id}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip=f"{point_id}",
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.8
            ).add_to(map_obj)
    
    def draw(self, sv_dir): 
        max_lat, min_lon, min_lat, max_lon = self.area_range 
        
        center_lat = (max_lat + min_lat) / 2
        center_lon = (max_lon + min_lon) / 2 
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        folium.Rectangle(
            bounds=[
                [min_lat, min_lon],  
                [max_lat, max_lon]   
            ],
            color="blue",            
            fill=True,               
            fill_opacity=0.2         
        ).add_to(m)
        self._add_coordinate_points(m) 
        
        sv_path = Path(sv_dir) / f"request_range.html"
        m.save(sv_path)