import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import location_map as lm
import pandas as pd
from ecmwfapi import ECMWFService
from tqdm import tqdm


def silent_log(msg): 
    pass 

class ECMWFRetriever:
    """Manage the retrieval of weather data from the ECMWF (European Centre for Medium-Range Weather Forecasts) service.

    Attributes:
        sv_dir (Path): Directory where the retrieved files will be saved.
        info_sv_dir (Path): Parent directory of `sv_dir`, used for storing progress information.
        state_file (Path): Path to the JSON file that tracks download progress.
        logger (logging.Logger): Logger instance for logging messages and progress.
        process_queue (List[Dict]): Queue of files to be downloaded, containing date, time, and filename information.
        progress (Dict): Dictionary tracking the download progress, including session start time, total files, success count, failed files, and last updated timestamp.
        request_params (Dict): Parameters for the ECMWF data request, including class, date, time, experiment version, parameters, steps, target, stream, type, level type, and area.
        server_object (ECMWFService): Instance of the ECMWFService used to execute data requests.

    Methods:
        __init__(sv_dir: str, date_range: pd.DatetimeIndex, area: str, additional_settings: Optional[Dict] = None):
            Initializes the ECMWFRetriever instance, sets up directories, logger, and request parameters, and initializes progress tracking.
        _initialize_progress():
            Initializes or resumes download progress from a previous session, saving progress to a JSON file.
        _save_progress():
            Saves the current progress to the state file in JSON format.
        _setup_logger() -> logging.Logger:
            Sets up and returns a logger instance for logging messages and progress.
        _set_queue(date_range: Iterable[datetime]) -> List[Dict]:
            Creates a queue of files to be downloaded based on the provided date range, excluding already existing files.
        request() -> None:
            Executes the download process for files in the queue, handling retries, interruptions, and progress updates.
    """
    
    def __init__(self, sv_dir:str, date_range:pd.DatetimeIndex, area:str, additional_settings: Optional[Dict]=None): 
        self.sv_dir = Path(sv_dir) 
        self.sv_dir.mkdir(parents=True, exist_ok=True)
        
        self.info_sv_dir = self.sv_dir.parent 
        self.state_file = self.info_sv_dir / "download_progress.json" 
        self.logger = self._setup_logger() 
        self.process_queue = self._set_queue(date_range) 
        self._initialize_progress()
        self._cleanup_failed_files()
        
        self.request_params = {
            "class": "od", 
            "date": None, 
            "time": None, 
            "expver": 1, 
            "params": WSPD_PARAMS, 
            "steps": "/".join(map(str, range(55))), 
            "target": "output", 
            "stream": "enfo",
            "type": "cf",
            "levtype": "sfc",
            "area": area
        }
        self.server_object = ECMWFService(
            "mars", 
            quiet=True, 
            verbose=False,
            log=silent_log
            ) 

        # request params reset 
        if additional_settings is not None: 
            self.request_params.update(additional_settings) 
    
    def _initialize_progress(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.progress = json.load(f)
            
            self.logger.info("Resuming from previous session")
            self.logger.info(f"Total files: {self.progress['total_files']}")
            self.logger.info(f"Successfully downloaded: {self.progress['success_count']}")
            self.logger.info(f"Failed files: {len(self.progress['failed_files'])}")
        else:
            self.progress = {
                "session_start": datetime.now().isoformat(),
                "total_files": len(self.process_queue),
                "success_count": 0,
                "failed_files": [],
                "last_updated": datetime.now().isoformat()
            }
            self._save_progress()
            self.logger.info(f"Starting new download session with {self.progress['total_files']} files")
    
    def _save_progress(self): 
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ECMWFRetriever_{self.info_sv_dir.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.sv_dir.parent / "ECMWF_retriever.log"
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _set_queue(self, date_range: Iterable[datetime]): 
        existing_files = set(f.name for f in self.sv_dir.glob("*.gb2")) 
        process_queue = [] 
        
        for dt in date_range: 
            file_name = f"{dt.strftime('%Y-%m-%d_%H')}.gb2"
            if file_name not in existing_files: 
                process_queue.append({
                    'date': dt.strftime("%Y-%m-%d"),
                    'time': dt.strftime("%H:00:00"),
                    'filename': file_name
                })
        
        return process_queue
    
    def request(self) -> None: 
        if not self.process_queue:
            print("All files already retrieved.")
            return
        
        completed = 0
        max_retries = 3
        
        initial_progress = self.progress["success_count"] 
        total_files = self.progress["total_files"]
        
        with tqdm(
            total=total_files, 
            initial=initial_progress,
            desc="Downloading",
            unit="files"
        ) as pbar:
            
            while completed < len(self.process_queue):
                file_info = self.process_queue[completed]
                target_file = self.sv_dir / file_info['filename']
                
                # 파일이 이미 존재하고 크기가 0보다 크면 스킵
                if target_file.exists() and target_file.stat().st_size > 0:
                    self.logger.info(f"File {file_info['filename']} already exists, skipping")
                    completed += 1
                    continue
                
                retry_count = 0
                success = False 
                
                while retry_count < max_retries and not success:
                    try:
                        # 기존 불완전한 파일 삭제
                        if target_file.exists():
                            target_file.unlink()
                        
                        # request parameter setting 
                        request_params = self.request_params.copy()
                        request_params["date"] = file_info['date']
                        request_params["time"] = file_info['time']
                        
                        self.server_object.execute(request_params, str(target_file))
                        
                        # 다운로드 후 파일 크기 확인
                        if target_file.exists() and target_file.stat().st_size > 0:
                            success = True
                            self.progress['success_count'] += 1
                            
                            # failed_files에서 제거 (재시도 성공한 경우)
                            if file_info['filename'] in self.progress['failed_files']:
                                self.progress['failed_files'].remove(file_info['filename'])
                                
                            self._save_progress()
                            self.logger.info(f"Successfully downloaded {file_info['filename']}")
                        else:
                            raise Exception("Downloaded file is empty or corrupted")
                            
                    except KeyboardInterrupt:
                        self.logger.info("Download interrupted by user")
                        print("\nDownload interrupted. Progress saved. Resume with the same command.")
                        print(f"Progress: {self.progress['success_count']}/{total_files} files completed")
                        return
                        
                    except Exception as e:
                        retry_count += 1
                        
                        # 실패한 파일 삭제
                        if target_file.exists():
                            target_file.unlink()
                        
                        if retry_count < max_retries:
                            wait_time = retry_count * 5  
                            self.logger.warning(f"Retry {retry_count}/{max_retries} for {file_info['filename']}: {e}")
                            time.sleep(wait_time)
                        else:
                            error_msg = f"Failed {file_info['filename']} after {max_retries} attempts: {e}"
                            self.logger.error(error_msg)
                            
                            # 마지막 재시도까지 실패한 경우에만 failed_files에 추가
                            if file_info['filename'] not in self.progress['failed_files']:
                                self.progress['failed_files'].append(file_info['filename'])
                            self._save_progress()
                
                completed += 1
                
                if success:
                    time.sleep(1) 
                    pbar.update(1)

    def _cleanup_failed_files(self):
        """실제로 다운로드된 파일들을 failed_files에서 제거"""
        existing_files = set(f.name for f in self.sv_dir.glob("*.gb2") if f.stat().st_size > 0)
        
        original_failed = self.progress['failed_files'].copy()
        self.progress['failed_files'] = [
            filename for filename in self.progress['failed_files'] 
            if filename not in existing_files
        ]
        
        removed_count = len(original_failed) - len(self.progress['failed_files'])
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} files from failed_files list (actually downloaded)")
            self._save_progress()



SFC_LEVEL_PARAMS = "59.128/134.128/151.128/164.128/165.128/166.128/167.128/228.128/235.128/168.128/146.128/176.128/177.128"
PRESSURE_LEVEL_PARAMS = "129.128/130.128/131/132/133.128"
WSPD_PARAMS = "165.128/166.128/239.228/240.228/246.228/247.228" 
PROJECT_ROOT = Path(__file__).resolve().parents[3]
metadata_root = PROJECT_ROOT / "data" / "meta"


if __name__ == "__main__": 
    location = "dongbok" 
    sv_dir = str(PROJECT_ROOT / "data" / "original" / location / "ECMWF")
    
    with open(metadata_root / "preprocess" / "turbine_coordinate_information.json", "r") as f: 
        coordinate_list = json.load(f)[location] 
    
    with open(metadata_root / "request" / "request_time_range_ECMWF.json", "r") as f: 
        metadata_date_range = json.load(f)[location] 
    
    #date_range = pd.date_range(start=metadata_date_range[0], end=metadata_date_range[1], freq="6h")
    date_range = pd.date_range("2026-03-14 18:00:00", "2024-01-01", freq="-6h")
    area = "/".join(map(str, lm.get_request_area(coordinate_list)))
    
    retriever = ECMWFRetriever(sv_dir, date_range, area) 
    retriever.request()
    