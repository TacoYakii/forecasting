import requests
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, List, TypedDict
import logging
from threading import Thread
from dotenv import load_dotenv


load_dotenv() # load .env file


class DownloadProgress(TypedDict):
    session_start: str
    total_files: int
    success_count: int
    failed_files: List[str]
    daily_download_count: int
    last_reset_date: str
    last_updated: str

class KMARetriever:
    """
    Manage the retrieval of weather forecast data from the Korea Meteorological Administration (KMA) service.

    Attributes:
        sv_dir (Path): Directory where the retrieved files will be saved.
        state_file (Path): Path to the JSON file that tracks download progress.
        logger (logging.Logger): Logger instance for logging messages and progress.
        process_queue (List[Dict]): Queue of files to be downloaded, containing time and filename information.
        progress (Dict): Dictionary tracking the download progress, including session start time, total files, success count, failed files, and last updated timestamp.
        request_params (Dict): Parameters for the KMA data request, including group, nwp, data, varn, level, lat, lon, and any additional settings.
        api_url (str): KMA API endpoint URL.
        auth_keys (List[str]): List of API authentication keys.
        current_auth_index (int): Index of the currently used API key.
        daily_download_count (int): Number of downloads performed today.
        last_reset_date (date): Date when the daily download count was last reset.
        chunk_size (int): Number of files to download concurrently in each thread batch.

    Methods:
        __init__(sv_dir: str, lat: float, lon: float, date_range: pd.DatetimeIndex, dataset_type: str = "LDAPS", chunk_size: int = 3, additional_settings: Optional[Dict] = None):
            Initializes the KMARetriever instance, sets up directories, logger, request parameters, and initializes progress tracking.
        _setup_logger() -> logging.Logger:
            Sets up and returns a logger instance for logging messages and progress.
        _set_queue(date_range: pd.DatetimeIndex) -> List[Dict]:
            Creates a queue of files to be downloaded based on the provided date range, excluding already existing files.
        _initialize_progress():
            Initializes or resumes download progress from a previous session, saving progress to a JSON file.
        _save_progress():
            Saves the current progress to the state file in JSON format.
        _manage_auth_key():
            Manages API key usage, switches keys, or waits if daily limits are reached.
        _request_single_file(tmfc: str, filename: str, progress_bar):
            Downloads a single file, with retries and error handling.
        _process_file_wrapper(file_info: Dict, progress_bar):
            Thread-safe wrapper for downloading a file and updating progress.
        request():
            Executes the download process for files in the queue, handling retries, 
            
    Note:
        - This script uses the `dotenv` approach for API key management. 
        - You must create a `.env` file in your execution directory and add your KMA API key as `KMA_PRIMARY_KEY`.
        - (Optional) To increase your daily download limit, you can add a secondary key as `KMA_SECONDARY_KEY` in the `.env` file.
        - You can obtain your API key(s) from the official KMA API website.
    """

    def __init__(
        self, 
        sv_dir: str, 
        lat: float, 
        lon: float,
        date_range: pd.DatetimeIndex,
        dataset_type: str = "LDAPS",
        chunk_size: int = 3,
        additional_settings: Optional[Dict] = None
    ):

        self.sv_dir = Path(sv_dir) 
        self.sv_dir.mkdir(parents=True, exist_ok=True)
        self.info_sv_dir = self.sv_dir.parent
        self.lat = lat
        self.lon = lon
        self.dataset_type = dataset_type
        self.chunk_size = chunk_size
        self.state_file = self.info_sv_dir / "kma_download_progress.json"
        
        self.api_url = "https://apihub.kma.go.kr/api/typ06/url/um_grib_pt_tmef.php"
        primary_key = "***REMOVED***"
        secondary_key = "***REMOVED***"
        
        if not primary_key:
            raise ValueError(
                "KMA_PRIMARY_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        
        self.auth_keys = [primary_key]
        if secondary_key:
            self.auth_keys.append(secondary_key)
        
        self.request_params = {
            "group": "UMKR" if dataset_type == "LDAPS" else "UMGL", 
            "nwp": "N512",
            "data": "P",
            "varn": "2009,2002,2003,3005,0,1001,1194",
            "level": "800,850,875,900,925,950,975,1000",
            "lat": str(lat),
            "lon": str(lon)
        }
        
        # request params reset 
        if additional_settings:
            self.request_params.update(additional_settings)
        
        self.current_auth_index = 0
        self.daily_download_count = 0
        self.last_reset_date = datetime.now().date()
        
        self.logger = self._setup_logger()

        self.process_queue = self._set_queue(date_range)
        self._initialize_progress()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("KMARetriever")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.info_sv_dir / "KMA_retriever.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
        
        return logger
    
    def _set_queue(self, date_range: pd.DatetimeIndex) -> List[Dict]:
        existing_files = set(f.name for f in self.sv_dir.glob("*.txt"))
        process_queue = []
        
        for dt in date_range:
            file_name = f"{dt.strftime('%Y%m%d%H')}.txt"
            if file_name not in existing_files:
                process_queue.append({
                    'tmfc': dt.strftime('%Y%m%d%H'),
                    'filename': file_name
                })
        
        self.already_downloaded = len(existing_files)
        self.logger.info(f"Found {self.already_downloaded} existing files")
        self.logger.info(f"{len(process_queue)} files to download")
        print(f"Already downloaded: {self.already_downloaded} files / Remaining: {len(process_queue)} files")
        print(f"Already downloaded: {self.already_downloaded} files / Remaining: {len(process_queue)} files")

        return process_queue
    
    def _initialize_progress(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.progress: DownloadProgress = json.load(f)
            
            self.daily_download_count = self.progress.get('daily_download_count', 0)
            last_date_str = self.progress.get('last_reset_date', datetime.now().date().isoformat()) # try to get last_reset_date if None -> get current date 
            self.last_reset_date = datetime.fromisoformat(last_date_str).date()
            
            self.logger.info("Resuming from previous session")
            self.logger.info(f"Total files: {self.progress['total_files']}")
            self.logger.info(f"Successfully downloaded: {self.progress['success_count']}")
            self.logger.info(f"Failed files: {len(self.progress['failed_files'])}")
            self.logger.info(f"Daily download count: {self.daily_download_count}")
        else:
            self.progress: DownloadProgress = {
                "session_start": datetime.now().isoformat(),
                "total_files": len(self.process_queue),
                "success_count": 0,
                "failed_files": [],
                "daily_download_count": 0,
                "last_reset_date": datetime.now().date().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._save_progress()
            self.logger.info(f"Starting new download session with {self.progress['total_files']} files")
    
    def _save_progress(self):
        self.progress["last_updated"] = datetime.now().isoformat()
        self.progress["daily_download_count"] = self.daily_download_count
        self.progress["last_reset_date"] = self.last_reset_date.isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _manage_auth_key(self):
        """
        Manages the authentication key rotation and daily download limits for API requests.
        This method performs the following actions:
        - Resets the daily download counter and switches to the primary authentication key if the date has changed.
        - Switches to the next available authentication key if the daily download limit for the current key is reached.
        - If all authentication keys are exhausted for the day, waits until the next day before resuming downloads.
        - Logs and prints relevant information about key changes and waiting periods.
        Side Effects:
            - Modifies `self.daily_download_count`, `self.current_auth_index`, and `self.last_reset_date`.
            - May cause the process to sleep until the next day if all keys are exhausted.
            - Logs and prints status messages regarding key management.
        """
        
        now = datetime.now()

        # Date change -> reset counter 
        if now.date() > self.last_reset_date: # last reset date = download start date -> if day changes reset to current date 
            self.daily_download_count = 0
            self.current_auth_index = 0
            self.last_reset_date = now.date()
            self.logger.info("Daily download count reset. Changed to primary auth key.")
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Changed authkey to original")

        # Exceed daily limit -> change to next key 
        if self.daily_download_count >= 19950:
            if self.current_auth_index < len(self.auth_keys) - 1:
                old_index = self.current_auth_index
                self.current_auth_index += 1
                self.daily_download_count = 0
                self.logger.info(f"API key changed due to daily limit. Index {old_index} -> {self.current_auth_index}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] API key switched due to daily limit")
            else:
                # Used every key -> Wait until next day to resume download 
                next_day = (self.last_reset_date + pd.Timedelta(days=1))
                wait_seconds = (datetime.combine(next_day, datetime.min.time()) - now).total_seconds()
                wait_hours = wait_seconds / 3600
                msg = (
                    f"All API keys exhausted for today. "
                    f"Waiting until {next_day.strftime('%Y-%m-%d')} ({wait_hours:.2f} hours)..."
                )
                self.logger.warning(msg)
                print(msg)
                time.sleep(max(0, wait_seconds))

                self._manage_auth_key()
    
    def _request_single_file(self, tmfc: str, filename: str, progress_bar: tqdm) -> bool:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._manage_auth_key()
                
                params = self.request_params.copy()
                params["tmfc"] = tmfc
                params["authKey"] = self.auth_keys[self.current_auth_index]
                
                response = requests.get(self.api_url, params=params, timeout=30)
                self.daily_download_count += 1
                
                response.raise_for_status()
            
                target_file = self.sv_dir / filename
                with open(target_file, "wb") as f:
                    f.write(response.content)
                
                time.sleep(0.25)  
                progress_bar.update(1)
                return True
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Error downloading {filename} (attempt {retry_count}/{max_retries}): {e}"
                
                if retry_count < max_retries:
                    self.logger.warning(error_msg)
                    time.sleep(1 * retry_count)  
                else:
                    self.logger.error(f"Failed to download {filename} after {max_retries} attempts: {e}")
        
        return False
    
    def request(self) -> None:
        if not self.process_queue:
            print("All files already retrieved.")
            return
        
        total_files = self.already_downloaded + len(self.process_queue)
        initial_progress = self.already_downloaded + self.progress["success_count"]

        with tqdm(
            total=total_files,
            initial=initial_progress,
            desc="Downloading",
            unit="files"
        ) as pbar:
            
            # 청크 단위로 멀티스레딩 처리
            chunks = [
                self.process_queue[i:i+self.chunk_size] 
                for i in range(0, len(self.process_queue), self.chunk_size)
                ]
            
            for chunk in chunks:
                try:
                    threads = []
                    
                    for file_info in chunk:
                        thread = Thread(
                            target=self._process_file_wrapper,
                            args=(file_info, pbar)
                        )
                        threads.append(thread)
                        thread.start()
                    
                    for thread in threads:
                        thread.join()
                
                except KeyboardInterrupt:
                    self.logger.info("Download interrupted by user")
                    print("\nDownload interrupted. Progress saved.")
                    print(f"Progress: {self.progress['success_count']}/{total_files} files completed")
                    return
        
        final_stats = {
            "completed_at": datetime.now().isoformat(),
            "total_files": self.progress['total_files'],
            "success_count": self.progress['success_count'],
            "failed_count": len(self.progress['failed_files']),
            "failed_files": self.progress['failed_files']
        }
        self.logger.info("Download session completed")
        self.logger.info(f"Success: {final_stats['success_count']}/{final_stats['total_files']}")
        self.logger.info(f"Failed: {final_stats['failed_count']}")

    
    def _process_file_wrapper(self, file_info: Dict, progress_bar: tqdm):
        """파일 처리 래퍼 (스레드 안전성을 위한)"""
        success = self._request_single_file(
            file_info['tmfc'], 
            file_info['filename'],
            progress_bar
        )
        
        if success:
            self.progress['success_count'] += 1
        else:
            self.progress['failed_files'].append(file_info[f'{self.sv_dir.parent.name}/filename'])
        
        self._save_progress()



if __name__ == "__main__":
    import pandas as pd
    import json 

    location = "dongbok"
    mode = "GDAPS"
    
    with open("data/meta/request/request_time_range.json", "r") as f: 
        request_time_range = json.load(f)

    with open("data/meta/request/group_information.json", "r") as f: 
        group_information = json.load(f)
        
    date_range = pd.date_range(request_time_range[location][0], request_time_range[location][1], freq="6h")

    for group, info in group_information[mode][location].items():
        retriever = KMARetriever(
            sv_dir=f"data/original/{location}/{mode}/{group}",
            lat=info["coordinate"][0],  
            lon=info["coordinate"][1], 
            date_range=date_range,
        dataset_type=mode,
        chunk_size=3,
        additional_settings={
            "varn": "2009,2002,2003,3005,0,1001,1194",
            "level": "850,925,950,1000"
        }
    )
    
    retriever.request()