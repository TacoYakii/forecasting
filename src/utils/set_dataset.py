import pandas as pd
from typing import Tuple, Union, Optional, Iterable


def _sort_dataset_by_index(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataset by index, handling different index types (datetime, string, int).
    
    Args:
        dataset: Input DataFrame to sort
        
    Returns:
        pd.DataFrame: Sorted DataFrame
    """
    try:
        # Try direct sort_index first (works for most cases)
        return dataset.sort_index()
    except Exception:
        if dataset.index.dtype == 'object':
            # Try to convert to datetime first
            datetime_index = pd.to_datetime(dataset.index, errors='coerce')
            if not datetime_index.isna().all():
                # If conversion successful, create new DataFrame with datetime index
                sorted_dataset = dataset.copy()
                sorted_dataset.index = datetime_index
                return sorted_dataset.sort_index()
            else:
                # If datetime conversion fails, sort as strings
                return dataset.sort_index()
        else:
            # Fallback to regular sort
            return dataset.sort_index()

            
def prepare_dataset(
    dataset: pd.DataFrame, 
    y_col: Union[str, int],
    exog_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
    training_period: Optional[Tuple[Union[str, pd.Timestamp, int], Union[str, pd.Timestamp, int]]] = None, 
    forecast_period: Optional[Tuple[Union[str, pd.Timestamp, int], Union[str, pd.Timestamp, int]]] = None, 
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Index]:
    """
    Prepare training dataset by splitting features and target.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Index, pd.Index]: 
            (X_train, y_train, X_forecast, y_forecast, base_idx, forecast_idx)
    """
    # Sort dataset by index to ensure chronological order
    dataset = _sort_dataset_by_index(dataset)
    
    # Set default periods if None
    if training_period is None:
        training_period = (dataset.index[0], dataset.index[int(len(dataset) * 0.7)])
    if forecast_period is None:
        forecast_period = (dataset.index[int(len(dataset) * 0.7)], dataset.index[-1])
        
    # Get training data slice
    train_data = dataset.loc[training_period[0]:training_period[1]]
    # Get forecast data slice
    forecast_slice = dataset.loc[forecast_period[0]:forecast_period[1]]
    if len(forecast_slice) > 0:
        forecast_data = forecast_slice
    else:
        # Edge case: if there's only one row or empty, create empty DataFrame with same structure
        forecast_data = pd.DataFrame(columns=dataset.columns, index=pd.Index([], dtype=dataset.index.dtype))
    
    # Handle exog_cols            
    if exog_cols is None:
        # Use all columns except y_col
        x_columns = [col for col in train_data.columns if col != y_col]
    else:
        # Convert single value to list, keep iterables as-is
        if isinstance(exog_cols, (str, int)):
            x_columns = [exog_cols]
        else:
            x_columns = list(exog_cols)
    
    # Extract features and target
    train_X = train_data[x_columns]
    train_y = train_data[y_col]
    
    forecast_X = forecast_data[x_columns] 
    forecast_y = forecast_data[y_col] 
    
    # Extract indices for the prediction period
    base_idx = forecast_y.index 

    
    return train_X, train_y, forecast_X, forecast_y, base_idx