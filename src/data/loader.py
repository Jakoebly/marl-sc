import pandas as pd
from pathlib import Path


def load_empirical_data(path: str) -> pd.DataFrame:
    """
    Load empirical demand data from CSV file.
    
    Expected CSV format:
    - Columns: timestep, region, order_id, sku_id, quantity
    - timestep: int - timestep identifier
    - region: int - region identifier
    - order_id: int - order identifier
    - sku_id: int - SKU identifier
    - quantity: float - demand quantity
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with validated structure
        
    Raises:
        ValueError: If file doesn't exist or has invalid structure
    """
    path_obj = Path(path)
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_columns = ['timestep', 'region', 'order_id', 'sku_id', 'quantity']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate data types
    if not pd.api.types.is_integer_dtype(df['timestep']):
        raise ValueError("Column 'timestep' must be integer type")
    if not pd.api.types.is_integer_dtype(df['region']):
        raise ValueError("Column 'region' must be integer type")
    if not pd.api.types.is_integer_dtype(df['order_id']):
        raise ValueError("Column 'order_id' must be integer type")
    if not pd.api.types.is_integer_dtype(df['sku_id']):
        raise ValueError("Column 'sku_id' must be integer type")
    if not pd.api.types.is_numeric_dtype(df['quantity']):
        raise ValueError("Column 'quantity' must be numeric type")
    
    # Validate non-negative quantities
    if (df['quantity'] < 0).any():
        raise ValueError("Column 'quantity' must contain non-negative values")
    
    # Sort by timestep for efficient querying
    df = df.sort_values(['timestep', 'region', 'order_id', 'sku_id']).reset_index(drop=True)
    
    return df

