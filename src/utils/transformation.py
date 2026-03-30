import numpy as np 
import pandas as pd 
from sklearn.preprocessing import PowerTransformer 
from typing import Union, List, Optional, Tuple, Literal


class BoxCoxTransformer:
    """
    Box-Cox and Yeo-Johnson power transformations for normalizing data.
    
    This class provides a convenient wrapper around scikit-learn's PowerTransformer
    with enhanced support for pandas DataFrames and Series. It applies power
    transformations to make data more Gaussian-like, which can improve the
    performance of many machine learning algorithms.
    
    The Box-Cox transformation is defined as:
        y(λ) = (y^λ - 1) / λ    if λ ≠ 0
        y(λ) = ln(y)            if λ = 0
        
    The Yeo-Johnson transformation extends Box-Cox to handle zero and negative values:
        y(λ) = ((y + 1)^λ - 1) / λ           if λ ≠ 0, y ≥ 0
        y(λ) = ln(y + 1)                     if λ = 0, y ≥ 0
        y(λ) = -((-y + 1)^(2-λ) - 1) / (2-λ) if λ ≠ 2, y < 0
        y(λ) = -ln(-y + 1)                   if λ = 2, y < 0
    
    Parameters:
        method : {'box-cox', 'yeo-johnson'}, default='yeo-johnson'
            Transformation method to apply:
            - 'box-cox': Requires strictly positive data. More commonly used.
            - 'yeo-johnson': Works with any real-valued data including zeros and negatives.
        standardize : bool, default=False
            Whether to standardize the transformed data to have zero mean and unit variance.
            If True, applies StandardScaler after power transformation.
        
    Attributes:
        lambdas_ : ndarray of shape (n_features,)
            The fitted lambda parameters for each feature. Available after fitting.
        feature_names_ : list of str
            Names of features seen during fit. Used for maintaining DataFrame column names.
        is_fitted : bool
            Whether the transformer has been fitted.
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from transformation.box_cox import BoxCoxTransformer
        
        Basic usage with pandas Series:
        
        >>> # Create right-skewed data
        >>> np.random.seed(42)
        >>> data = pd.Series(np.random.exponential(2, 1000), name='wind_power')
        >>> print(f"Original skewness: {data.skew():.3f}")
        
        >>> # Apply Box-Cox transformation
        >>> transformer = BoxCoxTransformer(method='box-cox')
        >>> transformed = transformer.fit_transform(data)
        >>> print(f"Transformed skewness: {transformed.skew():.3f}")
        >>> print(f"Lambda parameter: {transformer.lambdas_[0]:.3f}")
        
        >>> # Perfect inverse transformation
        >>> recovered = transformer.inverse_transform(transformed)
        >>> print(f"Recovery error: {np.abs(data - recovered).max():.10f}")
        
        DataFrame usage with multiple columns:
        
        >>> # Create DataFrame with different distributions
        >>> df = pd.DataFrame({
        ...     'exponential': np.random.exponential(2, 500),
        ...     'lognormal': np.random.lognormal(0, 1, 500)
        ... })
        
        >>> # Transform all columns
        >>> transformer = BoxCoxTransformer(method='box-cox', standardize=True)
        >>> df_transformed = transformer.fit_transform(df)
        >>> print(f"Lambdas: {transformer.lambdas_}")
        >>> print(f"Transformed means: {df_transformed.mean()}")  # Should be ~0
        >>> print(f"Transformed stds: {df_transformed.std()}")   # Should be ~1
    
    Notes:
        - Box-Cox transformation requires strictly positive data (> 0)
        - Yeo-Johnson transformation works with any real-valued data
        - Lambda parameters are estimated using maximum likelihood estimation
        - When standardize=True, inverse_transform automatically reverses both
        standardization and power transformation
        - Always fit the transformer on training data only, then apply to test data
        - The transformation preserves the original data type (Series→Series, DataFrame→DataFrame)
    
    References:
        - Box, G. E. P. and Cox, D. R. (1964). "An Analysis of Transformations".
           Journal of the Royal Statistical Society, Series B, 26, 211-252.
        - Yeo, I. K. and Johnson, R. A. (2000). "A new family of power transformations
           to improve normality or symmetry". Biometrika, 87, 954-959.
           
    See Also
    --------
    sklearn.preprocessing.PowerTransformer : The underlying scikit-learn transformer
    sklearn.preprocessing.StandardScaler : Used when standardize=True
    """

    def __init__(self, method: Literal['box-cox', 'yeo-johnson'] = 'yeo-johnson', standardize=False):
        """
        Initialize Box-Cox transformer.
        
        Parameters:
            method : str, default='yeo-johnson'
                Transformation method. Options: 'box-cox', 'yeo-johnson'
            standardize : bool, default=False
                Whether to standardize the transformed data to have zero mean 
                and unit variance.
        """
        self.method = method
        self.standardize = standardize
        self.transformer = PowerTransformer(method=method, standardize=standardize)
        self.is_fitted = False
        self.feature_names_ = None

    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> 'BoxCoxTransformer':
        """
        Fit the Box-Cox transformation.
        
        Parameters:
            data : pd.DataFrame, pd.Series, or np.ndarray
                Input data to fit the transformation.
            
        Returns:
            self : BoxCoxTransformer
        """
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            self.feature_names_ = data.columns.tolist()
            X = data.to_numpy()
        elif isinstance(data, pd.Series):
            self.feature_names_ = [data.name] if data.name else ['feature_0']
            X = data.to_numpy().reshape(-1, 1)
        else:
            X = np.asarray(data)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        # Check for positive values (Box-Cox requirement)
        if self.method == 'box-cox' and np.any(X <= 0):
            raise ValueError("Box-Cox transformation requires positive values. "
                            "Consider using method='yeo-johnson' for data with zeros/negatives.")

        # Fit the transformer
        self.transformer.fit(X)
        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, 
np.ndarray]:
        """
        Apply the Box-Cox transformation.
        
        Parameters:
            data : pd.DataFrame, pd.Series, or np.ndarray
                Input data to transform.
            
        Returns:
            transformed : same type as input
                Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        # Handle different input types
        if isinstance(data, pd.DataFrame):
            X = data.to_numpy() 
            X_transformed = self.transformer.transform(X)
            return pd.DataFrame(X_transformed, index=data.index, columns=data.columns)

        elif isinstance(data, pd.Series):
            X = data.to_numpy().reshape(-1, 1)
            X_transformed = self.transformer.transform(X)
            return pd.Series(X_transformed.flatten(), index=data.index, name=data.name)

        else:
            X = np.asarray(data)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                return self.transformer.transform(X).flatten()
            else:
                return self.transformer.transform(X)

    def fit_transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, 
np.ndarray]:
        """Fit the transformer and transform data in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, 
pd.Series, np.ndarray]:
        """
        Apply the inverse Box-Cox transformation.
        
        Parameters:
            data : pd.DataFrame, pd.Series, or np.ndarray
                Transformed data to inverse transform.
            
        Returns:
            original : same type as input
                Data in original scale.
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse transform")

        # Handle different input types
        if isinstance(data, pd.DataFrame):
            X = data.to_numpy() 
            X_original = self.transformer.inverse_transform(X)
            return pd.DataFrame(X_original, index=data.index, columns=data.columns)

        elif isinstance(data, pd.Series):
            X = data.to_numpy().reshape(-1, 1)
            X_original = self.transformer.inverse_transform(X)
            return pd.Series(X_original.flatten(), index=data.index, name=data.name)

        else:
            X = np.asarray(data)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                return self.transformer.inverse_transform(X).flatten()
            else:
                return self.transformer.inverse_transform(X)

    @property
    def lambdas_(self) -> np.ndarray:
        """Get the fitted lambda parameters."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted to access lambdas_")
        return self.transformer.lambdas_


# Pilot code 
if __name__ == "__main__":
    df = pd.DataFrame({
    'col1': np.random.normal(loc=3, scale=4, size=100),
    'col2': np.random.normal(loc=0, scale=4, size=100)
    })

    transformer_df = BoxCoxTransformer()
    transformed_df = transformer_df.fit_transform(df)
    recovered_df = transformer_df.inverse_transform(transformed_df)

    error_df = np.abs(df - recovered_df).max().max()

    print(f"   DataFrame test:")
    print(f"   Original shape: {df.shape}")
    print(f"   Transformed shape: {transformed_df.shape}")
    print(f"   Recovery error: {error_df:.10f}")
    print(f"   Lambdas: {transformer_df.lambdas_}")