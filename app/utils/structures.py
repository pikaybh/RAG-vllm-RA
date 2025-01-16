from typing import Dict, List, Sequence, Union
import pandas as pd


__all__ = ["BYPandas", "get_unique"]



class BYPandas:
    @staticmethod
    def read(path: str, **kwargs) -> pd.DataFrame:
        """
        Reads a file into a pandas DataFrame, supporting various file extensions.

        Args:
            path (str): The path to the file.
            **kwargs: Additional arguments to pass to the pandas read function.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            ValueError: If the file extension is not supported.
        """
        if path.endswith(".csv"):
            return pd.read_csv(path, **kwargs)
        elif path.endswith(".xlsx"):
            return pd.read_excel(path, **kwargs)
        elif path.endswith(".json"):
            return pd.read_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension for file: {path}")


def get_unique(path: str, *args: Union[Sequence[str], str], **kwargs) -> Union[Dict[str, List[str]], List[str]]:
    """
    Extracts unique non-null items from specified columns in a file.

    Args:
        path (str): Path to the file to read.
        *args (Union[Sequence[str], str]): Columns to extract unique values from.
        **kwargs: Additional arguments for reading the file.

    Returns:
        Union[Dict[str, List[str]], List[str]]: Unique values from the specified column(s).

    Raises:
        ValueError: If no column names are provided.
    """

    if not args:
        raise ValueError("At least one column name must be provided.")

    df = BYPandas.read(path, **kwargs)

    def helper(column: str) -> List[str]:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the file.")
        return df[column].dropna().unique().tolist()

    if isinstance(args[0], str) and len(args) == 1:
        return helper(args[0])
    else:
        return {arg: helper(arg) for arg in args}

if __name__ == "__main__":
    # Example usage
    path = r"D:\문서\D__1_Projects\202501\vessl_llama_ra\app\assets\KRAS_찬영수정_250114.csv"
    unique_items = get_unique(path, "공종", "공정", encoding="utf-8")
    print(unique_items)