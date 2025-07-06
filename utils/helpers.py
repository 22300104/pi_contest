import pandas as pd

def format_number(num: int) -> str:
    return f"{num:,}"

def get_memory_usage(df: pd.DataFrame) -> str:
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes < 1024 * 1024:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 * 1024 * 1024:
        return f"{memory_bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{memory_bytes / 1024 / 1024 / 1024:.2f} GB"