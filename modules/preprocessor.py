import pandas as pd
from typing import Dict, Any, Optional

class DataPreprocessor:
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
    
    def load_data(self, file, file_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        try:
            file.seek(0)
            
            if file_info['type'] == 'csv':
                chunks = []
                for chunk in pd.read_csv(file, 
                                        encoding=file_info['encoding'],
                                        chunksize=self.chunk_size,
                                        low_memory=False):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_excel(file, engine='openpyxl')
            
            return df
            
        except Exception as e:
            raise Exception(f"파일 로드 오류: {str(e)}")