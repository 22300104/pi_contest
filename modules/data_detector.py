import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Any, List, Tuple
import chardet
import platform

class DataDetector:
    def __init__(self):
        # 한글 폰트 설정
        self.set_korean_font()
    
    def set_korean_font(self):
        """운영체제별 한글 폰트 설정"""
        system = platform.system()
        
        if system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif system == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        else:
            plt.rcParams['font.family'] = 'NanumGothic'
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def detect_file_info(self, file) -> Dict[str, Any]:
        file.seek(0)
        file_info = {
            'name': file.name,
            'size': file.size,
            'type': 'csv' if file.name.endswith('.csv') else 'excel'
        }
        
        if file_info['type'] == 'csv':
            file.seek(0)
            raw_data = file.read(10000)
            file.seek(0)
            result = chardet.detect(raw_data)
            file_info['encoding'] = result['encoding'] or 'utf-8'
        
        return file_info
    
    def get_appropriate_stats(self, main_type: str, sub_type: str) -> List[str]:
        """데이터 타입에 따른 적절한 통계 옵션 반환"""
        if main_type == "numeric":
            if sub_type == "continuous":
                return ["기술통계", "분위수", "분포특성", "이상치"]
            elif sub_type == "discrete":
                return ["기술통계", "빈도분석", "분포특성"]
            elif sub_type == "binary":
                return ["이진분포", "비율분석"]
        
        elif main_type == "categorical":
            if sub_type == "nominal":
                return ["빈도분석", "상위카테고리", "희소카테고리", "비율분석", "엔트로피"]  # 🔴 추가
            elif sub_type == "ordinal":
                return ["빈도분석", "상위카테고리", "순서통계", "비율분석", "엔트로피"]  # 🔴 추가
            elif sub_type == "binary":
                return ["이진분포", "비율분석", "엔트로피"]  # 🔴 추가
        
        elif main_type == "text":
            if sub_type == "short":
                return ["빈도분석", "텍스트통계", "패턴분석", "엔트로피"]  # 🔴 추가
            elif sub_type == "long":
                return ["텍스트통계", "패턴분석", "단어통계"]
        
        elif main_type == "datetime":
            return ["시간범위", "주기분석", "시간간격통계"]
        
        return []
    
    def get_appropriate_visualizations(self, main_type: str, sub_type: str) -> List[str]:
        """데이터 타입에 따른 적절한 시각화 옵션 반환"""
        if main_type == "numeric":
            if sub_type == "continuous":
                return ["히스토그램", "박스플롯", "밀도플롯", "바이올린플롯", "산점도"]
            elif sub_type == "discrete":
                return ["막대그래프", "히스토그램", "파이차트", "상위N막대그래프", "산점도"]
            elif sub_type == "binary":
                return ["파이차트", "막대그래프", "도넛차트"]
        
        elif main_type == "categorical":
            if sub_type == "nominal" or sub_type == "ordinal":
                return ["막대그래프", "파이차트", "가로막대그래프", "상위N막대그래프", "파레토차트"]
            elif sub_type == "binary":
                return ["파이차트", "막대그래프", "도넛차트"]
        
        elif main_type == "text":
            if sub_type == "short":
                return ["상위N막대그래프", "단어빈도", "가로막대그래프"]
            elif sub_type == "long":
                return ["길이분포", "단어빈도", "상위N막대그래프"]
        
        elif main_type == "datetime":
            return ["시계열그래프", "월별분포", "요일분포", "시간대분포", "산점도"]
        
        return ["막대그래프"]
    
    def calculate_statistics(self, df: pd.DataFrame, column: str, stat_type: str) -> Dict:
        """선택된 통계 계산"""
        col_data = df[column].dropna()
        
        # 기본정보
        if stat_type == "기본정보":
            return {
                '전체 개수': f"{len(df[column]):,}",
                '결측값': f"{df[column].isnull().sum():,}",
                '결측값 비율': f"{df[column].isnull().sum()/len(df[column])*100:.1f}%",
                '고유값': f"{df[column].nunique():,}"
            }
        
        # 기술통계
        elif stat_type == "기술통계":
            return {
                '개수': f"{len(col_data):,}",
                '평균': f"{col_data.mean():.4f}",
                '표준편차': f"{col_data.std():.4f}",
                '최소값': f"{col_data.min():.4f}",
                '최대값': f"{col_data.max():.4f}",
                '중앙값': f"{col_data.median():.4f}"
            }
        
        # 분위수
        elif stat_type == "분위수":
            return {
                '10%': f"{col_data.quantile(0.10):.4f}",
                '25%': f"{col_data.quantile(0.25):.4f}",
                '50%': f"{col_data.quantile(0.50):.4f}",
                '75%': f"{col_data.quantile(0.75):.4f}",
                '90%': f"{col_data.quantile(0.90):.4f}"
            }
        
        # 분포특성
        elif stat_type == "분포특성":
            from scipy import stats
            return {
                '왜도': f"{col_data.skew():.4f}",
                '첨도': f"{col_data.kurtosis():.4f}",
                '변동계수': f"{(col_data.std() / col_data.mean()):.4f}" if col_data.mean() != 0 else "N/A"
            }
        
        # 이상치
        elif stat_type == "이상치":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            return {
                'IQR': f"{IQR:.4f}",
                '하한': f"{Q1 - 1.5 * IQR:.4f}",
                '상한': f"{Q3 + 1.5 * IQR:.4f}",
                '이상치 개수': f"{len(outliers):,}",
                '이상치 비율': f"{len(outliers)/len(col_data)*100:.2f}%"
            }
        
        # 빈도분석
        elif stat_type == "빈도분석":
            value_counts = df[column].value_counts()
            
            # 20개 이상일 때 DataFrame으로 반환하여 자동 스크롤
            if len(value_counts) > 20:
                freq_df = pd.DataFrame({
                    '값': value_counts.index,
                    '빈도': value_counts.values,
                    '비율(%)': (value_counts.values / len(df) * 100).round(2)
                })
                return freq_df  # DataFrame은 자동으로 스크롤 처리됨
            else:
                return value_counts.to_dict()
        

            # 🔴 엔트로피 통계 추가 (빈도분석 다음에 추가)
        elif stat_type == "엔트로피":
            value_counts = col_data.value_counts()
            
            if len(value_counts) > 0:
                # 확률 계산
                probabilities = value_counts / len(col_data)
                
                # 엔트로피 계산: H = -Σ p_i * log2(p_i)
                entropy = 0
                for p in probabilities:
                    if p > 0:  # log(0) 방지
                        entropy -= p * np.log2(p)
                
                # 최대 엔트로피 (모든 값이 동일 확률일 때)
                max_entropy = np.log2(len(value_counts))
                
                # 정규화된 엔트로피 (0~1 범위)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                return {
                    '엔트로피 (H)': f"{entropy:.4f} bits",
                    '최대 가능 엔트로피': f"{max_entropy:.4f} bits",
                    '정규화 엔트로피': f"{normalized_entropy:.4f}",
                    '고유값 수': f"{len(value_counts)}개",
                    '해석': self._interpret_entropy(normalized_entropy)
                }
            else:
                return {'오류': '데이터가 없습니다'}
    

    
        # 상위카테고리
        elif stat_type == "상위카테고리":
            value_counts = col_data.value_counts()
            total = len(col_data)
            top10 = value_counts.head(10)
            result = {}
            for val, count in top10.items():
                result[f"{str(val)[:30]}"] = f"{count:,} ({count/total*100:.1f}%)"
            result['상위 10개 합계'] = f"{top10.sum():,} ({top10.sum()/total*100:.1f}%)"
            return result
        
        # 희소카테고리
        elif stat_type == "희소카테고리":
            value_counts = col_data.value_counts()
            rare = value_counts[value_counts <= 5]
            return {
                '희소값 개수': f"{len(rare)}개",
                '1회 출현': f"{(value_counts == 1).sum()}개",
                '5회 이하': f"{len(rare)}개",
                '희소값 비율': f"{len(rare)/len(value_counts)*100:.1f}%"
            }
        
        # 이진분포
        elif stat_type == "이진분포":
            value_counts = col_data.value_counts()
            if len(value_counts) >= 2:
                val1, val2 = value_counts.index[:2]
                return {
                    f'{val1}': f"{value_counts[val1]:,} ({value_counts[val1]/len(col_data)*100:.1f}%)",
                    f'{val2}': f"{value_counts.get(val2, 0):,} ({value_counts.get(val2, 0)/len(col_data)*100:.1f}%)"
                }
            else:
                return {'결과': '이진 데이터가 아닙니다'}
        
        # 비율분석
        elif stat_type == "비율분석":
            value_counts = col_data.value_counts()
            return {
                '최빈값': f"{value_counts.index[0]}",
                '최빈값 비율': f"{value_counts.iloc[0]/len(col_data)*100:.1f}%",
                '고유값 수': f"{len(value_counts)}개"
            }
        
        # 순서통계
        elif stat_type == "순서통계":
            try:
                # 순서형 데이터를 숫자로 변환 시도
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    return {
                        '최소값': f"{numeric_data.min():.0f}",
                        '최대값': f"{numeric_data.max():.0f}",
                        '중앙값': f"{numeric_data.median():.0f}",
                        '범위': f"{numeric_data.max() - numeric_data.min():.0f}"
                    }
            except:
                pass
            
            # 숫자 변환 실패시 문자열로 처리
            sorted_values = sorted(col_data.unique())
            return {
                '최소값': str(sorted_values[0]),
                '최대값': str(sorted_values[-1]),
                '고유값 수': f"{len(sorted_values)}개"
            }
        
        # 텍스트통계
        elif stat_type == "텍스트통계":
            text_series = col_data.astype(str)
            lengths = text_series.str.len()
            return {
                '평균 길이': f"{lengths.mean():.1f}",
                '최대 길이': f"{lengths.max()}",
                '최소 길이': f"{lengths.min()}",
                '길이 표준편차': f"{lengths.std():.1f}"
            }
        
        # 패턴분석
        elif stat_type == "패턴분석":
            text_series = col_data.astype(str)
            return {
                '숫자 포함': f"{text_series.str.contains(r'\d', na=False).sum():,}개",
                '영문 포함': f"{text_series.str.contains(r'[a-zA-Z]', na=False).sum():,}개",
                '한글 포함': f"{text_series.str.contains(r'[가-힣]', na=False).sum():,}개",
                '특수문자 포함': f"{text_series.str.contains(r'[^a-zA-Z0-9가-힣\s]', na=False).sum():,}개"
            }
        
        # 단어통계
        elif stat_type == "단어통계":
            text_series = col_data.astype(str)
            word_counts = text_series.str.split().str.len()
            return {
                '평균 단어 수': f"{word_counts.mean():.1f}",
                '최대 단어 수': f"{word_counts.max()}",
                '최소 단어 수': f"{word_counts.min()}",
                '총 단어 수': f"{word_counts.sum():,}"
            }
        
        # 시간범위
        elif stat_type == "시간범위":
            try:
                datetime_data = pd.to_datetime(col_data)
                return {
                    '시작일': str(datetime_data.min()),
                    '종료일': str(datetime_data.max()),
                    '기간': str(datetime_data.max() - datetime_data.min()),
                    '고유일수': f"{datetime_data.dt.date.nunique():,}"
                }
            except:
                return {'오류': '날짜 형식 변환 실패'}
        
        # 주기분석
        elif stat_type == "주기분석":
            try:
                datetime_data = pd.to_datetime(col_data)
                return {
                    '최빈 요일': datetime_data.dt.day_name().mode()[0] if not datetime_data.dt.day_name().mode().empty else "N/A",
                    '최빈 월': f"{datetime_data.dt.month.mode()[0]}월" if not datetime_data.dt.month.mode().empty else "N/A",
                    '주말 비율': f"{(datetime_data.dt.dayofweek >= 5).sum() / len(datetime_data) * 100:.1f}%"
                }
            except:
                return {'오류': '날짜 형식 변환 실패'}
        
        # 시간간격통계
        elif stat_type == "시간간격통계":
            try:
                datetime_data = pd.to_datetime(col_data).sort_values()
                intervals = datetime_data.diff().dropna()
                return {
                    '평균 간격': str(intervals.mean()),
                    '최소 간격': str(intervals.min()),
                    '최대 간격': str(intervals.max()),
                    '간격 표준편차': str(intervals.std())
                }
            except:
                return {'오류': '날짜 형식 변환 실패'}
        
        return {'오류': '알 수 없는 통계 타입'}
    
    def create_visualization(self, df: pd.DataFrame, column: str, viz_type: str, params: Dict) -> plt.Figure:
        """선택된 시각화 생성"""
        col_data = df[column].dropna()
        
        # 이미지 크기를 작게 조정
        fig, ax = plt.subplots(figsize=(6, 4))
        
        try:
            # 히스토그램
            if viz_type == "히스토그램":
                bins = params.get('bins', 30)
                col_data.hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel(column)
                ax.set_ylabel('빈도')
                ax.set_title(f'{column} 히스토그램')
            
            # 박스플롯
            elif viz_type == "박스플롯":
                ax.boxplot(col_data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
                ax.set_ylabel(column)
                ax.set_title(f'{column} 박스플롯')
            
            # 밀도플롯
            elif viz_type == "밀도플롯":
                col_data.plot.density(ax=ax, color='blue')
                ax.set_xlabel(column)
                ax.set_ylabel('밀도')
                ax.set_title(f'{column} 밀도플롯')
            
            # 바이올린플롯
            elif viz_type == "바이올린플롯":
                parts = ax.violinplot([col_data], positions=[1], showmeans=True, showmedians=True)
                ax.set_xticks([1])
                ax.set_xticklabels([column])
                ax.set_ylabel('값')
                ax.set_title(f'{column} 바이올린플롯')
            
                # 산점도 수정
            elif viz_type == "산점도":
                if 'other_column' in params:
                    other_col = params['other_column']
                    
                    # 두 열의 데이터 가져오기 (전처리된 데이터 우선 사용)
                    if hasattr(self, 'get_processed_data'):
                        x_data = self.get_processed_data(df, column)
                        y_data = self.get_processed_data(df, other_col)
                    else:
                        x_data = df[column]
                        y_data = df[other_col]
                    
                    # 결측값 제거
                    valid_idx = x_data.notna() & y_data.notna()
                    x_data = x_data[valid_idx]
                    y_data = y_data[valid_idx]
                    
                    ax.scatter(x_data, y_data, alpha=0.6, s=30)
                    ax.set_xlabel(column)
                    ax.set_ylabel(other_col)
                    ax.set_title(f'{column} vs {other_col} 산점도')
                    
                    # 상관계수 표시
                    if len(x_data) > 1:
                        corr = x_data.corr(y_data)
                        ax.text(0.05, 0.95, f'상관계수: {corr:.3f}', 
                            transform=ax.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    # 회귀선 옵션
                    if params.get('show_regression', False):
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        ax.plot(sorted(x_data), p(sorted(x_data)), "r--", alpha=0.8, linewidth=2)
                else:
                    # 인덱스를 x축으로 사용하는 산점도
                    ax.scatter(range(len(col_data)), col_data, alpha=0.6, s=30)
                    ax.set_xlabel('인덱스')
                    ax.set_ylabel(column)
                    ax.set_title(f'{column} 산점도')

            
            # 막대그래프
            elif viz_type == "막대그래프":
                value_counts = col_data.value_counts()
                if len(value_counts) > 30:
                    value_counts = value_counts.head(30)
                value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel(column)
                ax.set_ylabel('빈도')
                ax.set_title(f'{column} 막대그래프')
                plt.xticks(rotation=45, ha='right')
            
            # 가로막대그래프
            elif viz_type == "가로막대그래프":
                value_counts = col_data.value_counts()
                if len(value_counts) > 30:
                    value_counts = value_counts.head(30)
                value_counts.plot(kind='barh', ax=ax, color='lightgreen', edgecolor='black')
                ax.set_xlabel('빈도')
                ax.set_ylabel(column)
                ax.set_title(f'{column} 가로막대그래프')
            
            # 상위N막대그래프
            elif viz_type == "상위N막대그래프":
                top_n = params.get('top_n', 20)
                value_counts = col_data.value_counts().head(top_n)
                value_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
                ax.set_xlabel(column)
                ax.set_ylabel('빈도')
                ax.set_title(f'{column} 상위 {top_n}개')
                plt.xticks(rotation=45, ha='right')
            
            # 파이차트
            elif viz_type == "파이차트":
                value_counts = col_data.value_counts()
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{column} 파이차트')
            
            # 도넛차트
            elif viz_type == "도넛차트":
                value_counts = col_data.value_counts()
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                                  autopct='%1.1f%%', startangle=90)
                centre_circle = plt.Circle((0,0), 0.70, fc='white')
                ax.add_artist(centre_circle)
                ax.set_title(f'{column} 도넛차트')
            
            # 파레토차트
            elif viz_type == "파레토차트":
                top_n = params.get('top_n', 20)
                value_counts = col_data.value_counts().head(top_n)
                cumsum = value_counts.cumsum() / value_counts.sum() * 100
                
                ax.bar(range(len(value_counts)), value_counts.values, color='skyblue', edgecolor='black')
                ax.set_ylabel('빈도', color='blue')
                
                ax2 = ax.twinx()
                ax2.plot(range(len(value_counts)), cumsum.values, 'ro-', linewidth=2)
                ax2.set_ylabel('누적 비율 (%)', color='red')
                ax2.set_ylim(0, 105)
                
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels([str(x)[:15] for x in value_counts.index], rotation=45, ha='right')
                ax.set_title(f'{column} 파레토차트')
            
            # 단어빈도
            elif viz_type == "단어빈도":
                words = ' '.join(col_data.astype(str)).split()
                word_freq = pd.Series(words).value_counts().head(20)
                word_freq.plot(kind='bar', ax=ax, color='purple', edgecolor='black')
                ax.set_xlabel('단어')
                ax.set_ylabel('빈도')
                ax.set_title(f'{column} 단어 빈도 (상위 20개)')
                plt.xticks(rotation=45, ha='right')
            
            # 길이분포
            elif viz_type == "길이분포":
                lengths = col_data.astype(str).str.len()
                lengths.hist(bins=30, ax=ax, edgecolor='black', color='green', alpha=0.7)
                ax.set_xlabel('문자열 길이')
                ax.set_ylabel('빈도')
                ax.set_title(f'{column} 문자열 길이 분포')
            
            # 시계열그래프
            elif viz_type == "시계열그래프":
                try:
                    datetime_data = pd.to_datetime(col_data)
                    datetime_data.value_counts().sort_index().plot(ax=ax, color='navy')
                    ax.set_xlabel('날짜')
                    ax.set_ylabel('빈도')
                    ax.set_title(f'{column} 시계열 분포')
                    plt.xticks(rotation=45)
                except:
                    ax.text(0.5, 0.5, '날짜 형식 변환 실패', ha='center', va='center')
            
            # 월별분포
            elif viz_type == "월별분포":
                try:
                    datetime_data = pd.to_datetime(col_data)
                    month_counts = datetime_data.dt.month.value_counts().sort_index()
                    month_counts.plot(kind='bar', ax=ax, color='orange')
                    ax.set_xlabel('월')
                    ax.set_ylabel('빈도')
                    ax.set_title(f'{column} 월별 분포')
                    ax.set_xticklabels([f'{i}월' for i in month_counts.index])
                except:
                    ax.text(0.5, 0.5, '날짜 형식 변환 실패', ha='center', va='center')
            
            # 요일분포
            elif viz_type == "요일분포":
                try:
                    datetime_data = pd.to_datetime(col_data)
                    weekday_counts = datetime_data.dt.dayofweek.value_counts().sort_index()
                    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
                    weekday_counts.plot(kind='bar', ax=ax, color='teal')
                    ax.set_xlabel('요일')
                    ax.set_ylabel('빈도')
                    ax.set_title(f'{column} 요일별 분포')
                    ax.set_xticklabels([weekday_names[i] for i in weekday_counts.index])
                except:
                    ax.text(0.5, 0.5, '날짜 형식 변환 실패', ha='center', va='center')
            
            # 시간대분포
            elif viz_type == "시간대분포":
                try:
                    datetime_data = pd.to_datetime(col_data)
                    hour_counts = datetime_data.dt.hour.value_counts().sort_index()
                    hour_counts.plot(kind='bar', ax=ax, color='brown')
                    ax.set_xlabel('시간')
                    ax.set_ylabel('빈도')
                    ax.set_title(f'{column} 시간대별 분포')
                    ax.set_xticklabels([f'{i}시' for i in hour_counts.index])
                except:
                    ax.text(0.5, 0.5, '날짜 형식 변환 실패', ha='center', va='center')
            
        except Exception as e:
            plt.close(fig)
            raise Exception(f"시각화 생성 실패: {str(e)}")
        
        plt.tight_layout()
        return fig
    
    def create_dtype_distribution_chart(self, df: pd.DataFrame):
        """데이터 타입 분포 차트 생성"""
        dtype_counts = df.dtypes.value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Set3(range(len(dtype_counts)))
        bars = dtype_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        
        # 막대 위에 값 표시
        for i, (index, value) in enumerate(dtype_counts.items()):
            ax.text(i, value + 0.1, str(value), ha='center', va='bottom')
        
        ax.set_title("열별 데이터 타입 분포")
        ax.set_xlabel("데이터 타입")
        ax.set_ylabel("개수")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def _interpret_entropy(self, normalized_entropy: float) -> str:
        """정규화된 엔트로피 값 해석"""
        if normalized_entropy < 0.3:
            return "매우 낮음 (데이터가 특정 값에 집중)"
        elif normalized_entropy < 0.5:
            return "낮음 (일부 값이 우세)"
        elif normalized_entropy < 0.7:
            return "중간 (적당한 다양성)"
        elif normalized_entropy < 0.9:
            return "높음 (값이 고르게 분포)"
        else:
            return "매우 높음 (거의 균등 분포)"
