import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque

def process_data(inputfile, outputfile, alpha = 0.0001):
    # calculate the micro price for each level, and add it as a new column to the dataframe
    def price_micro(df, level: int):
        price_cols = [f'bid{level}', f'ask{level}']
        size_cols = [f'bidSize{level}', f'askSize{level}']   
        df[f'price_micro_{level}'] = (df[price_cols[0]] * df[size_cols[1]] + df[price_cols[1]] * df[size_cols[0]]) / (df[size_cols[0]] + df[size_cols[1]])
        return None
    
    # calculate the price move label 
    def move_label(x, alpha):
        if x > alpha:
            return 1
        elif x < -alpha:
            return -1
        else:
            return 0
    
    # read the csv file, only keep the timestamp and the 20 levels of price and volume
    df = pd.read_csv(inputfile, usecols = [0] + list(range(5,25)))

    # Caculate the mid price
    df['MidPrice'] = (df['bid1'] + df['ask1']) / 2

    # calculate the micro price for each level
    for i in range(1,6):
        price_micro(df, level = i)
    
    # select the timestamp, mid price and micro price columns
    df_selected = pd.concat([df.iloc[:, [0]], df.iloc[:, -6:]], axis=1)

    # calculate the price move and price move percentage
    m_plus_t = df_selected['MidPrice'].rolling(window=10).mean().shift(-10)
    df_selected['price_move'] = (m_plus_t - df_selected['MidPrice'])
    df_selected['price_move_pctg'] = df_selected['price_move'] / df_selected['MidPrice']
    
    # calculate the price move label
    df_selected['price_move_label'] = df_selected['price_move_pctg'].apply(lambda x: move_label(x, alpha))

    # drop the rows with NaN values
    df_selected.dropna(inplace=True)

    # save the processed data to a new csv file
    df_selected.to_csv(outputfile, index=False)
    
    print(f"Data processing completed. Processed data saved to {outputfile}.")

def window_normalize(inputpath, outputpath, window_size = 5):
    # read the processed csv file
    folder = Path(inputpath)
    csv_files = sorted(folder.glob('*.csv'))

    buffer = deque(maxlen=window_size + 1)

    cols_to_keep = [2, 3, 4, 5, 6, 9]
    processed_rows = 0
    
    for file in csv_files:
        print(f"Loading file: {file.name}")
        
        df = pd.read_csv(file, usecols=cols_to_keep)
        buffer.append(df)
        processing_date = file.stem[-4:]

        if len(buffer) == window_size + 1:
            stats_list = list(buffer)[:-1]
            stats_df = pd.concat(stats_list, ignore_index=True)

            target_df = buffer[-1]
            feature_cols = [col for col in target_df.columns if col.startswith('price_micro')]
            processed_rows += len(target_df)

            hist_mean = stats_df[feature_cols].mean()
            hist_std = stats_df[feature_cols].std()

            normalized_df = target_df.copy()
            normalized_df[feature_cols] = (normalized_df[feature_cols] - hist_mean) / hist_std

            file_name = f'normalized_{processing_date}'
            normalized_df.to_csv(f'{outputpath}{file_name}.csv', index=False)
            print(f'Data saved to {outputpath}{file_name}.csv')
    print(f"Window normalization completed. Total processed rows: {processed_rows}.")

if __name__ == "__main__":

    inputpath = 'Data/Raw_data/'
    outputpath = 'Data/Processed_data/'
    path = Path(outputpath)
    path.mkdir(parents=True, exist_ok=True)
    alpha = 0.00005

    raw_files = list(Path(inputpath).glob('*.csv'))

    for raw_file in raw_files:
        print(f'Processing file:{raw_file.name}')
        outputfile = f'{outputpath}processed_{raw_file.stem[-4:]}.csv'
        process_data(raw_file, outputfile, alpha)

    inputpath_normalized = 'Data/Processed_data/'
    outputpath_normalized = 'Data/Normalized_data/' 
    path = Path(outputpath_normalized)
    path.mkdir(parents=True, exist_ok=True)
    window_normalize(inputpath_normalized, outputpath_normalized, window_size = 5)