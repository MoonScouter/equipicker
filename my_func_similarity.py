
import pandas as pd
import time
import json


def similar_companies(similarity_matrix_csv, symbol: str = None, sectors: list = None,
                      limit: int = 5, mkcap_category: list = None, threshold: float = 0.95):
    # Start time
    start_time_function = time.time()  # Start time for the function

    # Load the stock data
    if 'df_stock' not in globals():
        try:
            global df_stock
            df_stock = pd.read_csv('us_stock_data.csv')
            df_stock.drop(df_stock[df_stock['MarketCapCategory'] == 'Unknown'].index, inplace=True)
            df_stock.reset_index(drop=True, inplace=True)
        except FileNotFoundError:
            print("File containing us stock data not found. Please check the file name and path.")

    # Load the sector data
    if 'sector_data' not in globals():
        try:
            global sector_data
            sector_data = pd.read_csv('pmi_sectors.csv')
            sector_data.reset_index(drop=True, inplace=True)
        except FileNotFoundError:
            print("File containing sector data not found. Please check the file name and path.")


    # Handling of case when symbol is provided. This is specific for when similar companies are sought
    if symbol is not None:
        # Load similarity matrix
        if 'similarity_df' not in globals():
            try:
                global similarity_df
                similarity_df = pd.read_csv(similarity_matrix_csv, header=None)
                similarity_df.set_index(0, inplace=True)
                similarity_df.columns = similarity_df.iloc[0]
                similarity_df = similarity_df.iloc[1:]
            except FileNotFoundError:
                print("File containing similarity stock data not found. Please check the file name and path.")

        # Find the index corresponding to the symbol
        try:
            symbol_index = df_stock[df_stock['Code'] == symbol].index[0]
            # print(symbol_index)
        except IndexError:
            print(f"Symbol {symbol} not found in stock data.")

        # Filter by market cap category if provided and get relevant similarity scores
        if mkcap_category is not None:
            mkcap_indices = df_stock.loc[df_stock['MarketCapCategory'].isin(mkcap_category)].index
            similarity_scores = similarity_df.iloc[mkcap_indices, symbol_index]
        else:
            similarity_scores = similarity_df.iloc[:, symbol_index]


        # Filter scores by the threshold and get the top similar indices, while excluding the symbol itself
        filtered_scores = similarity_scores[similarity_scores >= threshold]
        top_similar_indices = filtered_scores.nlargest(limit + 1).index

        # Create a dictionary from filtered_scores for easy mapping
        scores_dict = filtered_scores.to_dict()

        # Get company data for the similar companies
        similar_companies = df_stock.loc[top_similar_indices].copy()

        # Map the similarity scores to the similar companies
        similar_companies['SimilarityScore'] = similar_companies.index.map(scores_dict)
        similar_companies_result = similar_companies.to_json(orient='index')
        similar_companies_dict = json.loads(similar_companies_result)

        end_time_function = time.time()  # End time for the function
        print(f"Function processed in {end_time_function - start_time_function:.2f} seconds.")

        return similar_companies_dict

    # Handling of case when sector is provided. This is specific for when companies belonging to sectors are sought
    elif sectors is not None:
        result_dict = {}  # Dictionary to store the result for each sector
        # Load similarity matrix
        if 'similarity_sector_df' not in globals():
            try:
                global similarity_sector_df
                similarity_sector_df = pd.read_csv(similarity_matrix_csv, header=None)
                similarity_sector_df.set_index(0, inplace=True)
                similarity_sector_df.columns = similarity_sector_df.iloc[0]
                similarity_sector_df = similarity_sector_df.iloc[1:]
            except FileNotFoundError:
                print("File containing similarity sector data not found. Please check the file name and path.")

        for sector in sectors:
            # Find the sector index
            try:
                sector_index = sector_data[sector_data['Sector'] == sector].index[0]
            except IndexError:
                print(f"Sector {sector} not found in sector data.")
                continue

            # Logic for processing by sector - similar to processing by symbol

            # Filter by market cap category if provided and get relevant similarity scores
            if mkcap_category is not None:
                mkcap_indices = df_stock.loc[df_stock['MarketCapCategory'].isin(mkcap_category)].index
                similarity_scores = similarity_sector_df.iloc[mkcap_indices, sector_index]
            else:
                similarity_scores = similarity_sector_df.iloc[:, sector_index]

            # Filter scores by the threshold and get the top similar indices, while excluding the symbol itself
            filtered_scores = similarity_scores[similarity_scores >= threshold]
            top_similar_indices = filtered_scores.nlargest(limit).index

            # Create a dictionary from filtered_scores for easy mapping
            scores_dict = filtered_scores.to_dict()

            # Get company data for the similar companies
            similar_companies = df_stock.loc[top_similar_indices].copy()

            # Map the similarity scores to the similar companies
            similar_companies['SimilarityScore'] = similar_companies.index.map(scores_dict)

            similar_companies_result = similar_companies.to_json(orient='index')
            similar_companies_dict = json.loads(similar_companies_result)

            result_dict[sector] = similar_companies_dict

        end_time_function = time.time()  # End time for the function
        print(f"Function processed in {end_time_function - start_time_function:.2f} seconds.")

        return result_dict


# Example usage
# similarity_matrix_combined_csv = 'similarity_combined.csv'  # Replace with your actual similarity matrix CSV file path
# similarity_matrix_combined_simple_csv = 'similarity_combined_simple.csv'
# similarity_matrix_sector_csv = 'similarity_sector.csv'
# similarity_matrix_sector_simple_csv = 'similarity_sector_simple.csv'
# symbol = 'UBER'  # Replace with the symbol of interest
# sectors = ['Computer & Electronic Products', 'Transportation & Warehousing', 'Transportation Equipment',
#           'Apparel, Leather & Allied Products']
# limit = 10  # Number of similar companies to retrieve
#
# similar_companies_data_combined = similar_companies(similarity_matrix_csv=similarity_matrix_combined_csv, symbol=symbol,
#                                            limit=limit, mkcap_category=['Mid', 'Large', 'Mega'],
#                                            threshold=0.9)
#
# similar_companies_data_combined_simple = similar_companies(similarity_matrix_csv=similarity_matrix_combined_simple_csv, symbol=symbol,
#                                            limit=limit, mkcap_category=[ 'Large', 'Mega'],
#                                            threshold=0.9)
#
# similar_sector_data_combined = similar_companies(similarity_matrix_csv=similarity_matrix_sector_csv, sectors=sectors,
#                                            limit=limit, mkcap_category=[ 'Mega'],
#                                            threshold=0.5)
# print(similar_sector_data_combined)
# similar_sector_data_combined_simple = similar_companies(similarity_matrix_csv=similarity_matrix_sector_simple_csv, sectors=sectors,
#                                            limit=limit, mkcap_category=['Mid', 'Large', 'Mega'],
#                                            threshold=0.5)






