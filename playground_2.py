import requests
import pandas as pd

# API endpoint
#url = "https://eodhd.com/api/fundamentals/GSPC.INDX?api_token=65ef60f366dea7.61773890&fmt=json"
#url = "https://eodhd.com/api/fundamentals/NDX.INDX?api_token=65ef60f366dea7.61773890&fmt=json"
url = "https://eodhd.com/api/fundamentals/DJI.INDX?api_token=65ef60f366dea7.61773890&fmt=json"

# Fetch the data from the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse JSON data
    api_response = response.json()

    # Extract Components data
    components = api_response.get("Components", {})

    # Convert to list of dictionaries
    components_list = [v for k, v in components.items()]

    # Create a DataFrame
    df = pd.DataFrame(components_list)

    # Save the DataFrame to a CSV file
    csv_file_name = "components_table.csv"
    df.to_csv(csv_file_name, index=False)

    print(f"Data successfully saved to {csv_file_name}")
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
