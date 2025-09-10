import pandas as pd
import numpy as np

def create_dummy_groundwater_data():
    """
    Create realistic dummy groundwater data with proper column structure
    """
    
    # Sample data
    states = ['Rajasthan', 'Telangana', 'Maharashtra', 'Karnataka', 'Gujarat', 'Tamil Nadu', 'Uttar Pradesh']
    districts = ['Jaipur', 'Hyderabad', 'Mumbai', 'Bangalore', 'Ahmedabad', 'Pune', 'Nagpur', 'Chennai', 'Lucknow']
    blocks = ['Block_A', 'Block_B', 'Block_C', 'Block_D', 'Block_E', 'Block_F']
    years = [2020, 2021, 2022, 2023, 2024]
    categories = ['Safe', 'Semi-Critical', 'Critical', 'Over-Exploited']
    
    # Create sample data with proper column names
    data = {
        'S_NO': range(1, 101),
        'STATE': np.random.choice(states, 100),
        'DISTRICT': np.random.choice(districts, 100),
        'ASSESSMENT_UNIT': np.random.choice(blocks, 100),
        'YEAR': np.random.choice(years, 100),
        'Rainfall_mm': np.random.uniform(500, 1500, 100).round(1),
        'Total_Geographical_Area_ha': np.random.uniform(10000, 500000, 100).round(0),
        'Total_Annual_Recharge_ham': np.random.uniform(1000, 50000, 100).round(2),
        'Net_Ground_Water_Availability_ham': np.random.uniform(800, 45000, 100).round(2),
        'Annual_Extractable_Ground_Water_Resource_ham': np.random.uniform(700, 40000, 100).round(2),
        'Ground_Water_Extraction_Domestic_ham': np.random.uniform(100, 5000, 100).round(2),
        'Ground_Water_Extraction_Industrial_ham': np.random.uniform(50, 3000, 100).round(2),
        'Ground_Water_Extraction_Irrigation_ham': np.random.uniform(500, 35000, 100).round(2),
        'Total_Ground_Water_Extraction_ham': np.random.uniform(600, 40000, 100).round(2),
        'Stage_of_Ground_Water_Extraction_pct': np.random.uniform(40, 150, 100).round(1),
        'Category': np.random.choice(categories, 100),
        'Net_Annual_Ground_Water_Availability_Future_Use_ham': np.random.uniform(100, 10000, 100).round(2),
        'Monsoon_Recharge_ham': np.random.uniform(500, 30000, 100).round(2),
        'Non_Monsoon_Recharge_ham': np.random.uniform(200, 15000, 100).round(2),
        'Recharge_From_Canals_ham': np.random.uniform(50, 5000, 100).round(2),
        'Recharge_From_Tanks_ham': np.random.uniform(30, 3000, 100).round(2),
        'Population': np.random.randint(10000, 5000000, 100),
        'Agricultural_Area_ha': np.random.uniform(5000, 200000, 100).round(0),
        'Water_Table_Depth_m': np.random.uniform(5, 50, 100).round(1),
        'Aquifer_Type': np.random.choice(['Alluvial', 'Basalt', 'Granite', 'Sandstone'], 100),
        'Water_Quality_Index': np.random.uniform(50, 95, 100).round(1)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add calculated columns
    df['Extraction_Utilization_Ratio'] = (df['Total_Ground_Water_Extraction_ham'] / df['Annual_Extractable_Ground_Water_Resource_ham']).round(3)
    df['Recharge_Extraction_Ratio'] = (df['Total_Annual_Recharge_ham'] / df['Total_Ground_Water_Extraction_ham']).round(3)
    df['Per_Capita_Water_Availability'] = (df['Net_Ground_Water_Availability_ham'] / df['Population'] * 1000).round(2)
    
    return df

def save_dummy_data():
    """
    Create and save dummy groundwater data
    """
    
    print("Creating dummy groundwater data...")
    
    # Create the dummy data
    df = create_dummy_groundwater_data()
    
    # Save to CSV
    output_file = "dummy_groundwater_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Dummy data saved to: {output_file}")
    print(f"Data shape: {df.shape}")
    
    # Show column information
    print("\nCOLUMNS CREATED:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
        print(f"{i+1:2d}. {col:<40} {str(dtype):<10} {sample}")
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Show first few rows
    print("\nFIRST 3 ROWS:")
    print(df.head(3).to_string())
    
    return df

# Update your backend to use this dummy data
def update_backend_to_use_dummy_data():
    """
    Instructions to update your backend
    """
    print("\n" + "="*60)
    print("TO USE THIS DUMMY DATA IN YOUR BACKEND:")
    print("="*60)
    print("1. Replace your CANDIDATE_DATA_PATHS with:")
    print("   CANDIDATE_DATA_PATHS = ['dummy_groundwater_data.csv']")
    print("2. Update your ALIAS_HINTS to match these columns:")
    print("   ALIAS_HINTS = {")
    print("       'recharge': ['Total_Annual_Recharge_ham', 'Monsoon_Recharge_ham'],")
    print("       'extraction': ['Total_Ground_Water_Extraction_ham', 'Ground_Water_Extraction'],")
    print("       'stage': ['Stage_of_Ground_Water_Extraction_pct'],")
    print("       'availability': ['Net_Ground_Water_Availability_ham'],")
    print("       'state': ['STATE'],")
    print("       'district': ['DISTRICT'],")
    print("       'year': ['YEAR']")
    print("   }")
    print("3. Restart your backend")
    print("="*60)

# Run the script
if __name__ == "__main__":
    try:
        df = save_dummy_data()
        update_backend_to_use_dummy_data()
        
        print("\nYour chatbot will now work with queries like:")
        print("  - 'Show recharge trend for Rajasthan'")
        print("  - 'Compare districts by extraction in 2023'")
        print("  - 'Stage distribution in Telangana'")
        print("  - 'Forecast next year extraction for Jaipur'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()