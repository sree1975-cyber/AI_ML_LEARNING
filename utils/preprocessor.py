def preprocess_data(df):
    # Calculate target
    df['CA_Status'] = (df['Attendance_Percentage'] <= 0.9).astype(int)
    
    # Handle missing student IDs
    if 'Student_ID' not in df.columns:
        df['Student_ID'] = df.index
    
    # One-hot encode
    categoricals = ['Grade', 'Gender', 'Meal_Code']
    df = pd.get_dummies(df, columns=categoricals)
    
    return df
