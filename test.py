import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("ML System Test App")
    # Basic test to ensure Streamlit and dependencies work
    st.write("Dependencies are working!")

    # Simple data generation
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })

    st.write("Random Data Preview:")
    st.dataframe(data.head())

    # Simple plot
    st.line_chart(data)

if __name__ == "__main__":
    main()