import streamlit as st

st.title("Streamlit Web-App Prototype")
"""
## João Lousada, Lisboa, PT, Sep 2020
---
"""
st.header("Pre-Commit Engine ")

st.markdown("We determine which tests cases are more likely to affect the modified files in your commit")

options = st.multiselect(
        'What files were modified?',
        ['File1', 'File2', 'File3', 'File4'],
        )

str1 = ', '
st.write('You selected:', str1.join(options))

