import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from pipeline import Regressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def main() -> None:
    hide_st_styles = '''
        <style>
        footer {visibility: hidden;}
        
        </style>
    '''
    st.markdown(hide_st_styles, unsafe_allow_html=True)

    st.markdown('# Model Builder')

    tabs = st.tabs(
        ['Import Data', 'Data Viewer', 'Statistics', 'Train/Test', 'Visualize']
    )

    # Data Importing Page
    with tabs[0]:
        data = st.file_uploader('Upload a file')
        if data:
            df = pd.read_csv(data)

            my_model = Regressor(df)

    # DataViewer Page
            with tabs[1]:
                st.markdown('### DataViewer')
                
                cols = st.columns(4)
                with cols[0]:
                    drop_column = st.checkbox('Drop Column')
                    if drop_column:
                        column_name = st.multiselect('columns', my_model.df.columns)

                        my_model.drop_column(column_name)

                with cols[1]:
                    drop_null_ = st.checkbox('Drop Null Rows')
                    if drop_null_:
                        my_model.drop_null()

                with cols[2]:
                    drop_duplicates_ = st.checkbox('Remove Duplicates')
                    if drop_duplicates_:
                        my_model.drop_duplicates()

                with cols[3]:
                    replace_ = st.checkbox('Replace Value in Data')
                    
                    if replace_:
                        column_ = st.selectbox('Select Column to Alter', my_model.df.columns)
                        old_val = st.text_input('Old Value')
                        new_val = st.text_input('New Value')
                        
                        my_model.df[column_] = my_model.df[column_].apply(
                            lambda x: str(x).replace(old_val, new_val)
                        )

                with st.container():    
                    st.dataframe(my_model.df, use_container_width=True)

            # Statistics Page
            with tabs[2]:
                st.markdown('### Statistics')

                table = st.selectbox(
                    'Statistics Table', 
                    ['Describe', 'DataTypes', 'Correlation', 'Value Counts']
                    )

                with st.container():
                    if table.lower() == 'describe':
                        st.dataframe(
                            my_model.df.describe().transpose(), 
                            use_container_width=True
                        )

                    if table.lower() == 'datatypes':
                        st.dataframe(my_model.df.dtypes, 
                        use_container_width=True
                    )

                    if table.lower() == 'correlation':
                        st.dataframe(
                            my_model.df.corr(), 
                            use_container_width=True
                        )

                    if table.lower() == 'value counts':
                        col = st.selectbox('Column', my_model.df.columns)
                        st.dataframe(
                            my_model.df[col].value_counts(), 
                            use_container_width=True
                        )

            # Training/Testing Page
            try:
                with tabs[3]:
                    st.markdown('### Train Model')
                    target = st.selectbox('Pick a Target Column', my_model.df.columns)

                    models = [LinearRegression, RandomForestRegressor]
                    pick_model = st.selectbox('Pick Model', models)
                    my_model.data_split(target)
                    my_model.train(pick_model)
                    st.markdown('#### Model is trained')

                    st.markdown('### Test Model')
                    my_model.test()
                    st.write(my_model.score)

                with tabs[4]:
                    st.markdown('### Visualize Predictions')

                    fig, ax = plt.subplots()
                    ax.scatter(
                        x=my_model.test_, 
                        y=my_model.test_target
                    )
                    ax.plot(
                        my_model.test_,
                        my_model.predictions)
                    st.pyplot(fig)

            except ValueError:
                pass
        
        else:
            for tab in tabs[1:]:
                with tab:
                    st.markdown('First upload data')


if __name__ == '__main__':
    main()